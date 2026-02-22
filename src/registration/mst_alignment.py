"""
MST-Based Hierarchical Alignment for character clusters.

Given a cluster of characters, builds a minimum spanning tree of
pairwise alignment residuals and propagates transformations from a
root (medoid) to all leaves.  This avoids the blurry averaging of
the Fréchet mean and bounds alignment drift to O(log m) for
well-clustered sets.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Dict, Any, List, Tuple
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from collections import deque

from .multiscale_registration import MultiscaleIC
from .transformations import PlanarTransform
from .single_scale import InverseCompositional
from .gaussian_pyramid import GaussianPyramid
from .gradients import Gradients


def _default_aligner(device: str = 'cuda') -> MultiscaleIC:
    """Create a default multiscale IC aligner."""
    gradient_method = Gradients(method='farid5', C=1, device=device)
    ic = InverseCompositional(
        transform_type='homography',
        gradient_method=gradient_method,
        error_function='lorentzian',
        delta=5,
        epsilon=1e-3,
        max_iter=5,
    )
    pyramid = GaussianPyramid(
        eta=0.5, sigma_0=0.6, ksize_factor=8, min_size=32,
    )
    return MultiscaleIC(singleScaleIC=ic, gaussianPyramid=pyramid)


class MSTAlignment:
    """
    MST-based hierarchical alignment for a cluster of character images.

    Algorithm:
        1. Build alignment graph (k-NN by NLFA or full pairwise)
        2. Compute pairwise alignment residuals as edge weights
        3. Build MST of this graph
        4. Root at cluster medoid
        5. Propagate transformations root → leaves (BFS)
        6. Optional: refine by aligning to pixel-wise median template

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbors for the alignment graph.
        If None, use all pairs (expensive for large clusters).
    n_refinement_rounds : int
        Number of rounds of median-template refinement after tree propagation.
    device : str
        Torch device for registration.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        n_refinement_rounds: int = 2,
        device: str = 'cuda',
        aligner: Optional[MultiscaleIC] = None,
    ):
        self.k_neighbors = k_neighbors
        self.n_refinement_rounds = n_refinement_rounds
        self.device = device
        self.aligner = aligner or _default_aligner(device)

    def _pairwise_align(
        self, I1: torch.Tensor, I2: torch.Tensor,
    ) -> Tuple[PlanarTransform, float]:
        """Align I2 onto I1 and return (transform, residual)."""
        I1_4d = I1.unsqueeze(0).unsqueeze(0).to(self.device)
        I2_4d = I2.unsqueeze(0).unsqueeze(0).to(self.device)
        try:
            T = self.aligner.run(I1_4d, I2_4d)
            I2w = T.warp(I2_4d)
            mask = T.visibility_mask(I1_4d.shape[2], I1_4d.shape[3], delta=0)
            # Residual = mean absolute difference in visible region
            diff = (I1_4d[:, 0] - I2w[:, 0]).abs()
            diff[~mask] = 0
            n_visible = mask.float().sum()
            residual = diff.sum().item() / max(n_visible.item(), 1.0)
            return T, residual
        except Exception:
            return PlanarTransform('homography', device=self.device), float('inf')

    def _warp_image(
        self, image: torch.Tensor, transform: PlanarTransform,
    ) -> torch.Tensor:
        """Warp an image by a transform, zero-padding outside."""
        I = image.unsqueeze(0).unsqueeze(0).to(self.device)
        warped = transform.warp(I)[:, 0]
        H, W = image.shape
        mask = transform.visibility_mask(H, W, delta=0)
        warped[~mask] = 0
        return warped[0].cpu()

    def align_cluster(
        self,
        images: List[torch.Tensor],
        nlfa_row: Optional[np.ndarray] = None,
        distance_row: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Align all images in a cluster to a common coordinate frame.

        Parameters
        ----------
        images : list of Tensor (H, W)
            Character images in the cluster.
        nlfa_row : ndarray (m, m) or None
            Intra-cluster NLFA sub-matrix (for k-NN selection).
        distance_row : ndarray (m, m) or None
            Intra-cluster distance matrix (for medoid computation).

        Returns
        -------
        dict with keys:
            ``aligned`` : list of Tensor (H, W) — aligned images
            ``transforms`` : list of PlanarTransform — global transforms
            ``template`` : Tensor (H, W) — pixel-wise median template
            ``medoid_idx`` : int — root character index
        """
        m = len(images)
        if m == 0:
            return {'aligned': [], 'transforms': [], 'template': None, 'medoid_idx': -1}
        if m == 1:
            return {
                'aligned': [images[0]],
                'transforms': [PlanarTransform('homography', device='cpu')],
                'template': images[0],
                'medoid_idx': 0,
            }

        # Step 1: Find medoid (root)
        if distance_row is not None:
            medoid_idx = int(distance_row.sum(axis=1).argmin())
        else:
            medoid_idx = 0

        # Step 2: Build alignment graph edges
        k = min(self.k_neighbors, m - 1)
        pairs = set()
        if nlfa_row is not None and k < m - 1:
            # Use k-NN from NLFA
            for i in range(m):
                row = nlfa_row[i].copy()
                row[i] = -np.inf
                nn_indices = np.argpartition(row, -k)[-k:]
                for j in nn_indices:
                    if i != j:
                        pairs.add((min(i, j), max(i, j)))
        else:
            # All pairs
            for i in range(m):
                for j in range(i + 1, m):
                    pairs.add((i, j))

        # Step 3: Compute pairwise residuals
        edge_weights = np.full((m, m), np.inf)
        edge_transforms = {}

        for (i, j) in pairs:
            T, residual = self._pairwise_align(images[i], images[j])
            edge_weights[i, j] = residual
            edge_weights[j, i] = residual
            edge_transforms[(i, j)] = T

        np.fill_diagonal(edge_weights, 0)

        # Step 4: Build MST
        W_sparse = csr_matrix(edge_weights)
        mst = minimum_spanning_tree(W_sparse)
        mst_dense = mst.toarray()

        # Step 5: BFS from medoid to propagate transforms
        global_transforms = [None] * m
        global_transforms[medoid_idx] = PlanarTransform(
            'homography', device=self.device,
        )

        visited = {medoid_idx}
        queue = deque([medoid_idx])

        # Make MST symmetric for traversal
        mst_sym = mst_dense + mst_dense.T

        while queue:
            node = queue.popleft()
            # Find children in MST
            children = np.where(mst_sym[node] > 0)[0]
            for child in children:
                if child in visited:
                    continue
                visited.add(child)

                # Get pairwise transform child -> node
                key = (min(node, child), max(node, child))
                if key in edge_transforms:
                    T_pair = edge_transforms[key]
                    # If key is (child, node), T maps node -> child; we need child -> node
                    if key[0] == child:
                        # T maps images[child] -> images[node], that's what we want
                        T_child_to_parent = T_pair
                    else:
                        # T maps images[node] -> images[child], invert
                        T_child_to_parent = T_pair.inv
                else:
                    # Compute on the fly
                    T_child_to_parent, _ = self._pairwise_align(
                        images[node], images[child],
                    )

                # Global = parent's global @ child_to_parent
                global_transforms[child] = global_transforms[node] @ T_child_to_parent
                queue.append(child)

        # Handle disconnected nodes (shouldn't happen with MST, but safety)
        for i in range(m):
            if global_transforms[i] is None:
                global_transforms[i] = PlanarTransform(
                    'homography', device=self.device,
                )

        # Step 6: Apply transforms and compute template
        aligned = []
        for i in range(m):
            if i == medoid_idx:
                aligned.append(images[i])
            else:
                aligned.append(self._warp_image(images[i], global_transforms[i]))

        template = self._compute_median_template(aligned)

        # Step 7: Refinement rounds (re-align to median template)
        for _ in range(self.n_refinement_rounds):
            new_aligned = []
            new_transforms = []
            for i in range(m):
                T, _ = self._pairwise_align(template, images[i])
                warped = self._warp_image(images[i], T)
                new_aligned.append(warped)
                new_transforms.append(T)
            aligned = new_aligned
            global_transforms = new_transforms
            template = self._compute_median_template(aligned)

        return {
            'aligned': aligned,
            'transforms': global_transforms,
            'template': template,
            'medoid_idx': medoid_idx,
        }

    @staticmethod
    def _compute_median_template(aligned: List[torch.Tensor]) -> torch.Tensor:
        """Pixel-wise median of aligned images (robust to outliers)."""
        stack = torch.stack(aligned)  # (m, H, W)
        template = stack.median(dim=0).values
        return template
