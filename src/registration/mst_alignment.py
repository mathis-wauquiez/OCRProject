"""
MST-Based Hierarchical Alignment for character clusters.

Builds a minimum spanning tree of pairwise alignment residuals and
propagates transformations from a root (medoid) to all leaves.
Bounds alignment drift to O(log m) for well-clustered sets.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Dict, Any, List, Tuple
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from collections import deque

from .transformations import PlanarTransform
from .alignment_utils import create_default_aligner, pairwise_align, warp_image, median_template


class MSTAlignment:
    """MST-based hierarchical alignment for a cluster of character images.

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbors for the alignment graph.
    n_refinement_rounds : int
        Rounds of median-template refinement after tree propagation.
    """

    def __init__(self, k_neighbors=5, n_refinement_rounds=2,
                 device='cuda', aligner=None):
        self.k_neighbors = k_neighbors
        self.n_refinement_rounds = n_refinement_rounds
        self.device = device
        self.aligner = aligner or create_default_aligner(device)

    def align_cluster(self, images, nlfa_sub=None, distance_sub=None):
        """Align all images in a cluster to a common coordinate frame.

        Returns dict with keys: aligned, transforms, template, medoid_idx.
        """
        m = len(images)
        if m <= 1:
            T_id = PlanarTransform('homography', device='cpu')
            return {
                'aligned': list(images), 'transforms': [T_id] * m,
                'template': images[0] if m else None, 'medoid_idx': 0 if m else -1,
            }

        # Medoid
        medoid = int(distance_sub.sum(axis=1).argmin()) if distance_sub is not None else 0

        # Build edge set (k-NN or all pairs)
        k = min(self.k_neighbors, m - 1)
        pairs = set()
        if nlfa_sub is not None and k < m - 1:
            for i in range(m):
                row = nlfa_sub[i].copy(); row[i] = -np.inf
                for j in np.argpartition(row, -k)[-k:]:
                    if i != j:
                        pairs.add((min(i, j), max(i, j)))
        else:
            pairs = {(i, j) for i in range(m) for j in range(i + 1, m)}

        # Pairwise residuals
        W = np.full((m, m), np.inf)
        T_map = {}
        for (i, j) in pairs:
            T, res = pairwise_align(self.aligner, images[i], images[j], self.device)
            W[i, j] = W[j, i] = res
            T_map[(i, j)] = T
        np.fill_diagonal(W, 0)

        # MST + BFS propagation
        mst = (minimum_spanning_tree(csr_matrix(W)).toarray())
        mst_sym = mst + mst.T

        global_T = [None] * m
        global_T[medoid] = PlanarTransform('homography', device=self.device)
        visited = {medoid}
        queue = deque([medoid])

        while queue:
            node = queue.popleft()
            for child in np.where(mst_sym[node] > 0)[0]:
                if child in visited:
                    continue
                visited.add(child)
                key = (min(node, child), max(node, child))
                if key in T_map:
                    T_pair = T_map[key] if key[0] == child else T_map[key].inv
                else:
                    T_pair, _ = pairwise_align(self.aligner, images[node], images[child], self.device)
                global_T[child] = global_T[node] @ T_pair
                queue.append(child)

        for i in range(m):
            if global_T[i] is None:
                global_T[i] = PlanarTransform('homography', device=self.device)

        # Apply transforms
        aligned = [images[i] if i == medoid else warp_image(images[i], global_T[i], self.device)
                   for i in range(m)]
        template = median_template(aligned)

        # Refinement rounds (re-align to median)
        for _ in range(self.n_refinement_rounds):
            aligned, global_T = [], []
            for i in range(m):
                T, _ = pairwise_align(self.aligner, template, images[i], self.device)
                aligned.append(warp_image(images[i], T, self.device))
                global_T.append(T)
            template = median_template(aligned)

        return {'aligned': aligned, 'transforms': global_T, 'template': template, 'medoid_idx': medoid}
