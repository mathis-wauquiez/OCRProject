"""
K-Medoids with A Contrario Split/Merge — principled iterative refinement.

Replaces the three ad hoc refinement stages (Hausdorff splitting, OCR
rematching, PCA z-score) with a single principled algorithm:
  1. Reassignment step (move characters to better medoids)
  2. Split step (spectral bisection of impure clusters)
  3. Merge step (NFA-validated merge of redundant clusters)
  4. Medoid recomputation

Iterates until convergence or a maximum number of rounds.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

from .refinement import ClusterRefinementStep, RefinementResult


class KMedoidsSplitMerge:
    """
    K-Medoids clustering with NFA-guided split/merge operations.

    Parameters
    ----------
    max_iter : int
        Maximum reassign-split-merge iterations.
    split_threshold_sigma : float
        A cluster is split when its mean intra-cluster distance exceeds
        ``mu_0 + split_threshold_sigma * sigma_0`` where (mu_0, sigma_0)
        are estimated from the background model.
    min_split_size : int
        Minimum cluster size to consider splitting.
    merge_nlfa_threshold : float
        Minimum NLFA between medoids to consider merging.
    merge_cross_edge_fraction : float
        Minimum fraction of cross-cluster NFA edges for a merge.
    k_neighbors : int
        Number of nearest-medoid clusters to consider for reassignment.
    """

    def __init__(
        self,
        max_iter: int = 20,
        split_threshold_sigma: float = 2.0,
        min_split_size: int = 10,
        merge_nlfa_threshold: float = 5.0,
        merge_cross_edge_fraction: float = 0.3,
        k_neighbors: int = 5,
    ):
        self.max_iter = max_iter
        self.split_threshold_sigma = split_threshold_sigma
        self.min_split_size = min_split_size
        self.merge_nlfa_threshold = merge_nlfa_threshold
        self.merge_cross_edge_fraction = merge_cross_edge_fraction
        self.k_neighbors = k_neighbors

    def _compute_medoids(self, labels: np.ndarray, D: np.ndarray) -> Dict[int, int]:
        """Compute medoid (most central member) of each cluster."""
        medoids = {}
        for cid in np.unique(labels):
            if cid < 0:
                continue
            members = np.where(labels == cid)[0]
            if len(members) == 0:
                continue
            intra_D = D[np.ix_(members, members)]
            medoid_local = intra_D.sum(axis=1).argmin()
            medoids[cid] = members[medoid_local]
        return medoids

    def _reassign_step(
        self, labels: np.ndarray, D: np.ndarray, medoids: Dict[int, int],
    ) -> int:
        """Reassign each character to the nearest medoid. Returns count of moves."""
        n_moved = 0
        medoid_ids = list(medoids.keys())
        medoid_indices = np.array([medoids[c] for c in medoid_ids])

        for i in range(len(labels)):
            if labels[i] < 0:
                continue
            # Distance from i to all medoids
            dists = D[i, medoid_indices]
            # Consider only k nearest
            k = min(self.k_neighbors, len(medoid_ids))
            nearest_k = np.argpartition(dists, k)[:k]
            best_local = nearest_k[dists[nearest_k].argmin()]
            best_cid = medoid_ids[best_local]

            if best_cid != labels[i]:
                labels[i] = best_cid
                n_moved += 1

        return n_moved

    def _split_step(
        self, labels: np.ndarray, D: np.ndarray, nlfa: np.ndarray,
        bg_mu: float, bg_sigma: float,
    ) -> int:
        """Split impure clusters via spectral bisection. Returns splits done."""
        n_splits = 0
        next_id = int(labels.max()) + 1
        threshold = bg_mu + self.split_threshold_sigma * bg_sigma

        for cid in list(np.unique(labels)):
            if cid < 0:
                continue
            members = np.where(labels == cid)[0]
            if len(members) < self.min_split_size:
                continue

            # Mean intra-cluster distance
            intra_D = D[np.ix_(members, members)]
            mean_d = intra_D.mean()
            if mean_d <= threshold:
                continue

            # Spectral bisection using Fiedler vector
            # Build affinity from NLFA subgraph
            nlfa_sub = nlfa[np.ix_(members, members)]
            W = np.maximum(nlfa_sub, 0)
            np.fill_diagonal(W, 0)

            if W.sum() == 0:
                continue

            W_sparse = csr_matrix(W)
            L = laplacian(W_sparse, normed=True)

            try:
                # Fiedler vector = 2nd smallest eigenvector
                eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')
                fiedler = eigenvectors[:, 1]
            except Exception:
                continue

            # Split at zero crossing
            group_a = members[fiedler <= 0]
            group_b = members[fiedler > 0]

            if len(group_a) < 2 or len(group_b) < 2:
                continue

            # Assign groups
            labels[group_a] = cid
            labels[group_b] = next_id
            next_id += 1
            n_splits += 1

        return n_splits

    def _merge_step(
        self, labels: np.ndarray, nlfa: np.ndarray, medoids: Dict[int, int],
    ) -> int:
        """Merge cluster pairs with similar medoids and high cross-edge density."""
        n_merges = 0
        cluster_ids = sorted(medoids.keys())

        merged_into = {}  # track merge targets to avoid chain merges

        for i, ca in enumerate(cluster_ids):
            if ca in merged_into:
                continue
            for cb in cluster_ids[i + 1:]:
                if cb in merged_into:
                    continue

                ma, mb = medoids[ca], medoids[cb]
                nlfa_val = 0.5 * (nlfa[ma, mb] + nlfa[mb, ma])

                if nlfa_val < self.merge_nlfa_threshold:
                    continue

                # Check cross-edge fraction
                members_a = np.where(labels == ca)[0]
                members_b = np.where(labels == cb)[0]
                cross_nlfa = nlfa[np.ix_(members_a, members_b)]
                n_possible = len(members_a) * len(members_b)
                if n_possible == 0:
                    continue
                n_validated = (cross_nlfa > 0).sum()
                fraction = n_validated / n_possible

                if fraction >= self.merge_cross_edge_fraction:
                    # Merge cb into ca
                    labels[labels == cb] = ca
                    merged_into[cb] = ca
                    n_merges += 1

        return n_merges

    def fit(
        self, init_labels: np.ndarray, nlfa: np.ndarray, D: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run K-medoids with split/merge refinement.

        Parameters
        ----------
        init_labels : ndarray (N,)
            Initial cluster assignments (e.g., from connected components).
        nlfa : ndarray (N, N)
            NLFA matrix.
        D : ndarray (N, N)
            Distance matrix (lower = more similar).

        Returns
        -------
        dict with keys:
            ``labels`` : ndarray (N,)
            ``medoids`` : dict {cluster_id: character_index}
            ``n_iter`` : int
            ``history`` : list of per-iteration stats
        """
        labels = init_labels.copy()
        history = []

        # Background model statistics (from off-diagonal distances)
        triu_d = D[np.triu_indices_from(D, k=1)]
        bg_mu = triu_d.mean()
        bg_sigma = triu_d.std()

        for it in range(self.max_iter):
            # Compute medoids
            medoids = self._compute_medoids(labels, D)

            # Reassign
            n_moved = self._reassign_step(labels, D, medoids)

            # Split
            n_splits = self._split_step(labels, D, nlfa, bg_mu, bg_sigma)

            # Recompute medoids after split
            if n_splits > 0:
                medoids = self._compute_medoids(labels, D)

            # Merge
            n_merges = self._merge_step(labels, nlfa, medoids)

            history.append({
                'iteration': it,
                'n_moved': n_moved,
                'n_splits': n_splits,
                'n_merges': n_merges,
                'n_clusters': len(np.unique(labels[labels >= 0])),
            })

            if n_moved == 0 and n_splits == 0 and n_merges == 0:
                break

        # Renumber contiguously
        unique_ids = np.unique(labels[labels >= 0])
        remap = {old: new for new, old in enumerate(unique_ids)}
        remap[-1] = -1
        labels = np.array([remap.get(l, -1) for l in labels])
        medoids = self._compute_medoids(labels, D)

        return {
            'labels': labels,
            'medoids': medoids,
            'n_iter': it + 1,
            'history': history,
        }


# ---------------------------------------------------------------------------
#  Refinement step wrapper
# ---------------------------------------------------------------------------


class KMedoidsSplitMergeStep(ClusterRefinementStep):
    """
    Refinement step wrapping K-Medoids with split/merge.

    Replaces Hausdorff splitting + OCR rematching + PCA z-score
    with a single principled iterative algorithm.
    """

    name = "kmedoids_split_merge"

    def __init__(
        self,
        max_iter: int = 20,
        split_threshold_sigma: float = 2.0,
        min_split_size: int = 10,
        merge_nlfa_threshold: float = 5.0,
        d_max: float = 50.0,
    ):
        self.kmedoids = KMedoidsSplitMerge(
            max_iter=max_iter,
            split_threshold_sigma=split_threshold_sigma,
            min_split_size=min_split_size,
            merge_nlfa_threshold=merge_nlfa_threshold,
        )
        self.d_max = d_max

    def run(self, dataframe, membership, renderer, *,
            target_lbl, graph=None, nlfa=None, **ctx) -> RefinementResult:
        import torch

        if nlfa is None:
            nlfa = ctx.get('nlfa')
        if nlfa is None:
            return RefinementResult(
                membership=membership, log=[],
                metadata={'error': 'no NLFA matrix provided'},
            )

        if isinstance(nlfa, torch.Tensor):
            nlfa_np = nlfa.cpu().numpy()
        else:
            nlfa_np = np.asarray(nlfa)

        # Build distance from NLFA
        nlfa_sym = 0.5 * (nlfa_np + nlfa_np.T)
        D = self.d_max - nlfa_sym
        np.clip(D, 0.0, self.d_max, out=D)
        np.fill_diagonal(D, 0.0)

        result = self.kmedoids.fit(
            init_labels=np.asarray(membership),
            nlfa=nlfa_sym,
            D=D,
        )

        return RefinementResult(
            membership=result['labels'],
            log=result['history'],
            metadata=result,
        )
