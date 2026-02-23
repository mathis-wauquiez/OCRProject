"""
K-Medoids with A Contrario Split/Merge — principled iterative refinement.

Replaces the three ad hoc stages with: reassign → spectral split → merge.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

from .refinement import ClusterRefinementStep, RefinementResult

log = logging.getLogger(__name__)


def nlfa_to_distance(nlfa, d_max=50.0):
    """Convert NLFA similarity matrix to distance matrix for precomputed clustering.

    NLFA values are higher for more similar pairs, so distance = d_max - NLFA.
    """
    nlfa_sym = 0.5 * (nlfa + nlfa.T)
    D = d_max - nlfa_sym
    np.clip(D, 0.0, d_max, out=D)
    np.fill_diagonal(D, 0.0)
    return D


class KMedoidsSplitMerge:
    """K-Medoids with NFA-guided split/merge.

    Parameters
    ----------
    max_iter : int
        Maximum reassign-split-merge iterations.
    split_threshold_sigma : float
        Split when intra-cluster distance > mu_0 + sigma * sigma_0.
    min_split_size : int
        Minimum cluster size to consider splitting.
    merge_nlfa_threshold : float
        Minimum NLFA between medoids to consider merging.
    merge_cross_edge_fraction : float
        Minimum cross-cluster NFA edge fraction for a merge.
    k_neighbors : int
        Nearest-medoid clusters to consider for reassignment.
    """

    def __init__(self, max_iter=20, split_threshold_sigma=2.0,
                 min_split_size=10, merge_nlfa_threshold=5.0,
                 merge_cross_edge_fraction=0.3, k_neighbors=5):
        self.max_iter = max_iter
        self.split_threshold_sigma = split_threshold_sigma
        self.min_split_size = min_split_size
        self.merge_nlfa_threshold = merge_nlfa_threshold
        self.merge_cross_edge_fraction = merge_cross_edge_fraction
        self.k_neighbors = k_neighbors

    @staticmethod
    def _medoids(labels, D):
        medoids = {}
        for cid in np.unique(labels):
            if cid < 0:
                continue
            members = np.where(labels == cid)[0]
            intra = D[np.ix_(members, members)]
            medoids[cid] = members[intra.sum(axis=1).argmin()]
        return medoids

    def _reassign(self, labels, D, medoids):
        n_moved = 0
        cids = list(medoids.keys())
        midx = np.array([medoids[c] for c in cids])
        for i in range(len(labels)):
            if labels[i] < 0:
                continue
            dists = D[i, midx]
            k = min(self.k_neighbors, len(cids))
            nearest = np.argpartition(dists, k)[:k]
            best = cids[nearest[dists[nearest].argmin()]]
            if best != labels[i]:
                labels[i] = best
                n_moved += 1
        return n_moved

    def _split(self, labels, D, nlfa, threshold):
        n_splits = 0
        next_id = int(labels.max()) + 1
        for cid in list(np.unique(labels)):
            if cid < 0:
                continue
            members = np.where(labels == cid)[0]
            if len(members) < self.min_split_size:
                continue
            if D[np.ix_(members, members)].mean() <= threshold:
                continue
            W = np.maximum(nlfa[np.ix_(members, members)], 0)
            np.fill_diagonal(W, 0)
            if W.sum() == 0:
                continue
            try:
                _, evecs = eigsh(laplacian(csr_matrix(W), normed=True), k=2, which='SM')
                fiedler = evecs[:, 1]
            except Exception:
                continue
            a, b = members[fiedler <= 0], members[fiedler > 0]
            if len(a) < 2 or len(b) < 2:
                continue
            labels[a] = cid
            labels[b] = next_id
            next_id += 1
            n_splits += 1
        return n_splits

    def _merge(self, labels, nlfa, medoids):
        n_merges = 0
        cids = sorted(medoids.keys())
        merged = set()
        for i, ca in enumerate(cids):
            if ca in merged:
                continue
            for cb in cids[i + 1:]:
                if cb in merged:
                    continue
                ma, mb = medoids[ca], medoids[cb]
                if 0.5 * (nlfa[ma, mb] + nlfa[mb, ma]) < self.merge_nlfa_threshold:
                    continue
                a_idx = np.where(labels == ca)[0]
                b_idx = np.where(labels == cb)[0]
                n_possible = len(a_idx) * len(b_idx)
                if n_possible and (nlfa[np.ix_(a_idx, b_idx)] > 0).sum() / n_possible >= self.merge_cross_edge_fraction:
                    labels[labels == cb] = ca
                    merged.add(cb)
                    n_merges += 1
        return n_merges

    def fit(self, init_labels, nlfa, D):
        labels = init_labels.copy()
        triu = D[np.triu_indices_from(D, k=1)]
        threshold = triu.mean() + self.split_threshold_sigma * triu.std()
        history = []

        for it in range(self.max_iter):
            medoids = self._medoids(labels, D)
            n_moved = self._reassign(labels, D, medoids)
            n_splits = self._split(labels, D, nlfa, threshold)
            if n_splits:
                medoids = self._medoids(labels, D)
            n_merges = self._merge(labels, nlfa, medoids)
            history.append({'iteration': it, 'n_moved': n_moved,
                            'n_splits': n_splits, 'n_merges': n_merges,
                            'n_clusters': len(np.unique(labels[labels >= 0]))})
            if n_moved == 0 and n_splits == 0 and n_merges == 0:
                break

        # Renumber
        uids = np.unique(labels[labels >= 0])
        remap = {old: new for new, old in enumerate(uids)}
        remap[-1] = -1
        labels = np.array([remap.get(l, -1) for l in labels])
        return {'labels': labels, 'medoids': self._medoids(labels, D),
                'n_iter': it + 1, 'history': history}


class KMedoidsSplitMergeStep(ClusterRefinementStep):
    """Refinement step wrapping K-Medoids with split/merge.

    Requires ``nlfa`` in ``ctx``.
    """

    name = "kmedoids_split_merge"

    def __init__(self, max_iter=20, split_threshold_sigma=2.0,
                 min_split_size=10, merge_nlfa_threshold=5.0, d_max=50.0):
        self.kmedoids = KMedoidsSplitMerge(
            max_iter=max_iter, split_threshold_sigma=split_threshold_sigma,
            min_split_size=min_split_size, merge_nlfa_threshold=merge_nlfa_threshold)
        self.d_max = d_max

    def run(self, dataframe, membership, renderer, *, target_lbl, **ctx):
        import torch

        nlfa = ctx.get('nlfa')
        if nlfa is None:
            log.warning("KMedoidsSplitMergeStep: no NLFA in ctx, skipping.")
            return RefinementResult(membership=membership, log=[], metadata={'skipped': True})

        nlfa_np = nlfa.cpu().numpy() if isinstance(nlfa, torch.Tensor) else np.asarray(nlfa)
        nlfa_sym = 0.5 * (nlfa_np + nlfa_np.T)
        D = nlfa_to_distance(nlfa_np, self.d_max)

        result = self.kmedoids.fit(np.asarray(membership), nlfa_sym, D)
        return RefinementResult(membership=result['labels'],
                                log=result['history'], metadata=result)
