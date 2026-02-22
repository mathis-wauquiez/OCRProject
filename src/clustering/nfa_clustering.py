"""
NFA-based clustering methods — alternatives to Leiden on the binarized NFA graph.

Implements:
  1. HDBSCAN with NFA-derived distance metric (Section 1)
  2. Affinity Propagation on NFA similarities  (Section 2)

Both operate directly on the continuous NLFA matrix instead of
binarizing into edges and running community detection.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Any

# ---------------------------------------------------------------------------
#  1. HDBSCAN with NFA-derived metric
# ---------------------------------------------------------------------------


class HDBSCANClustering:
    """
    HDBSCAN clustering using an NFA-derived distance matrix.

    Given a (possibly sparse) NLFA similarity matrix, converts it to a
    distance matrix and runs HDBSCAN with ``metric='precomputed'``.

    Parameters
    ----------
    min_cluster_size : int
        Minimum cluster size.  Set low (3–5) to detect rare characters.
    min_samples : int
        Controls conservatism: higher → more noise points.
    cluster_selection_method : str
        ``'eom'`` (Excess of Mass, stability-based) or ``'leaf'``.
    d_max : float
        Sentinel distance for pairs without computed NFA or with NFA >= 1.
        Corresponds to NFA = 10^{-d_max}.
    noise_reassign_threshold : float or None
        If set, noise points (label = -1) closer than this distance to
        a cluster medoid are reassigned.  ``None`` → leave as singletons.
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 3,
        cluster_selection_method: str = 'eom',
        d_max: float = 50.0,
        noise_reassign_threshold: Optional[float] = None,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.d_max = d_max
        self.noise_reassign_threshold = noise_reassign_threshold

    def _nlfa_to_distance(self, nlfa: np.ndarray) -> np.ndarray:
        """
        Convert NLFA similarity matrix to distance matrix.

        NLFA values are higher for more similar pairs (NLFA = -log10(NFA)),
        so distance = d_max - NLFA.  Clamp to [0, d_max].
        """
        # Symmetrise if needed
        nlfa_sym = 0.5 * (nlfa + nlfa.T)
        distance = self.d_max - nlfa_sym
        np.clip(distance, 0.0, self.d_max, out=distance)
        np.fill_diagonal(distance, 0.0)
        return distance

    def fit(self, nlfa: np.ndarray) -> Dict[str, Any]:
        """
        Run HDBSCAN on an NLFA matrix.

        Parameters
        ----------
        nlfa : ndarray, shape (N, N)
            Symmetrised (or asymmetric) NLFA matrix.
            Higher values = more similar.

        Returns
        -------
        dict with keys:
            ``labels`` : ndarray (N,) — cluster labels (-1 = noise)
            ``probabilities`` : ndarray (N,) — cluster membership strengths
            ``n_clusters`` : int
            ``n_noise`` : int
            ``distance_matrix`` : ndarray (N, N)
        """
        import hdbscan

        D = self._nlfa_to_distance(nlfa)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='precomputed',
            cluster_selection_method=self.cluster_selection_method,
        )
        clusterer.fit(D)

        labels = clusterer.labels_.copy()
        probabilities = clusterer.probabilities_.copy()

        # Optional: reassign noise points to nearest cluster
        if self.noise_reassign_threshold is not None:
            labels = self._reassign_noise(labels, D)

        return {
            'labels': labels,
            'probabilities': probabilities,
            'n_clusters': int(labels.max() + 1) if labels.max() >= 0 else 0,
            'n_noise': int((labels == -1).sum()),
            'distance_matrix': D,
        }

    def _reassign_noise(self, labels: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Reassign noise points to the nearest cluster if close enough."""
        labels = labels.copy()
        noise_mask = labels == -1
        if not noise_mask.any() or labels.max() < 0:
            return labels

        # Compute medoid of each cluster
        cluster_ids = np.unique(labels[labels >= 0])
        medoids = {}
        for cid in cluster_ids:
            members = np.where(labels == cid)[0]
            intra_d = D[np.ix_(members, members)]
            medoid_local = intra_d.sum(axis=1).argmin()
            medoids[cid] = members[medoid_local]

        # Reassign noise
        noise_indices = np.where(noise_mask)[0]
        for ni in noise_indices:
            best_cid = None
            best_d = np.inf
            for cid, mi in medoids.items():
                d = D[ni, mi]
                if d < best_d:
                    best_d = d
                    best_cid = cid
            if best_d < self.noise_reassign_threshold and best_cid is not None:
                labels[ni] = best_cid

        return labels


# ---------------------------------------------------------------------------
#  2. Affinity Propagation on NFA similarities
# ---------------------------------------------------------------------------


class AffinityPropagationClustering:
    """
    Affinity Propagation clustering on NLFA similarities.

    Operates directly on the NLFA matrix (higher = more similar) without
    requiring the number of clusters K.  Produces exemplar-based clusters
    where each exemplar is a prototypical character instance.

    Parameters
    ----------
    preference_quantile : float
        Quantile of the similarity distribution used as the self-similarity
        (preference).  Lower → fewer clusters.  ``0.5`` = median.
    damping : float
        Message damping factor in [0.5, 1.0) to prevent oscillation.
    max_iter : int
        Maximum number of message-passing iterations.
    convergence_iter : int
        Stop if exemplar assignments are stable for this many iterations.
    fill_value : float
        Value for uncomputed/missing pairs in the sparse NLFA matrix.
        Should be a large negative number (strong dissimilarity).
    """

    def __init__(
        self,
        preference_quantile: float = 0.5,
        damping: float = 0.7,
        max_iter: int = 300,
        convergence_iter: int = 15,
        fill_value: float = 0.0,
    ):
        self.preference_quantile = preference_quantile
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.fill_value = fill_value

    def fit(self, nlfa: np.ndarray) -> Dict[str, Any]:
        """
        Run Affinity Propagation on an NLFA similarity matrix.

        Parameters
        ----------
        nlfa : ndarray, shape (N, N)
            NLFA matrix.  Higher values = more similar.

        Returns
        -------
        dict with keys:
            ``labels`` : ndarray (N,) — cluster labels
            ``exemplars`` : ndarray — indices of cluster exemplars
            ``n_clusters`` : int
        """
        from sklearn.cluster import AffinityPropagation

        # Symmetrise
        S = 0.5 * (nlfa + nlfa.T)

        # Replace zeros / missing with fill_value
        S[S == 0] = self.fill_value
        np.fill_diagonal(S, 0.0)  # will be set by preference

        # Compute preference
        triu_vals = S[np.triu_indices_from(S, k=1)]
        nonzero = triu_vals[triu_vals != self.fill_value]
        if len(nonzero) > 0:
            preference = np.quantile(nonzero, self.preference_quantile)
        else:
            preference = np.median(triu_vals)

        ap = AffinityPropagation(
            damping=self.damping,
            max_iter=self.max_iter,
            convergence_iter=self.convergence_iter,
            preference=preference,
            affinity='precomputed',
            random_state=42,
        )
        ap.fit(S)

        labels = ap.labels_
        exemplar_indices = ap.cluster_centers_indices_
        if exemplar_indices is None:
            exemplar_indices = np.array([], dtype=int)

        return {
            'labels': labels,
            'exemplars': exemplar_indices,
            'n_clusters': int(labels.max() + 1) if labels.max() >= 0 else 0,
            'preference': float(preference),
        }


# ---------------------------------------------------------------------------
#  Class-based wrappers for the graphClustering / sweep API
# ---------------------------------------------------------------------------


class HDBSCANNFACommunityDetection:
    """
    Drop-in replacement for ``communityDetectionBase`` that runs HDBSCAN
    on the full NLFA matrix rather than on the binarized graph.

    Usage in ``graphClusteringSweep.report_graph``:
        Instead of building a graph and running Leiden, pass the NLFA
        matrix directly to this object.
    """

    name = "hdbscan_nfa"

    def __init__(self, min_cluster_size: int = 3, min_samples: int = 3,
                 d_max: float = 50.0,
                 noise_reassign_threshold: Optional[float] = None):
        self._hdbscan = HDBSCANClustering(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            d_max=d_max,
            noise_reassign_threshold=noise_reassign_threshold,
        )

    def fit_nlfa(self, nlfa: np.ndarray) -> Dict[str, Any]:
        return self._hdbscan.fit(nlfa)


class AffinityPropagationNFACommunityDetection:
    """
    Drop-in replacement that runs Affinity Propagation on the NLFA matrix.
    """

    name = "affinity_propagation_nfa"

    def __init__(self, preference_quantile: float = 0.5,
                 damping: float = 0.7):
        self._ap = AffinityPropagationClustering(
            preference_quantile=preference_quantile,
            damping=damping,
        )

    def fit_nlfa(self, nlfa: np.ndarray) -> Dict[str, Any]:
        return self._ap.fit(nlfa)
