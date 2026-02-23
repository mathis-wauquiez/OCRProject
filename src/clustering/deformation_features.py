"""
Deformation-Invariant Features — registration-free character descriptors.

1. Shape Context descriptors (Belongie et al.)
2. Persistent Homology via sublevel set filtration (gudhi)
3. TopologicalPreFilter for fast rejection of incompatible pairs
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt, binary_erosion


# ===========================================================================
#  Shape Context
# ===========================================================================

class ShapeContextDescriptor:
    """Shape Context descriptors for binary character images.

    Samples N_s contour points and computes log-polar histograms of
    relative positions.  Distance = min-cost Hungarian matching on
    chi-squared histogram costs.
    """

    def __init__(self, n_sample_points=100, n_angular_bins=12,
                 n_radial_bins=5, normalize_scale=True):
        self.n_sample_points = n_sample_points
        self.n_angular_bins = n_angular_bins
        self.n_radial_bins = n_radial_bins
        self.normalize_scale = normalize_scale
        self._n_bins = n_angular_bins * n_radial_bins

    def _sample_contour(self, binary):
        contour = (binary > 0) & ~binary_erosion(binary > 0)
        pts = np.argwhere(contour)
        if len(pts) == 0:
            pts = np.argwhere(binary > 0)
        if len(pts) == 0:
            return np.zeros((self.n_sample_points, 2))
        idx = np.linspace(0, len(pts) - 1, self.n_sample_points, dtype=int)
        return pts[idx].astype(float)

    def describe(self, binary):
        """Returns (N_s, n_bins) histogram array."""
        pts = self._sample_contour(binary)
        N = len(pts)
        histograms = np.zeros((N, self._n_bins))

        for i in range(N):
            diffs = np.delete(pts - pts[i], i, axis=0)
            r = np.linalg.norm(diffs, axis=1)
            theta = np.arctan2(diffs[:, 0], diffs[:, 1]) + np.pi

            if self.normalize_scale:
                med = np.median(r)
                if med > 0:
                    r = r / med

            r_log = np.log(r + 1e-10)
            r_lo, r_hi = r_log.min(), max(r_log.max(), r_log.min() + 1)
            r_idx = np.clip(np.digitize(r_log, np.linspace(r_lo, r_hi, self.n_radial_bins + 1)) - 1,
                            0, self.n_radial_bins - 1)
            t_idx = np.clip(np.digitize(theta, np.linspace(0, 2 * np.pi, self.n_angular_bins + 1)) - 1,
                            0, self.n_angular_bins - 1)

            for ri, ti in zip(r_idx, t_idx):
                histograms[i, ri * self.n_angular_bins + ti] += 1
            s = histograms[i].sum()
            if s > 0:
                histograms[i] /= s
        return histograms

    def distance(self, desc_a, desc_b):
        """Chi-squared + Hungarian matching distance."""
        N = min(len(desc_a), len(desc_b))
        if N == 0:
            return float('inf')
        a, b = desc_a[:N], desc_b[:N]
        cost = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                denom = a[i] + b[j]
                denom[denom == 0] = 1
                cost[i, j] = 0.5 * ((a[i] - b[j])**2 / denom).sum()
        ri, ci = linear_sum_assignment(cost)
        return float(cost[ri, ci].sum()) / N


# ===========================================================================
#  Persistent Homology
# ===========================================================================

class PersistentHomologyDescriptor:
    """Persistent homology via distance-transform sublevel set filtration.

    Uses gudhi for cubical complex persistence.
    H0 = connected components, H1 = loops.
    """

    def compute_persistence(self, binary):
        """Returns {'H0': [(b,d),...], 'H1': [(b,d),...]}."""
        import gudhi
        dt = distance_transform_edt(~(binary > 0))
        cc = gudhi.CubicalComplex(top_dimensional_cells=dt.flatten(),
                                  dimensions=list(dt.shape))
        cc.compute_persistence()
        result = {'H0': [], 'H1': []}
        for dim, (b, d) in cc.persistence():
            if d == float('inf'):
                d = dt.max()
            if dim in (0, 1):
                result[f'H{dim}'].append((float(b), float(d)))
        return result

    def wasserstein_distance(self, dgm_a, dgm_b, q=2):
        """Wasserstein-q distance between persistence diagrams."""
        # Pad shorter diagram with diagonal projections
        na, nb = len(dgm_a), len(dgm_b)
        diag = lambda b, d: (0.5 * (b + d), 0.5 * (b + d))

        a_pad = list(dgm_a) + [diag(b, d) for b, d in dgm_b]
        b_pad = list(dgm_b) + [diag(b, d) for b, d in dgm_a]
        size = len(a_pad)

        cost = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                pi, pj = np.array(a_pad[i]), np.array(b_pad[j])
                cost[i, j] = np.linalg.norm(pi - pj, ord=np.inf)**q
        ri, ci = linear_sum_assignment(cost)
        return float(cost[ri, ci].sum())**(1.0 / q)

    def topological_distance(self, img_a, img_b):
        """Combined H0 + H1 Wasserstein distance."""
        pa, pb = self.compute_persistence(img_a), self.compute_persistence(img_b)
        return (self.wasserstein_distance(pa['H0'], pb['H0']) +
                self.wasserstein_distance(pa['H1'], pb['H1']))


# ===========================================================================
#  Topological Pre-Filter
# ===========================================================================

class TopologicalPreFilter:
    """Fast rejection of topologically incompatible pairs.

    Compares the number of significant loops (H1 features with
    persistence > threshold).
    """

    def __init__(self, persistence_threshold=2.0):
        self._ph = PersistentHomologyDescriptor()
        self._thresh = persistence_threshold

    def signature(self, binary):
        """(n_components, n_loops) topological signature."""
        p = self._ph.compute_persistence(binary)
        count = lambda pairs: sum(1 for b, d in pairs if d - b > self._thresh)
        return (count(p['H0']), count(p['H1']))

    def compatibility_mask(self, signatures):
        """(N, N) boolean mask: True = topologically compatible."""
        sigs = np.array(signatures)
        N = len(sigs)
        return np.array([[np.array_equal(sigs[i], sigs[j])
                          for j in range(N)] for i in range(N)])
