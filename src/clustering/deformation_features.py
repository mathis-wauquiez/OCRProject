"""
Deformation-Invariant Features — registration-free descriptors for characters.

Implements:
  1. Shape Context descriptors (Belongie et al.)
  2. Persistent Homology (topological features via sublevel set filtration)

These features are inherently invariant to geometric transformations
and can be used as pre-filters or standalone descriptors.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt


# ===========================================================================
#  1. Shape Context Descriptors
# ===========================================================================


class ShapeContextDescriptor:
    """
    Shape Context descriptors for binary character images.

    For each character, samples N_s points uniformly from the contour
    and computes log-polar histograms of relative point positions.
    The distance between two characters is the minimum-cost bipartite
    matching of their shape context histograms (Hungarian algorithm).

    Parameters
    ----------
    n_sample_points : int
        Number of contour points to sample per character.
    n_angular_bins : int
        Number of angular bins in the log-polar histogram.
    n_radial_bins : int
        Number of radial bins (log-spaced).
    normalize_scale : bool
        If True, normalize radial distances by the median pairwise
        distance, making the descriptor scale-invariant.
    """

    def __init__(
        self,
        n_sample_points: int = 100,
        n_angular_bins: int = 12,
        n_radial_bins: int = 5,
        normalize_scale: bool = True,
    ):
        self.n_sample_points = n_sample_points
        self.n_angular_bins = n_angular_bins
        self.n_radial_bins = n_radial_bins
        self.normalize_scale = normalize_scale

    def _sample_contour_points(self, binary_image: np.ndarray) -> np.ndarray:
        """Extract and uniformly sample contour points from a binary image."""
        # Find contour pixels (edge detection on binary)
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(binary_image)
        contour = binary_image & ~eroded
        points = np.argwhere(contour)  # (N, 2) — (row, col)

        if len(points) == 0:
            # Fallback to all foreground points
            points = np.argwhere(binary_image)

        if len(points) == 0:
            return np.zeros((self.n_sample_points, 2))

        # Uniform sampling
        if len(points) >= self.n_sample_points:
            indices = np.linspace(0, len(points) - 1,
                                  self.n_sample_points, dtype=int)
            return points[indices].astype(float)
        else:
            # Repeat if too few points
            repeats = (self.n_sample_points // len(points)) + 1
            points_rep = np.tile(points, (repeats, 1))[:self.n_sample_points]
            return points_rep.astype(float)

    def _compute_shape_context(self, points: np.ndarray) -> np.ndarray:
        """
        Compute shape context histograms for a set of sample points.

        Returns
        -------
        histograms : ndarray (N_s, n_angular_bins * n_radial_bins)
        """
        N = len(points)
        n_bins = self.n_angular_bins * self.n_radial_bins
        histograms = np.zeros((N, n_bins))

        # Pairwise relative vectors
        for i in range(N):
            diffs = points - points[i]  # (N, 2)
            diffs = np.delete(diffs, i, axis=0)  # exclude self

            # Convert to polar
            r = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
            theta = np.arctan2(diffs[:, 0], diffs[:, 1]) + np.pi  # [0, 2*pi]

            # Normalize radial
            if self.normalize_scale:
                median_r = np.median(r)
                if median_r > 0:
                    r = r / median_r

            # Log-polar binning
            r_log = np.log(r + 1e-10)
            r_min, r_max = r_log.min(), r_log.max()
            if r_max - r_min < 1e-8:
                r_max = r_min + 1

            r_bins = np.linspace(r_min, r_max, self.n_radial_bins + 1)
            theta_bins = np.linspace(0, 2 * np.pi, self.n_angular_bins + 1)

            r_idx = np.clip(
                np.digitize(r_log, r_bins) - 1, 0, self.n_radial_bins - 1)
            theta_idx = np.clip(
                np.digitize(theta, theta_bins) - 1, 0, self.n_angular_bins - 1)

            for ri, ti in zip(r_idx, theta_idx):
                histograms[i, ri * self.n_angular_bins + ti] += 1

            # Normalize histogram
            total = histograms[i].sum()
            if total > 0:
                histograms[i] /= total

        return histograms

    def describe(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Compute Shape Context descriptor for a binary character image.

        Parameters
        ----------
        binary_image : ndarray (H, W), bool or 0/1

        Returns
        -------
        histograms : ndarray (N_s, n_angular_bins * n_radial_bins)
        """
        points = self._sample_contour_points(binary_image > 0)
        return self._compute_shape_context(points)

    def distance(
        self, desc_a: np.ndarray, desc_b: np.ndarray,
    ) -> float:
        """
        Compute Shape Context distance between two characters.

        Uses chi-squared distance between histograms and Hungarian
        algorithm for minimum-cost matching.
        """
        N = min(len(desc_a), len(desc_b))
        if N == 0:
            return float('inf')

        # Truncate to same length
        a = desc_a[:N]
        b = desc_b[:N]

        # Chi-squared cost matrix
        cost = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                denom = a[i] + b[j]
                denom[denom == 0] = 1
                cost[i, j] = 0.5 * ((a[i] - b[j])**2 / denom).sum()

        # Hungarian matching
        row_idx, col_idx = linear_sum_assignment(cost)
        return float(cost[row_idx, col_idx].sum()) / N

    def pairwise_distances(
        self, images: List[np.ndarray],
    ) -> np.ndarray:
        """Compute all-pairs Shape Context distance matrix."""
        descs = [self.describe(img) for img in images]
        N = len(images)
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                d = self.distance(descs[i], descs[j])
                D[i, j] = d
                D[j, i] = d
        return D


# ===========================================================================
#  2. Persistent Homology
# ===========================================================================


class PersistentHomologyDescriptor:
    """
    Persistent Homology descriptors for binary character images.

    Computes the sublevel set filtration of the distance transform and
    extracts topological features (H0 = connected components, H1 = loops)
    as persistence diagrams.

    This is powerful for characters because:
      - 'o', 'b' have 1 loop (H1)
      - '8' has 2 loops
      - 'c', 'l' have 0 loops
    Topologically incompatible characters are rejected cheaply.

    Uses the gudhi library for persistence computation.
    """

    def __init__(self, max_edge_length: float = 50.0):
        self.max_edge_length = max_edge_length

    def compute_persistence(
        self, binary_image: np.ndarray,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Compute persistence diagram from the distance transform filtration.

        Parameters
        ----------
        binary_image : ndarray (H, W), bool or 0/1

        Returns
        -------
        dict with keys:
            ``H0`` : list of (birth, death) for connected components
            ``H1`` : list of (birth, death) for loops
        """
        import gudhi

        # Distance transform (distance from each pixel to nearest ink pixel)
        dt = distance_transform_edt(~(binary_image > 0))

        # Build cubical complex from distance transform
        cc = gudhi.CubicalComplex(
            top_dimensional_cells=dt.flatten(),
            dimensions=list(dt.shape),
        )
        cc.compute_persistence()

        # Extract persistence pairs
        result = {'H0': [], 'H1': []}
        for dim, (birth, death) in cc.persistence():
            if death == float('inf'):
                death = dt.max()
            if dim == 0:
                result['H0'].append((float(birth), float(death)))
            elif dim == 1:
                result['H1'].append((float(birth), float(death)))

        return result

    @staticmethod
    def persistence_to_vector(
        persistence: Dict[str, List[Tuple[float, float]]],
        n_features: int = 10,
    ) -> np.ndarray:
        """
        Convert persistence diagram to a fixed-length feature vector.

        For each homology dimension, extracts the top-n longest-lived
        features sorted by persistence (death - birth).
        """
        features = []
        for key in ['H0', 'H1']:
            pairs = persistence.get(key, [])
            # Sort by persistence (longest-lived first)
            persistences = sorted(
                [d - b for b, d in pairs], reverse=True,
            )
            # Pad or truncate
            padded = persistences[:n_features]
            padded += [0.0] * (n_features - len(padded))
            features.extend(padded)
        return np.array(features)

    def wasserstein_distance(
        self,
        dgm_a: List[Tuple[float, float]],
        dgm_b: List[Tuple[float, float]],
        q: int = 2,
    ) -> float:
        """
        Compute Wasserstein-q distance between two persistence diagrams.

        Points can be matched to each other or to the diagonal.
        """
        na, nb = len(dgm_a), len(dgm_b)
        n = max(na, nb)
        if n == 0:
            return 0.0

        # Pad shorter diagram with diagonal points
        a = list(dgm_a) + [(0.5 * (b + d), 0.5 * (b + d)) for b, d in dgm_b[na:]]
        b = list(dgm_b) + [(0.5 * (b + d), 0.5 * (b + d)) for b, d in dgm_a[nb:]]

        # Augment both with diagonal projections
        a_aug = list(dgm_a) + [(0.5 * (b + d), 0.5 * (b + d)) for b, d in dgm_b]
        b_aug = list(dgm_b) + [(0.5 * (b + d), 0.5 * (b + d)) for b, d in dgm_a]

        N = len(a_aug)
        M = len(b_aug)
        size = max(N, M)

        # Build cost matrix
        cost = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i < N and j < M:
                    pi = np.array(a_aug[i])
                    pj = np.array(b_aug[j])
                    cost[i, j] = np.linalg.norm(pi - pj, ord=np.inf)**q
                else:
                    cost[i, j] = 0.0

        row_idx, col_idx = linear_sum_assignment(cost)
        return float(cost[row_idx, col_idx].sum())**(1.0 / q)

    def topological_distance(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
    ) -> float:
        """
        Compute topological distance between two binary character images.

        Combines H0 and H1 Wasserstein distances.
        """
        pa = self.compute_persistence(img_a)
        pb = self.compute_persistence(img_b)

        d0 = self.wasserstein_distance(pa['H0'], pb['H0'])
        d1 = self.wasserstein_distance(pa['H1'], pb['H1'])
        return d0 + d1

    def is_topologically_compatible(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        threshold: float = 2.0,
    ) -> bool:
        """
        Quick topological compatibility check.

        Compares the number of significant loops (H1 features with
        persistence > threshold).
        """
        pa = self.compute_persistence(img_a)
        pb = self.compute_persistence(img_b)

        def count_significant(pairs, thresh):
            return sum(1 for b, d in pairs if d - b > thresh)

        loops_a = count_significant(pa['H1'], threshold)
        loops_b = count_significant(pb['H1'], threshold)
        return loops_a == loops_b

    def pairwise_distances(
        self, images: List[np.ndarray],
    ) -> np.ndarray:
        """Compute all-pairs topological distance matrix."""
        N = len(images)
        persistences = [self.compute_persistence(img) for img in images]

        D = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                d0 = self.wasserstein_distance(
                    persistences[i]['H0'], persistences[j]['H0'])
                d1 = self.wasserstein_distance(
                    persistences[i]['H1'], persistences[j]['H1'])
                d = d0 + d1
                D[i, j] = d
                D[j, i] = d
        return D


# ===========================================================================
#  Pre-filter combining both
# ===========================================================================


class TopologicalPreFilter:
    """
    Fast rejection of topologically incompatible character pairs.

    Before running the expensive NFA comparison, check if two characters
    have different numbers of significant topological features (loops).
    This is O(HW) per character (distance transform) and can reject
    pairs like ('o', 'c') immediately.
    """

    def __init__(self, persistence_threshold: float = 2.0):
        self.ph = PersistentHomologyDescriptor()
        self.persistence_threshold = persistence_threshold

    def compute_topology_signature(self, binary_image: np.ndarray) -> Tuple[int, int]:
        """
        Compute a compact topological signature: (n_components, n_loops).
        """
        persistence = self.ph.compute_persistence(binary_image)
        n_components = sum(
            1 for b, d in persistence['H0']
            if d - b > self.persistence_threshold
        )
        n_loops = sum(
            1 for b, d in persistence['H1']
            if d - b > self.persistence_threshold
        )
        return (n_components, n_loops)

    def build_signature_matrix(
        self, images: List[np.ndarray],
    ) -> np.ndarray:
        """
        Compute topology signatures for all images.

        Returns ndarray (N, 2) of (n_components, n_loops).
        """
        sigs = [self.compute_topology_signature(img) for img in images]
        return np.array(sigs)

    def compatibility_mask(
        self, signatures: np.ndarray,
    ) -> np.ndarray:
        """
        Build a boolean mask (N, N) where True = topologically compatible.
        """
        N = len(signatures)
        mask = np.zeros((N, N), dtype=bool)
        for i in range(N):
            for j in range(i, N):
                compatible = np.array_equal(signatures[i], signatures[j])
                mask[i, j] = compatible
                mask[j, i] = compatible
        return mask
