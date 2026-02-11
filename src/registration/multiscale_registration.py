"""
Multiscale inverse compositional algorithm (Algorithm 4).

Coarse-to-fine estimation using a Gaussian pyramid.
Fully batched: processes *B* image pairs in parallel.

Ref: Briand, Facciolo, Sánchez – IPOL 2018.



THIS CODE VERSION IS AI-TWEAKED. ORIGINAL, HUMAN MADE FILE IN ARCHIVE
"""

import torch

from .single_scale import InverseCompositional
from .transformations import PlanarTransform
from .gaussian_pyramid import GaussianPyramid


class MultiscaleIC:
    """
    Multiscale modified inverse compositional algorithm.

    Wraps a single-scale :class:`InverseCompositional` solver and iterates
    coarse → fine over a Gaussian pyramid.
    """

    def __init__(self,
                 singleScaleIC: InverseCompositional,
                 gaussianPyramid: GaussianPyramid,
                 first_scale: int = 0,
                 grayscale: bool = True,
                 dtype: torch.dtype = torch.float32):
        """
        Args:
            singleScaleIC: Single-scale IC solver (already batched).
            gaussianPyramid: Gaussian pyramid builder.
            first_scale: First (finest) scale to use.  0 = all scales.
            grayscale: Convert to grayscale before processing.
            dtype: Working precision.
        """
        self.ic = singleScaleIC
        self.pyramid = gaussianPyramid
        self.first_scale = first_scale
        self.grayscale = grayscale
        self.dtype = dtype

        self.eta = gaussianPyramid.eta
        self.transform_type = singleScaleIC.transform_type

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _to_grayscale(img):
        """RGB → grayscale (ITU-R BT.601 weights)."""
        if img.shape[1] == 1:
            return img
        return (0.299 * img[:, 0:1] +
                0.587 * img[:, 1:2] +
                0.114 * img[:, 2:3])

    # ── main entry point ────────────────────────────────────────────────────
    def run(self, I1, I2, p_init=None, return_all_scales=False):
        """
        Run multiscale inverse compositional algorithm.

        Args:
            I1: Reference images ``(B, C, H, W)`` or ``(C, H, W)``.
            I2: Target images ``(B, C, H, W)`` or ``(C, H, W)``.
            p_init: Initial :class:`PlanarTransform` (batch_size *B* or 1)
                or ``None``.
            return_all_scales: If ``True``, return a dict with per-scale info.

        Returns:
            :class:`PlanarTransform` with ``batch_size == B``.
        """
        if I1.dim() == 3:
            I1 = I1.unsqueeze(0)
        if I2.dim() == 3:
            I2 = I2.unsqueeze(0)

        B, C, H, W = I1.shape
        device = I1.device

        # Cast
        I1 = I1.to(dtype=self.dtype)
        I2 = I2.to(dtype=self.dtype)

        # Grayscale
        if self.grayscale and C > 1:
            I1_proc = self._to_grayscale(I1)
            I2_proc = self._to_grayscale(I2)
        else:
            I1_proc = I1
            I2_proc = I2

        # Build Gaussian pyramids (Lines 1-2)
        pyr1 = self.pyramid(I1_proc)          # [fine → coarse]
        pyr2 = self.pyramid(I2_proc)

        n_scales = len(pyr1)

        # Initialise transformation at coarsest scale (Line 3)
        if p_init is not None:
            T = p_init
            if T.batch_size == 1 and B > 1:
                T = PlanarTransform(self.transform_type,
                                    matrix=T.matrix.expand(B, -1, -1).clone(),
                                    device=device, dtype=self.dtype)
            for _ in range(n_scales - 1):
                T = T.scale(self.eta)
        else:
            T = PlanarTransform(self.transform_type, batch_size=B,
                                device=device, dtype=self.dtype)

        transforms_per_scale = []
        pyramid_sizes = [(p.shape[2], p.shape[3]) for p in pyr1]

        # Coarse → fine (Lines 4-8)
        for s in range(n_scales - 1, self.first_scale - 1, -1):
            T = self.ic.run(pyr1[s], pyr2[s], p_init=T)
            transforms_per_scale.append(T)

            if s > self.first_scale:
                T = T.scale(1.0 / self.eta)

        if return_all_scales:
            return {
                'transform': T,
                'transforms_per_scale': transforms_per_scale[::-1],
                'pyramid_sizes': pyramid_sizes,
                'n_scales': n_scales,
            }
        return T