"""
Gaussian pyramid for multiscale image alignment.

Ref: Briand, Facciolo, Sánchez – IPOL 2018, Section 3.4.
"""

import math
from functools import lru_cache

import torch
from torch import nn
import torch.nn.functional as F


class GaussianPyramid(nn.Module):
    """
    Builds a Gaussian pyramid ``[fine → coarse]``.

    Already batch-compatible: :meth:`forward` accepts ``(B, C, H, W)``
    for any *B*.
    """

    def __init__(self, eta=0.5, sigma_0=1.0, ksize_factor=8,
                 N_scales=None, min_size=8, dtype=torch.float32):
        """
        Args:
            eta: Downsampling factor in ``(0, 1)``.
            sigma_0: Std of the Gaussian blur kernel.
            ksize_factor: Kernel size ≈ ``2 · sigma · ksize_factor`` (odd).
            N_scales: Fixed number of scales (``None`` → auto from image size).
            min_size: Minimum spatial dimension at coarsest scale.
            dtype: Tensor dtype for kernels.
        """
        super().__init__()
        self.eta = eta
        self.sigma_0 = sigma_0
        self.ksize_factor = ksize_factor
        self.N_scales = N_scales
        self.min_size = min_size
        self.dtype = dtype

    @lru_cache(maxsize=16)
    def _gaussian_kernel_1d(self, sigma, device, dtype):
        """Normalised 1-D Gaussian kernel."""
        ksize = max(3, int(2 * sigma * self.ksize_factor) | 1)
        x = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        return kernel / kernel.sum()

    def _blur(self, x, sigma):
        """Separable Gaussian blur."""
        B, C, H, W = x.shape
        k1d = self._gaussian_kernel_1d(sigma, x.device, x.dtype)
        ksize = k1d.shape[0]
        pad = ksize // 2

        k_h = k1d.view(1, 1, 1, ksize).expand(C, 1, 1, ksize)
        x = F.conv2d(F.pad(x, [pad, pad, 0, 0], mode='reflect'), k_h, groups=C)

        k_v = k1d.view(1, 1, ksize, 1).expand(C, 1, ksize, 1)
        x = F.conv2d(F.pad(x, [0, 0, pad, pad], mode='reflect'), k_v, groups=C)

        return x

    def _max_scales(self, H, W):
        min_dim = min(H, W)
        if min_dim <= self.min_size:
            return 1
        return max(1, int(math.log(self.min_size / min_dim)
                          / math.log(self.eta)) + 1)

    def forward(self, x):
        """
        Build Gaussian pyramid.

        Args:
            x: ``(B, C, H, W)`` — any batch size.

        Returns:
            List of tensors ``[finest, …, coarsest]``.
        """
        B, C, H, W = x.shape
        N = self.N_scales or self._max_scales(H, W)
        pyramid = [x]

        for _ in range(1, N):
            blurred = self._blur(pyramid[-1], self.sigma_0)
            new_H = max(1, round(blurred.shape[2] * self.eta))
            new_W = max(1, round(blurred.shape[3] * self.eta))
            if new_H < 2 or new_W < 2:
                break
            pyramid.append(
                F.interpolate(blurred, (new_H, new_W),
                              mode='bilinear', align_corners=False))

        return pyramid


if __name__ == "__main__":
    pyr = GaussianPyramid(eta=0.5, sigma_0=1.0, N_scales=4)
    batch = torch.randn(2, 3, 256, 256)
    for i, level in enumerate(pyr(batch)):
        print(f"Level {i}: {tuple(level.shape)}")