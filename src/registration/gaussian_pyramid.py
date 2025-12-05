import torch
from torch import nn
import torch.nn.functional as F
import math
from functools import lru_cache
import numpy as np


class GaussianPyramid(nn.Module):

    def __init__(self, eta, sigma_0, ksize_factor=8, N_scales=None, min_size=8):
        """
        :param eta: Downsampling factor in (0, 1)
        :param sigma_0: Std of the gaussian kernel
        :param ksize_factor: Kernel size = sigma * ksize_factor
        :param N_scales: Number of scales (None = auto)
        :param min_size: Minimum dimension at coarsest scale
        """
        super().__init__()
        self.eta = eta
        self.sigma_0 = sigma_0
        self.ksize_factor = ksize_factor
        self.N_scales = N_scales
        self.min_size = min_size

    @lru_cache(maxsize=16) # cache to avoid redundant computations
    def _gaussian_kernel_1d(self, sigma, device, dtype):
        """Create normalized 1D Gaussian kernel."""
        ksize = max(3, int(2 * sigma * self.ksize_factor) | 1)  # n | 1 is an odd number
        x = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        return kernel / kernel.sum()

    def _blur(self, x, sigma):
        """Apply Gaussian blur using separable convolution (horizontal then vertical)."""
        B, C, H, W = x.shape
        k1d = self._gaussian_kernel_1d(sigma, x.device, x.dtype)
        ksize, pad = k1d.shape[0], k1d.shape[0] // 2

        # Horizontal: kernel shape (C, 1, 1, ksize)
        k_h = k1d.view(1, 1, 1, ksize).expand(C, 1, 1, ksize)
        x = F.conv2d(F.pad(x, [pad, pad, 0, 0], mode='reflect'), k_h, groups=C)

        # Vertical: kernel shape (C, 1, ksize, 1)
        k_v = k1d.view(1, 1, ksize, 1).expand(C, 1, ksize, 1)
        x = F.conv2d(F.pad(x, [0, 0, pad, pad], mode='reflect'), k_v, groups=C)

        return x

    def _max_scales(self, H, W):
        """Compute max pyramid levels based on image size."""
        min_dim = min(H, W)
        if min_dim <= self.min_size:
            return 1
        return max(1, int(math.log(self.min_size / min_dim) / math.log(self.eta)) + 1)

    def forward(self, x):
        """
        Compute Gaussian pyramid.
        
        :param x: Input tensor (B, C, H, W)
        :return: List of tensors from finest to coarsest
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

            pyramid.append(F.interpolate(blurred, (new_H, new_W), mode='bilinear', align_corners=False))

        return pyramid


if __name__ == "__main__":
    pyr = GaussianPyramid(eta=0.5, sigma_0=1.0, N_scales=4)
    batch = torch.randn(2, 3, 256, 256)

    for i, level in enumerate(pyr(batch)):
        print(f"Level {i}: {tuple(level.shape)}")

