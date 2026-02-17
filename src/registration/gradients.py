"""
Image gradient computation via separable convolution.

Ref: Briand, Facciolo, Sánchez – IPOL 2018, Section 3.2.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ..utils import get_gradient_kernel


class Gradients(nn.Module):
    """Separable-filter image gradients."""

    def __init__(self, method, C, device, dtype=torch.float32,
                 ksize_factor=8, grdt_sigma=0.6):
        """
        Args:
            method: ``'central'``, ``'hypomode'``, ``'farid3'``, ``'farid5'``,
                    ``'central_differences'``, ``'farid_3x3'``, ``'farid_5x5'``,
                    or ``'gaussian'``.
            C: Number of image channels.
            device: Torch device.
            dtype: Torch dtype (``float32`` / ``float64``).
            ksize_factor: Kernel size multiplier (``'gaussian'`` method only).
            grdt_sigma: Sigma for Gaussian derivative kernels.
        """
        super().__init__()

        if method == 'gaussian':
            import cv2
            ksize = int(grdt_sigma * ksize_factor) // 2 * 2 + 1
            k = cv2.getGaussianKernel(ksize, grdt_sigma).astype(np.float64)
            d, _ = cv2.getDerivKernels(dx=1, dy=0, ksize=ksize, normalize=True)
            d = np.asarray(d, dtype=np.float64)
        else:
            kernel = get_gradient_kernel(method)
            k = kernel['k']
            d = kernel['d']
            ksize = len(k)

        self.C = C
        self.ksize = ksize
        self.pad = ksize // 2

        k_t = torch.tensor(k.flatten(), dtype=dtype, device=device)
        d_t = torch.tensor(d.flatten(), dtype=dtype, device=device)

        # (C, 1, 1, ksize) / (C, 1, ksize, 1)
        self.register_buffer('k_h', k_t.view(1, 1, 1, ksize).expand(C, 1, 1, ksize).clone())
        self.register_buffer('k_v', k_t.view(1, 1, ksize, 1).expand(C, 1, ksize, 1).clone())
        self.register_buffer('d_h', d_t.view(1, 1, 1, ksize).expand(C, 1, 1, ksize).clone())
        self.register_buffer('d_v', d_t.view(1, 1, ksize, 1).expand(C, 1, ksize, 1).clone())

    def _conv_h(self, x, kernel):
        """Horizontal conv with Neumann BC (replicate padding)."""
        x = F.pad(x, [self.pad, self.pad, 0, 0], mode='replicate')
        return F.conv2d(x, kernel, groups=self.C)

    def _conv_v(self, x, kernel):
        """Vertical conv with Neumann BC (replicate padding)."""
        x = F.pad(x, [0, 0, self.pad, self.pad], mode='replicate')
        return F.conv2d(x, kernel, groups=self.C)

    def forward(self, batch):
        """
        Returns ``(dx, dy)``, each ``(B, C, H, W)``.

        ``dx = d_h(k_v(I))``, ``dy = d_v(k_h(I))``.
        """
        dx = self._conv_v(batch, self.k_v)
        dx = self._conv_h(dx, self.d_h)

        dy = self._conv_h(batch, self.k_h)
        dy = self._conv_v(dy, self.d_v)

        return dx, dy