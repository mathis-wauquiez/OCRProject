
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np



kernels = {
    'central':{
        'k': np.array([0,1,0]),
        'd': np.array([-.5,0,.5])
    },
    'hypomode':{
        'k': np.array([0,0.5,0.5]),
        'd': np.array([0,-1,1])
    },
    'farid3': {
        'k': np.array([0.229879, 0.540242, 0.229879]),
        'd': np.array([-0.425287, 0, 0.425287])
    },
    'farid5': {
        'k': np.array([0.037659, 0.249153, 0.426375, 0.249153, 0.037659]),
        'd': np.array([-0.109604, -0.276691, 0, 0.276691, 0.109604])
    }
}


class Gradients(nn.Module):
    def __init__(self, method, C, device, ksize_factor=8, grdt_sigma=0.6):
        super().__init__()

        if method == 'gaussian':
            import cv2
            ksize = int(grdt_sigma * ksize_factor) // 2 * 2 + 1
            k = cv2.getGaussianKernel(ksize, grdt_sigma)
            d, _ = cv2.getDerivKernels(dx=1, dy=0, ksize=ksize, normalize=True)
        elif method in kernels:
            k = kernels[method]['k']
            d = kernels[method]['d']
            ksize = len(k)
        else:
            raise ValueError(f'Gradient computation method not recognized: {method}')

        self.C = C
        self.ksize = ksize
        self.pad = ksize // 2

        # Create kernels for separable convolution
        k_tensor = torch.tensor(k.flatten(), dtype=torch.float32, device=device)
        d_tensor = torch.tensor(d.flatten(), dtype=torch.float32, device=device)

        # Shape: (C, 1, 1, ksize) for horizontal, (C, 1, ksize, 1) for vertical
        self.register_buffer('k_h', k_tensor.view(1, 1, 1, ksize).expand(C, 1, 1, ksize).clone())
        self.register_buffer('k_v', k_tensor.view(1, 1, ksize, 1).expand(C, 1, ksize, 1).clone())
        self.register_buffer('d_h', d_tensor.view(1, 1, 1, ksize).expand(C, 1, 1, ksize).clone())
        self.register_buffer('d_v', d_tensor.view(1, 1, ksize, 1).expand(C, 1, ksize, 1).clone())

    def _conv_h(self, x, kernel):
        """Horizontal conv with Neumann BC."""
        x = F.pad(x, [self.pad, self.pad, 0, 0], mode='replicate')
        return F.conv2d(x, kernel, groups=self.C)

    def _conv_v(self, x, kernel):
        """Vertical conv with Neumann BC."""
        x = F.pad(x, [0, 0, self.pad, self.pad], mode='replicate')
        return F.conv2d(x, kernel, groups=self.C)

    def forward(self, batch):
        # dx: smooth vertically (k), differentiate horizontally (d)
        dx = self._conv_v(batch, self.k_v)
        dx = self._conv_h(dx, self.d_h)

        # dy: smooth horizontally (k), differentiate vertically (d)
        dy = self._conv_h(batch, self.k_h)
        dy = self._conv_v(dy, self.d_v)

        return dx, dy