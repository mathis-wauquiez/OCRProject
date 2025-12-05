from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np

import matplotlib.pyplot as plt



class Transformation(ABC):
    """ Base class for a set of transformations with a group structure """

    @abstractmethod
    def jacobian(self, xx, yy):
        """
        Should return the jacobian of the transformation evaluated at points (x, y)
        
        :param xx: x-grid
        :param yy: y-grid
        :output:   (*xx.shape, 2, p) tensor
        """
        ...
    
    @abstractmethod
    def warp(self, img_batch):
        """
        Should warp an image.
        
        :param img_batch: Batch to be warped
        :output:          Warped batch
        """
        ...

    @property
    @abstractmethod
    def inv(self) -> 'Transformation':
        """Inverse transformation."""
        ...
        
    @abstractmethod
    def compose(self, other: 'Transformation') -> 'Transformation':
        """Return self o other."""
        ...

    @abstractmethod
    def visibility_mask(self, *args):
        """Returns \Omega^{\delta, p}"""
        ...

    def __matmul__(self, other: 'Transformation') -> 'Transformation':
        """Allow T1 @ T2 syntax for composition."""
        return self.compose(other)


class PlanarTransform(Transformation):
    """
    Unified planar transformation (Table 1).
    """

    # ── Type-specific: Jacobian functions (evaluated at p=0) ────────────────
    @staticmethod
    def _J_translation(x, y):
        J = torch.zeros(*x.shape, 2, 2, device=x.device, dtype=x.dtype)
        J[..., 0, 0] = J[..., 1, 1] = 1
        return J

    @staticmethod
    def _J_euclidean(x, y):
        J = torch.zeros(*x.shape, 2, 3, device=x.device, dtype=x.dtype)
        J[..., 0, 0] = J[..., 1, 1] = 1
        J[..., 0, 2], J[..., 1, 2] = -y, x
        return J

    @staticmethod
    def _J_similarity(x, y):
        J = torch.zeros(*x.shape, 2, 4, device=x.device, dtype=x.dtype)
        J[..., 0, 0] = J[..., 1, 1] = 1
        J[..., 0, 2], J[..., 1, 2] = x, y
        J[..., 0, 3], J[..., 1, 3] = -y, x
        return J

    @staticmethod
    def _J_affinity(x, y):
        J = torch.zeros(*x.shape, 2, 6, device=x.device, dtype=x.dtype)
        J[..., 0, 0] = J[..., 1, 1] = 1
        J[..., 0, 2], J[..., 0, 3] = x, y
        J[..., 1, 4], J[..., 1, 5] = x, y
        return J

    @staticmethod
    def _J_homography(x, y):
        J = torch.zeros(*x.shape, 2, 8, device=x.device, dtype=x.dtype)
        J[..., 0, 0], J[..., 0, 1], J[..., 0, 2] = x, y, 1
        J[..., 1, 3], J[..., 1, 4], J[..., 1, 5] = x, y, 1
        J[..., 0, 6], J[..., 0, 7] = -x*x, -x*y
        J[..., 1, 6], J[..., 1, 7] = -x*y, -y*y
        return J

    # ── Type-specific: params → matrix ──────────────────────────────────────
    @staticmethod
    def _M_translation(p):
        B = p.shape[0]
        H = torch.eye(3, device=p.device, dtype=p.dtype).unsqueeze(0).repeat(B, 1, 1)
        H[:, 0, 2], H[:, 1, 2] = p[:, 0], p[:, 1]
        return H

    @staticmethod
    def _M_euclidean(p):
        c, s = torch.cos(p[:, 2]), torch.sin(p[:, 2])
        H = torch.zeros(p.shape[0], 3, 3, device=p.device, dtype=p.dtype)
        H[:, 0, 0], H[:, 0, 1], H[:, 0, 2] = c, -s, p[:, 0]
        H[:, 1, 0], H[:, 1, 1], H[:, 1, 2] = s, c, p[:, 1]
        H[:, 2, 2] = 1
        return H

    @staticmethod
    def _M_similarity(p):
        H = torch.zeros(p.shape[0], 3, 3, device=p.device, dtype=p.dtype)
        H[:, 0, 0], H[:, 0, 1], H[:, 0, 2] = 1 + p[:, 2], -p[:, 3], p[:, 0]
        H[:, 1, 0], H[:, 1, 1], H[:, 1, 2] = p[:, 3], 1 + p[:, 2], p[:, 1]
        H[:, 2, 2] = 1
        return H

    @staticmethod
    def _M_affinity(p):
        H = torch.zeros(p.shape[0], 3, 3, device=p.device, dtype=p.dtype)
        H[:, 0, 0], H[:, 0, 1], H[:, 0, 2] = 1 + p[:, 2], p[:, 3], p[:, 0]
        H[:, 1, 0], H[:, 1, 1], H[:, 1, 2] = p[:, 4], 1 + p[:, 5], p[:, 1]
        H[:, 2, 2] = 1
        return H

    @staticmethod
    def _M_homography(p):
        H = torch.zeros(p.shape[0], 3, 3, device=p.device, dtype=p.dtype)
        H[:, 0, 0], H[:, 0, 1], H[:, 0, 2] = 1 + p[:, 0], p[:, 1], p[:, 2]
        H[:, 1, 0], H[:, 1, 1], H[:, 1, 2] = p[:, 3], 1 + p[:, 4], p[:, 5]
        H[:, 2, 0], H[:, 2, 1], H[:, 2, 2] = p[:, 6], p[:, 7], 1
        return H

    # ── Registry ────────────────────────────────────────────────────────────
    TYPES = {
        'translation': (2, _J_translation.__func__, _M_translation.__func__),
        'euclidean':   (3, _J_euclidean.__func__,   _M_euclidean.__func__),
        'similarity':  (4, _J_similarity.__func__,  _M_similarity.__func__),
        'affinity':    (6, _J_affinity.__func__,    _M_affinity.__func__),
        'homography':  (8, _J_homography.__func__,  _M_homography.__func__),
    }

    def __init__(self, ttype='homography', params=None, matrix=None, device=None, dtype=torch.float32):
        if ttype not in self.TYPES:
            raise ValueError(f"Unknown type: {ttype}")

        self.type = ttype
        self.n_params, self._J, self._M = self.TYPES[ttype]

        if matrix is not None:
            self.matrix = matrix.to(device=device, dtype=dtype)
        elif params is not None:
            p = torch.as_tensor(params, device=device, dtype=dtype)
            p = p.unsqueeze(0) if p.dim() == 1 else p
            self.matrix = self._M(p).squeeze(0)
        else:
            self.matrix = torch.eye(3, device=device, dtype=dtype)

    def transform_points(self, xx, yy):
        """Apply transformation: (x', y') = Ψ(x, y; p)"""
        H = self.matrix
        d = H[2, 0] * xx + H[2, 1] * yy + H[2, 2]
        xx_t = (H[0, 0] * xx + H[0, 1] * yy + H[0, 2]) / d
        yy_t = (H[1, 0] * xx + H[1, 1] * yy + H[1, 2]) / d
        return xx_t, yy_t

    def jacobian(self, xx, yy):
        """Jacobian ∂Ψ/∂p at p=0. Shape: (*xx.shape, 2, n_params)."""
        return self._J(xx, yy)

    def warp(self, img, mode='bicubic', padding_mode='zeros'):
        """
        Warp image: output[y, x] = input[Ψ(x, y)]
        """
        B, C, H, W = img.shape
        device, dtype = img.device, img.dtype
        
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        xx_t, yy_t = self.transform_points(xx, yy)
        
        # Normalize to [-1, 1] for grid_sample
        grid = torch.stack([
            2 * xx_t / (W - 1) - 1,
            2 * yy_t / (H - 1) - 1
        ], dim=-1)
        
        return F.grid_sample(img, grid[None].expand(B, -1, -1, -1),
                            mode=mode, padding_mode=padding_mode, align_corners=True)

    @property
    def inv(self):
        """Inverse transformation."""
        return PlanarTransform(self.type, matrix=torch.linalg.inv(self.matrix))

    def compose(self, other):
        """Composition: (self ∘ other)(x) = self(other(x))"""
        return PlanarTransform(self.type, matrix=self.matrix @ other.matrix)

    def __matmul__(self, other):
        return self.compose(other)

    def visibility_mask(self, H, W, delta=0):
        """Ω^{δ,p} mask (Eq. 23-25)."""
        device, dtype = self.matrix.device, self.matrix.dtype
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        interior = (xx >= delta) & (xx <= W-1-delta) & (yy >= delta) & (yy <= H-1-delta)
        xx_t, yy_t = self.transform_points(xx, yy)
        mapped = (xx_t >= delta) & (xx_t <= W-1-delta) & (yy_t >= delta) & (yy_t <= H-1-delta)
        return interior & mapped


    def scale(self, factor):
        """
        Scale transformation for pyramid level change.
        
        factor > 1: going to finer level (coarse→fine)
        factor < 1: going to coarser level (fine→coarse)
        
        Translation scales proportionally with image size.
        Perspective scales inversely with image size.
        """
        H = self.matrix.clone()
        H[0, 2] *= factor  # translation scales with image
        H[1, 2] *= factor
        if self.type == 'homography':
            H[2, 0] /= factor  # perspective scales inversely
            H[2, 1] /= factor
        return PlanarTransform(self.type, matrix=H)

    def __repr__(self):
        return f"PlanarTransform('{self.type}')"


