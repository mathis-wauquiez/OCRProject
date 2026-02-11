"""
Batched planar transformations for the inverse compositional algorithm.

All transforms store a (B, 3, 3) matrix internally.  B=1 is the
"scalar" case and broadcasts automatically against any batch size.

Ref: Briand, Facciolo, Sánchez – IPOL 2018, Table 1.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F


class Transformation(ABC):
    """Base class for a set of transformations with a group structure."""

    @abstractmethod
    def jacobian(self, xx, yy):
        """∂Ψ/∂p at p=0.  Returns (*xx.shape, 2, n_params)."""
        ...

    @abstractmethod
    def warp(self, img_batch, mode='bicubic', padding_mode='zeros'):
        ...

    @property
    @abstractmethod
    def inv(self) -> 'Transformation':
        ...

    @abstractmethod
    def compose(self, other: 'Transformation') -> 'Transformation':
        ...

    @abstractmethod
    def visibility_mask(self, *args):
        ...

    def __matmul__(self, other: 'Transformation') -> 'Transformation':
        return self.compose(other)


class PlanarTransform(Transformation):
    """
    Unified batched planar transformation.

    Internally stores ``self.matrix`` of shape **(B, 3, 3)**.
    B = 1 broadcasts against any other batch size in :meth:`compose` /
    :meth:`warp`.
    """

    # ── Jacobian functions (evaluated at p = 0) ─────────────────────────────
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
        J[..., 0, 6], J[..., 0, 7] = -x * x, -x * y
        J[..., 1, 6], J[..., 1, 7] = -x * y, -y * y
        return J

    # ── params → matrix  (all return (B, 3, 3)) ────────────────────────────
    @staticmethod
    def _M_translation(p):
        B = p.shape[0]
        H = torch.eye(3, device=p.device, dtype=p.dtype).unsqueeze(0).expand(B, -1, -1).clone()
        H[:, 0, 2], H[:, 1, 2] = p[:, 0], p[:, 1]
        return H

    @staticmethod
    def _M_euclidean(p):
        B = p.shape[0]
        c, s = torch.cos(p[:, 2]), torch.sin(p[:, 2])
        H = torch.zeros(B, 3, 3, device=p.device, dtype=p.dtype)
        H[:, 0, 0], H[:, 0, 1], H[:, 0, 2] = c, -s, p[:, 0]
        H[:, 1, 0], H[:, 1, 1], H[:, 1, 2] = s, c, p[:, 1]
        H[:, 2, 2] = 1
        return H

    @staticmethod
    def _M_similarity(p):
        B = p.shape[0]
        H = torch.zeros(B, 3, 3, device=p.device, dtype=p.dtype)
        H[:, 0, 0], H[:, 0, 1], H[:, 0, 2] = 1 + p[:, 2], -p[:, 3], p[:, 0]
        H[:, 1, 0], H[:, 1, 1], H[:, 1, 2] = p[:, 3], 1 + p[:, 2], p[:, 1]
        H[:, 2, 2] = 1
        return H

    @staticmethod
    def _M_affinity(p):
        B = p.shape[0]
        H = torch.zeros(B, 3, 3, device=p.device, dtype=p.dtype)
        H[:, 0, 0], H[:, 0, 1], H[:, 0, 2] = 1 + p[:, 2], p[:, 3], p[:, 0]
        H[:, 1, 0], H[:, 1, 1], H[:, 1, 2] = p[:, 4], 1 + p[:, 5], p[:, 1]
        H[:, 2, 2] = 1
        return H

    @staticmethod
    def _M_homography(p):
        B = p.shape[0]
        H = torch.zeros(B, 3, 3, device=p.device, dtype=p.dtype)
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

    def __init__(self, ttype='homography', params=None, matrix=None,
                 batch_size=1, device=None, dtype=torch.float32):
        """
        Args:
            ttype: Transform type name.
            params: ``(n_params,)`` or ``(B, n_params)`` parameter vector.
            matrix: ``(3, 3)`` or ``(B, 3, 3)`` transformation matrix.
            batch_size: Used only when constructing identity (no params / matrix).
            device, dtype: Tensor placement.
        """
        if ttype not in self.TYPES:
            raise ValueError(f"Unknown type: {ttype}")

        self.type = ttype
        self.n_params, self._J, self._M = self.TYPES[ttype]

        if matrix is not None:
            m = matrix.clone()
            if device is not None:
                m = m.to(device=device)
            if dtype is not None:
                m = m.to(dtype=dtype)
            self.matrix = m if m.dim() == 3 else m.unsqueeze(0)
        elif params is not None:
            p = torch.as_tensor(params, device=device, dtype=dtype)
            p = p.unsqueeze(0) if p.dim() == 1 else p          # (B, n_params)
            self.matrix = self._M(p)                            # (B, 3, 3)
        else:
            self.matrix = (torch.eye(3, device=device, dtype=dtype)
                           .unsqueeze(0)
                           .expand(batch_size, -1, -1)
                           .clone())                            # (B, 3, 3)

    # ── helpers ─────────────────────────────────────────────────────────────
    @property
    def batch_size(self):
        return self.matrix.shape[0]

    @property
    def device(self):
        return self.matrix.device

    @property
    def dtype(self):
        return self.matrix.dtype

    def _ensure_compatible(self, other):
        """Broadcast B=1 against B=N.  Returns two (B, 3, 3) tensors."""
        if self.batch_size == other.batch_size:
            return self.matrix, other.matrix
        if self.batch_size == 1:
            return self.matrix.expand_as(other.matrix), other.matrix
        if other.batch_size == 1:
            return self.matrix, other.matrix.expand_as(self.matrix)
        raise ValueError(
            f"Incompatible batch sizes: {self.batch_size} vs {other.batch_size}")

    # ── core operations ─────────────────────────────────────────────────────
    def transform_points(self, xx, yy):
        """
        Apply transformation:  ``(x', y') = Ψ(x, y; p)``.

        Args:
            xx, yy: ``(H, W)`` coordinate grids.

        Returns:
            ``(xx_t, yy_t)`` each of shape ``(B, H, W)``.
        """
        H_mat = self.matrix.to(device=xx.device, dtype=xx.dtype)  # (B, 3, 3)
        shape = xx.shape                                        # (H, W)
        HW = xx.numel()
        ones = torch.ones(1, HW, device=xx.device, dtype=xx.dtype)
        coords = torch.stack([xx.reshape(1, -1),
                               yy.reshape(1, -1),
                               ones], dim=1)                     # (1, 3, HW)
        warped = H_mat @ coords                                  # (B, 3, HW)
        denom = warped[:, 2:3, :]                                # (B, 1, HW)
        xx_t = (warped[:, 0:1, :] / denom).reshape(-1, *shape)  # (B, H, W)
        yy_t = (warped[:, 1:2, :] / denom).reshape(-1, *shape)  # (B, H, W)
        return xx_t, yy_t

    def jacobian(self, xx, yy):
        """∂Ψ/∂p at p=0.  Shape: ``(*xx.shape, 2, n_params)``."""
        return self._J(xx, yy)

    def warp(self, img, mode='bicubic', padding_mode='zeros'):
        """
        Warp image batch:  ``output[b, :, y, x] = input[b, :, Ψ_b(x, y)]``.

        Args:
            img: ``(B, C, H, W)``.

        ``self.matrix`` may be ``(B, 3, 3)`` or ``(1, 3, 3)`` (broadcasts).
        """
        B, C, H, W = img.shape
        device, dtype = img.device, img.dtype

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij')

        xx_t, yy_t = self.transform_points(xx, yy)               # (Bt, H, W)

        # Broadcast if transform is B=1 but img has B>1
        if xx_t.shape[0] == 1 and B > 1:
            xx_t = xx_t.expand(B, -1, -1)
            yy_t = yy_t.expand(B, -1, -1)

        grid = torch.stack([
            2.0 * xx_t / (W - 1) - 1.0,
            2.0 * yy_t / (H - 1) - 1.0,
        ], dim=-1)                                                # (B, H, W, 2)

        return F.grid_sample(img, grid, mode=mode,
                             padding_mode=padding_mode, align_corners=True)

    @property
    def inv(self):
        """Inverse transformation."""
        return PlanarTransform(self.type, matrix=torch.linalg.inv(self.matrix),
                               dtype=None)

    def compose(self, other):
        """Composition: ``(self ∘ other)(x) = self(other(x))``."""
        m_self, m_other = self._ensure_compatible(other)
        return PlanarTransform(self.type, matrix=m_self @ m_other, dtype=None)

    def __matmul__(self, other):
        return self.compose(other)

    def visibility_mask(self, H, W, delta=0):
        """
        ``Ω^{δ,p}`` mask (Eq. 23-25).  Returns ``(B, H, W)`` bool tensor.
        """
        device, dtype = self.device, self.dtype
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij')

        interior = ((xx >= delta) & (xx <= W - 1 - delta) &
                    (yy >= delta) & (yy <= H - 1 - delta))       # (H, W)

        xx_t, yy_t = self.transform_points(xx, yy)               # (B, H, W)
        mapped = ((xx_t >= delta) & (xx_t <= W - 1 - delta) &
                  (yy_t >= delta) & (yy_t <= H - 1 - delta))     # (B, H, W)

        return interior.unsqueeze(0) & mapped                     # (B, H, W)

    def scale(self, factor):
        """Scale transformation for pyramid level change."""
        H = self.matrix.clone()
        H[:, 0, 2] *= factor
        H[:, 1, 2] *= factor
        if self.type == 'homography':
            H[:, 2, 0] /= factor
            H[:, 2, 1] /= factor
        return PlanarTransform(self.type, matrix=H, dtype=None)

    def __repr__(self):
        return f"PlanarTransform('{self.type}', B={self.batch_size})"