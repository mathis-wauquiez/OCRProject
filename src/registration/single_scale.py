"""
Single-scale modified inverse compositional algorithm (Algorithm 3).

Fully batched: given ``(B, C, H, W)`` inputs, estimates *B* independent
transformations in parallel.

Ref: Briand, Facciolo, Sánchez – IPOL 2018.

THIS CODE VERSION IS AI-TWEAKED. ORIGINAL, HUMAN MADE FILE IN ARCHIVE

Example
-------
>>> import torch
>>> from single_scale import InverseCompositional
>>> from transformations import PlanarTransform
>>>
>>> # ── Recover a known translation ───────────────────────────────────
>>> H, W = 128, 128
>>> I1 = torch.randn(1, 1, H, W)
>>>
>>> # Apply a 3-pixel shift in x (ground-truth)
>>> T_gt = PlanarTransform('translation', params=torch.tensor([3.0, 0.0]))
>>> I2 = T_gt.warp(I1)
>>>
>>> ic = InverseCompositional(
...     transform_type='translation',
...     error_function='lorentzian',
...     max_iter=50,
...     epsilon=1e-4,
... )
>>> T_est = ic.run(I1, I2)
>>> print(T_est.matrix[0])          # should ≈ T_gt.matrix[0]
>>>
>>> # ── Batched estimation ────────────────────────────────────────────
>>> B = 4
>>> I1_batch = torch.randn(B, 1, 64, 64)
>>> shifts = torch.stack([torch.tensor([float(i), 0.0]) for i in range(B)])
>>> T_batch = PlanarTransform('translation', params=shifts)
>>> I2_batch = T_batch.warp(I1_batch)
>>>
>>> T_est_batch = ic.run(I1_batch, I2_batch)
>>> print(T_est_batch.batch_size)   # 4
>>>
>>> # ── Warm-start from an initial guess ──────────────────────────────
>>> T_init = PlanarTransform('translation',
...                          params=torch.tensor([[2.5, 0.1]]).expand(B, -1))
>>> T_refined = ic.run(I1_batch, I2_batch, p_init=T_init)
>>>
>>> # ── Homography estimation on RGB input ────────────────────────────
>>> ic_h = InverseCompositional(
...     transform_type='homography',
...     gradient_method='farid5',
...     error_function='geman_mcclure',
...     lambda_init=80.0,
...     lambda_decay=0.9,
...     dtype=torch.float64,
... )
>>> I1_rgb = torch.randn(1, 3, 128, 128, dtype=torch.float64)
>>> T_h = PlanarTransform('homography',
...                        params=torch.tensor([0.01, 0.0, 2.0,
...                                             0.0, -0.01, 1.0,
...                                             1e-5, 0.0],
...                                            dtype=torch.float64))
>>> I2_rgb = T_h.warp(I1_rgb)
>>> T_est_h = ic_h.run(I1_rgb, I2_rgb)
>>> print((T_est_h.matrix - T_h.matrix).abs().max().item() < 0.1)  # True
"""

import torch
import torch.nn.functional as F

from .gradients import Gradients
from .transformations import PlanarTransform
from .erfs import ERROR_FUNCTIONS


class InverseCompositional:
    """
    Batched single-scale modified inverse compositional algorithm.

    Estimates *B* transformations ``p_b`` such that
    ``I1_b(x) ≈ I2_b(Ψ(x; p_b))``.

    Example
    -------
    >>> import torch
    >>> from single_scale import InverseCompositional
    >>>
    >>> ic = InverseCompositional(
    ...     transform_type='affinity',
    ...     error_function='charbonnier',
    ...     max_iter=40,
    ... )
    >>> I1 = torch.randn(2, 1, 64, 64)
    >>> I2 = torch.randn(2, 1, 64, 64)
    >>> T = ic.run(I1, I2)
    >>> print(T)   # PlanarTransform('affinity', B=2)
    """

    def __init__(self,
                 transform_type='homography',
                 gradient_method='farid5',
                 error_function='lorentzian',
                 delta=5,
                 epsilon=1e-3,
                 max_iter=30,
                 lambda_init=80.0,
                 lambda_min=5.0,
                 lambda_decay=0.9,
                 dtype=torch.float32):
        """
        Args:
            transform_type: ``'translation'`` | ``'euclidean'`` |
                ``'similarity'`` | ``'affinity'`` | ``'homography'``
            gradient_method: ``'central'`` | ``'farid3'`` | ``'farid5'`` |
                ``'gaussian'`` **or** a pre-built :class:`Gradients` instance.
            error_function: ``'l2'`` | ``'lorentzian'`` | ``'geman_mcclure'``
                | ``'charbonnier'``
            delta: Boundary margin (pixels) for visibility mask.
            epsilon: Convergence threshold on ``‖Δp‖``.
            max_iter: Maximum IC iterations.
            lambda_init: Initial λ for robust error functions.
            lambda_min: Minimum λ.
            lambda_decay: Multiplicative decay per iteration.
            dtype: ``torch.float32`` or ``torch.float64``.
        """
        self.transform_type = transform_type
        self.gradient_method = gradient_method
        self.delta = delta
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.lambda_init = lambda_init
        self.lambda_min = lambda_min
        self.lambda_decay = lambda_decay
        self.dtype = dtype

        if error_function not in ERROR_FUNCTIONS:
            raise ValueError(f"Unknown error function: {error_function}")
        self.error_fn_class = ERROR_FUNCTIONS[error_function]
        self.robust = error_function != 'l2'

        # Lazy-initialised Gradients module (depends on C / device)
        self._grad = None
        self._grad_key = None           # (method, C, device, dtype)

        # Jacobian cache: key → (H, W, 2, n_params) tensor
        self._jacobian_cache = {}

    # ── internal helpers ────────────────────────────────────────────────────
    def _get_grad(self, C, device):
        """Return (lazily-built) Gradients module, re-create if config changed."""
        if isinstance(self.gradient_method, Gradients):
            return self.gradient_method

        key = (self.gradient_method, C, str(device), self.dtype)
        if self._grad is None or self._grad_key != key:
            self._grad = Gradients(self.gradient_method, C, device,
                                   dtype=self.dtype)
            self._grad_key = key
        return self._grad

    def _get_jacobian(self, H, W, device):
        """Cached Jacobian at identity — depends only on (H, W, type, dtype)."""
        key = (H, W, self.transform_type, str(device), self.dtype)
        if key not in self._jacobian_cache:
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device, dtype=self.dtype),
                torch.arange(W, device=device, dtype=self.dtype),
                indexing='ij')
            self._jacobian_cache[key] = (
                PlanarTransform(self.transform_type, dtype=self.dtype)
                .jacobian(xx, yy))
        return self._jacobian_cache[key]

    @staticmethod
    def _prefilter(img, grad_module):
        """Pre-filter: Ĩ = kᵀ * k * I  (Eq. 29)."""
        out = grad_module._conv_h(img, grad_module.k_h)
        out = grad_module._conv_v(out, grad_module.k_v)
        return out

    # ── main entry point ────────────────────────────────────────────────────
    def run(self, I1, I2, p_init=None):
        """
        Run the batched inverse compositional algorithm.

        Args:
            I1: Reference images ``(B, C, H, W)`` or ``(C, H, W)``.
            I2: Target images ``(B, C, H, W)`` or ``(C, H, W)``.
            p_init: Initial :class:`PlanarTransform` (batch_size *B* or 1),
                or ``None`` for identity.

        Returns:
            :class:`PlanarTransform` with ``batch_size == B``.
        """
        if I1.dim() == 3:
            I1 = I1.unsqueeze(0)
        if I2.dim() == 3:
            I2 = I2.unsqueeze(0)

        B, C, H, W = I1.shape
        device = I1.device

        # Cast to requested dtype
        I1 = I1.to(dtype=self.dtype)
        I2 = I2.to(dtype=self.dtype)

        # ── initialise transformation ───────────────────────────────────────
        if p_init is not None:
            T = p_init
            # Broadcast B=1 → B if needed
            if T.batch_size == 1 and B > 1:
                T = PlanarTransform(self.transform_type,
                                    matrix=T.matrix.expand(B, -1, -1).clone(),
                                    device=device, dtype=self.dtype)
        else:
            T = PlanarTransform(self.transform_type, batch_size=B,
                                device=device, dtype=self.dtype)

        # ── PRECOMPUTATIONS (Lines 1-5, Algorithm 3) ────────────────────────
        grad = self._get_grad(C, device)

        # Gradients ∇Ĩ₁
        dI1_dx, dI1_dy = grad(I1)

        # Pre-filtered images
        I1_tilde = self._prefilter(I1, grad)
        I2_tilde = self._prefilter(I2, grad)

        # Jacobian J(x, y) at identity — cached, shape (H, W, 2, n_params)
        J = self._get_jacobian(H, W, device)

        # G = ∇Ĩ₁ᵀ J   →  (B, C, H, W, n_params)
        grad_I = torch.stack([dI1_dx, dI1_dy], dim=-1)       # (B, C, H, W, 2)
        G = torch.einsum('bchwd,hwdp->bchwp', grad_I, J)     # (B, C, H, W, P)

        # GᵀG precomputed  →  (B, H, W, P, P)
        GTG = torch.einsum('bchwp,bchwq->bhwpq', G, G)

        # ── ITERATIVE REFINEMENT (Lines 6-20, Algorithm 3) ─────────────────
        error_fn = self.error_fn_class(self.lambda_init)

        for j in range(self.max_iter):
            # λ schedule
            if self.robust:
                error_fn.lam = max(self.lambda_decay ** j * self.lambda_init,
                                   self.lambda_min)

            # Visibility mask Ω^{δ,p}  →  (B, H, W) float
            mask = T.visibility_mask(H, W, delta=self.delta).to(self.dtype)

            # Warped pre-filtered image Ĩ₂(Ψ(x; p))
            I2_warped = T.warp(I2_tilde)

            # Difference image D̃I  (B, C, H, W)
            DI = I2_warped - I1_tilde

            # Robust weight ρ̃'  (B, H, W)
            DI_norm2 = (DI ** 2).sum(dim=1)
            weight = error_fn.rho_prime(DI_norm2) * mask

            # b = Σ_x ρ̃'(x) · G(x)ᵀ D̃I(x)   →  (B, P)
            b = torch.einsum('bhw,bchwp,bchw->bp', weight, G, DI)

            # H = Σ_x ρ̃'(x) · G(x)ᵀ G(x)     →  (B, P, P)
            H_mat = torch.einsum('bhw,bhwpq->bpq', weight, GTG)

            # Δp = H⁻¹ b  →  (B, P)
            delta_p = torch.linalg.solve(H_mat, b)

            # Convergence check
            if delta_p.norm(dim=-1).max().item() <= self.epsilon:
                break

            # Compositional update: Ψ(·; pⱼ) = Ψ(·; pⱼ₋₁) ∘ Ψ(·; Δpⱼ)⁻¹
            T_delta = PlanarTransform(self.transform_type, params=delta_p,
                                      device=device, dtype=self.dtype)
            T = T @ T_delta.inv

        return T