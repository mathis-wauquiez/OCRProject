import torch
import torch.nn.functional as F

from .gradients import Gradients
from .transformations import PlanarTransform
from .erfs import ERROR_FUNCTIONS


class InverseCompositional:
    """
    Single-scale modified inverse compositional algorithm (Algorithm 3).
    
    Estimates transformation p such that I1(x) ≈ I2(Ψ(x; p))
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
                 lambda_decay=0.9):
        """
        Args:
            transform_type: 'translation', 'euclidean', 'similarity', 'affinity', 'homography'
            gradient_method: 'central', 'farid3', 'farid5', 'gaussian'
            error_function: 'l2', 'lorentzian', 'geman_mcclure', 'charbonnier'
            delta: Boundary margin for discarding pixels (recommended: 5)
            epsilon: Convergence threshold
            max_iter: Maximum iterations
            lambda_init: Initial $$\lambda$$ for robust error functions
            lambda_min: Minimum $$\lambda$$
            lambda_decay: $$\lambda$$ multiplier per iteration (0.9 in paper)
        """
        self.transform_type = transform_type
        self.gradient_method = gradient_method
        self.delta = delta
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.lambda_init = lambda_init
        self.lambda_min = lambda_min
        self.lambda_decay = lambda_decay
        
        if error_function not in ERROR_FUNCTIONS:
            raise ValueError(f"Unknown error function: {error_function}")
        self.error_fn_class = ERROR_FUNCTIONS[error_function]
        self.robust = error_function != 'l2'

    def _prefilter(self, img, grad_module):
        """Prefilter: $$\tilde I = k^T * k * I$$ (Eq. 29)"""
        out = grad_module._conv_h(img, grad_module.k_h)
        out = grad_module._conv_v(out, grad_module.k_v)
        return out

    def run(self, I1, I2, p_init=None):
        """
        Run single-scale inverse compositional algorithm.
        
        Args:
            I1: Reference image (B, C, H, W) or (C, H, W)
            I2: Warped image (B, C, H, W) or (C, H, W)  
            p_init: Initial transformation (PlanarTransform) or None for identity
            
        Returns:
            Estimated transformation (PlanarTransform)
        """
        # Handle dimensions
        if I1.dim() == 3: I1 = I1.unsqueeze(0)
        if I2.dim() == 3: I2 = I2.unsqueeze(0)
        
        B, C, H, W = I1.shape
        device, dtype = I1.device, I1.dtype
        
        # Initialize transformation
        T = p_init if p_init else PlanarTransform(self.transform_type, device=device, dtype=dtype)
        
        # ─────────────────────────────────────────────────────────────────────
        # PRECOMPUTATIONS (Lines 1-5 of Algorithm 3)
        # ─────────────────────────────────────────────────────────────────────
        
        # Gradient module
        if type(self.gradient_method) == str:
            grad = Gradients(self.gradient_method, C, device)
        else:
            grad = self.gradient_method
        
        # Gradient $$\nabla \tilde I_1$$ (Eqs. 30-31)
        dI1_dx, dI1_dy = grad(I1)
        
        # Prefiltered images (Eq. 29)
        I1_tilde = self._prefilter(I1, grad)
        I2_tilde = self._prefilter(I2, grad)
        
        # Jacobian $$J(x,y)$$ at identity (Table 1)
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        J = PlanarTransform(self.transform_type).jacobian(xx, yy)  # (H, W, 2, n_params)
        
        # $$G = \nabla \tilde I_1^T J$$ (Eq. 7):  $$G[b,c,h,w,p] = \sum_i \nabla I[b,c,h,w,i] * J[h,w,i,p]$$
        grad_I = torch.stack([dI1_dx, dI1_dy], dim=-1)  # (B, C, H, W, 2)
        G = torch.einsum('bchwd,hwdp->bchwp', grad_I, J)  # (B, C, H, W, n_params)
        
        # $$G^T G$$ precomputed (for Eq. 13)
        GTG = torch.einsum('bchwp,bchwq->bhwpq', G, G)  # (B, H, W, n_params, n_params)
        
        # ─────────────────────────────────────────────────────────────────────
        # INCREMENTAL REFINEMENT (Lines 6-20 of Algorithm 3)
        # ─────────────────────────────────────────────────────────────────────
        
        error_fn = self.error_fn_class(self.lambda_init)
        
        for j in range(self.max_iter):
            # Update $$\lambda$$ (Section 2.3): $$\lambda_j= max(0.9^j \lambda _0, 5)$$
            if self.robust:
                error_fn.lam = max(self.lambda_decay ** j * self.lambda_init, self.lambda_min)
            
            # Visibility mask $$\Omega^{\delta,p}$$ (Eq. 25)
            mask = T.visibility_mask(H, W, delta=self.delta).float()  # (H, W)
            
            # Warped prefiltered image $$\tilde I _2(\Psi(x; p))$$
            I2_warped = T.warp(I2_tilde)
            
            # Difference image $$DI$$ (Eq. 32)
            DI = I2_warped - I1_tilde  # (B, C, H, W)
            
            # Weight ρ̃'(x) = ρ'(||D̃I(x)||²)
            DI_norm2 = (DI ** 2).sum(dim=1)  # (B, H, W)
            weight = error_fn.rho_prime(DI_norm2) * mask  # (B, H, W)
            
            # b = Σ_x ρ̃'(x) · G(x)^T D̃I(x) (Eq. 28)
            b = torch.einsum('bhw,bchwp,bchw->bp', weight, G, DI)
            
            # H = Σ_x ρ̃'(x) · G(x)^T G(x) (Eq. 27)
            H_mat = torch.einsum('bhw,bhwpq->bpq', weight, GTG)
            
            # Δp = H⁻¹ b (Eq. 26)
            delta_p = torch.linalg.solve(H_mat, b)  # (B, n_params)
            
            # Check convergence (Eq. 19)
            if delta_p.norm(dim=-1).max() <= self.epsilon:
                break
            
            # Update: Ψ(·; pⱼ) = Ψ(·; pⱼ₋₁) ∘ Ψ(·; Δpⱼ)⁻¹ (Eq. 3)
            T_delta = PlanarTransform(self.transform_type, params=delta_p[0], 
                                      device=device, dtype=dtype)
            T = T @ T_delta.inv
        
        return T
