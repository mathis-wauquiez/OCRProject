import torch

# ═══════════════════════════════════════════════════════════════════════════════
# ERROR FUNCTIONS (Table 2)
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorFunction:
    """Base class for error functions ρ(s²)."""
    def rho(self, s2): raise NotImplementedError
    def rho_prime(self, s2): raise NotImplementedError


class L2Error:
    """ρ(s²) = s², ρ'(s²) = 1"""
    def __init__(self, lam=None): pass
    def rho_prime(self, s2): return torch.ones_like(s2)


class LorentzianError(ErrorFunction):
    """ρ(s²) = log(1 + (s/λ)²), ρ'(s²) = 1/(s² + λ²)"""
    def __init__(self, lam=5.0): self.lam = lam
    def rho_prime(self, s2): return 1.0 / (s2 + self.lam ** 2)


class GemanMcClureError(ErrorFunction):
    """ρ(s²) = s²/(s² + λ²), ρ'(s²) = λ²/(s² + λ²)²"""
    def __init__(self, lam=5.0): self.lam = lam
    def rho_prime(self, s2): return (self.lam ** 2) / (s2 + self.lam ** 2) ** 2


class CharbonnierError(ErrorFunction):
    """ρ(s²) = 2√(s² + λ²), ρ'(s²) = 1/√(s² + λ²)"""
    def __init__(self, lam=5.0): self.lam = lam
    def rho_prime(self, s2): return 1.0 / torch.sqrt(s2 + self.lam ** 2)


ERROR_FUNCTIONS = {
    'l2': L2Error,
    'lorentzian': LorentzianError,
    'geman_mcclure': GemanMcClureError,
    'charbonnier': CharbonnierError,
}
