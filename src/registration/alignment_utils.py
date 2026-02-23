"""Shared utilities for cluster alignment (MST, congealing)."""

from __future__ import annotations

import torch
from typing import Tuple, List

from .transformations import PlanarTransform
from .multiscale_registration import MultiscaleIC
from .single_scale import InverseCompositional
from .gaussian_pyramid import GaussianPyramid
from .gradients import Gradients


def create_default_aligner(device: str = 'cuda') -> MultiscaleIC:
    """Create a default multiscale IC aligner for character registration."""
    gradient_method = Gradients(method='farid5', C=1, device=device)
    ic = InverseCompositional(
        transform_type='homography',
        gradient_method=gradient_method,
        error_function='lorentzian',
        delta=5, epsilon=1e-3, max_iter=5,
    )
    pyramid = GaussianPyramid(eta=0.5, sigma_0=0.6, ksize_factor=8, min_size=32)
    return MultiscaleIC(singleScaleIC=ic, gaussianPyramid=pyramid)


def pairwise_align(aligner, I1, I2, device='cuda'):
    """Align I2 onto I1. Returns (transform, residual)."""
    I1_4d = I1.unsqueeze(0).unsqueeze(0).to(device)
    I2_4d = I2.unsqueeze(0).unsqueeze(0).to(device)
    try:
        T = aligner.run(I1_4d, I2_4d)
        I2w = T.warp(I2_4d)
        mask = T.visibility_mask(I1_4d.shape[2], I1_4d.shape[3], delta=0)
        diff = (I1_4d[:, 0] - I2w[:, 0]).abs()
        diff[~mask] = 0
        residual = diff.sum().item() / max(mask.float().sum().item(), 1.0)
        return T, residual
    except Exception:
        return PlanarTransform('homography', device=device), float('inf')


def warp_image(image, transform, device='cuda'):
    """Warp a (H, W) image by a transform, zero-padding outside."""
    I = image.unsqueeze(0).unsqueeze(0).to(device)
    warped = transform.warp(I)[:, 0]
    H, W = image.shape
    mask = transform.visibility_mask(H, W, delta=0)
    warped[~mask] = 0
    return warped[0].cpu()


def median_template(aligned: List[torch.Tensor]) -> torch.Tensor:
    """Pixel-wise median of aligned images (robust to outliers)."""
    return torch.stack(aligned).median(dim=0).values
