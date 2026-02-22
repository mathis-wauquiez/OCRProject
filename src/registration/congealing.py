"""
Congealing with Robust Latent Template — joint alignment of character clusters.

Instead of aligning pairwise, jointly optimizes all transformations and a
latent template using alternating optimization:
  1. Fix transforms, update template (pixel-wise median)
  2. Fix template, update each transform (IC registration)

The pixel-wise median is robust to outliers and produces sharp templates
unlike the Fréchet mean which blurs under residual misalignment.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from .multiscale_registration import MultiscaleIC
from .transformations import PlanarTransform
from .single_scale import InverseCompositional
from .gaussian_pyramid import GaussianPyramid
from .gradients import Gradients


def _default_aligner(device: str = 'cuda') -> MultiscaleIC:
    """Create a default multiscale IC aligner."""
    gradient_method = Gradients(method='farid5', C=1, device=device)
    ic = InverseCompositional(
        transform_type='homography',
        gradient_method=gradient_method,
        error_function='lorentzian',
        delta=5,
        epsilon=1e-3,
        max_iter=5,
    )
    pyramid = GaussianPyramid(
        eta=0.5, sigma_0=0.6, ksize_factor=8, min_size=32,
    )
    return MultiscaleIC(singleScaleIC=ic, gaussianPyramid=pyramid)


class Congealing:
    """
    Congealing: joint alignment with robust latent template.

    Objective:
        min_{I_bar, {T_i}} sum_i L(I_i ∘ T_i, I_bar) + lambda * sum_i R(T_i)

    Alternating optimization:
        1. Fix {T_i} → I_bar = pixel-wise median of {I_i ∘ T_i}
        2. Fix I_bar → each T_i = argmin_T L(I_i ∘ T, I_bar) + lambda * R(T)

    Parameters
    ----------
    n_alternations : int
        Number of alternating optimization rounds.
    reg_lambda : float
        Regularization weight on transforms (penalizes deviation from Id).
    device : str
        Torch device.
    aligner : MultiscaleIC or None
        Registration engine.  If None, a default one is created.
    """

    def __init__(
        self,
        n_alternations: int = 10,
        reg_lambda: float = 0.0,
        device: str = 'cuda',
        aligner: Optional[MultiscaleIC] = None,
    ):
        self.n_alternations = n_alternations
        self.reg_lambda = reg_lambda
        self.device = device
        self.aligner = aligner or _default_aligner(device)

    def _register_to_template(
        self, template: torch.Tensor, image: torch.Tensor,
    ) -> Tuple[PlanarTransform, torch.Tensor]:
        """Register an image to the template. Returns (T, warped_image)."""
        T_ref = template.unsqueeze(0).unsqueeze(0).to(self.device)
        T_img = image.unsqueeze(0).unsqueeze(0).to(self.device)
        try:
            T = self.aligner.run(T_ref, T_img)
            warped = T.warp(T_img)[:, 0]
            H, W = template.shape
            mask = T.visibility_mask(H, W, delta=0)
            warped[~mask] = 0
            return T, warped[0].cpu()
        except Exception:
            return PlanarTransform('homography', device=self.device), image

    def congeal(
        self,
        images: List[torch.Tensor],
        medoid_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run congealing on a cluster of character images.

        Parameters
        ----------
        images : list of Tensor (H, W)
            Character images in the cluster.
        medoid_idx : int or None
            Index of the initial template image.
            If None, uses the first image.

        Returns
        -------
        dict with keys:
            ``aligned`` : list of Tensor (H, W) — final aligned images
            ``transforms`` : list of PlanarTransform — final transforms
            ``template`` : Tensor (H, W) — final robust template
            ``templates_per_iter`` : list of Tensor — template at each iteration
            ``residuals_per_iter`` : list of float — mean residual per iteration
        """
        m = len(images)
        if m == 0:
            return {
                'aligned': [], 'transforms': [],
                'template': None, 'templates_per_iter': [],
                'residuals_per_iter': [],
            }
        if m == 1:
            return {
                'aligned': [images[0]],
                'transforms': [PlanarTransform('homography', device='cpu')],
                'template': images[0],
                'templates_per_iter': [images[0]],
                'residuals_per_iter': [0.0],
            }

        # Initialize template as medoid
        if medoid_idx is None:
            medoid_idx = 0
        template = images[medoid_idx].clone()

        # Initialize transforms as identity
        transforms = [
            PlanarTransform('homography', device=self.device)
            for _ in range(m)
        ]
        aligned = list(images)  # shallow copy

        templates_per_iter = [template.clone()]
        residuals_per_iter = []

        for iteration in range(self.n_alternations):
            # Step 2: Fix template, update transforms
            new_aligned = []
            new_transforms = []
            total_residual = 0.0

            for i in range(m):
                T, warped = self._register_to_template(template, images[i])
                new_aligned.append(warped)
                new_transforms.append(T)

                # Compute residual
                diff = (template - warped).abs()
                total_residual += diff.mean().item()

            aligned = new_aligned
            transforms = new_transforms
            mean_residual = total_residual / m
            residuals_per_iter.append(mean_residual)

            # Step 1: Fix transforms, update template (pixel-wise median)
            stack = torch.stack(aligned)  # (m, H, W)
            template = stack.median(dim=0).values
            templates_per_iter.append(template.clone())

        return {
            'aligned': aligned,
            'transforms': transforms,
            'template': template,
            'templates_per_iter': templates_per_iter,
            'residuals_per_iter': residuals_per_iter,
        }
