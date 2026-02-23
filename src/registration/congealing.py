"""
Congealing with Robust Latent Template — joint alignment of character clusters.

Alternating optimization: fix transforms → median template,
fix template → IC registration per image.
"""

from __future__ import annotations

import torch
from typing import Optional, Dict, Any, List

from .transformations import PlanarTransform
from .alignment_utils import create_default_aligner, pairwise_align, warp_image, median_template


class Congealing:
    """Joint alignment with robust latent template.

    Parameters
    ----------
    n_alternations : int
        Number of alternating optimization rounds.
    """

    def __init__(self, n_alternations=10, device='cuda', aligner=None):
        self.n_alternations = n_alternations
        self.device = device
        self.aligner = aligner or create_default_aligner(device)

    def congeal(self, images, medoid_idx=None):
        """Run congealing on a cluster of character images.

        Returns dict with keys: aligned, transforms, template,
        templates_per_iter, residuals_per_iter.
        """
        m = len(images)
        if m <= 1:
            T_id = PlanarTransform('homography', device='cpu')
            return {
                'aligned': list(images), 'transforms': [T_id] * m,
                'template': images[0] if m else None,
                'templates_per_iter': list(images), 'residuals_per_iter': [0.0] * m,
            }

        template = images[medoid_idx or 0].clone()
        templates_per_iter = [template.clone()]
        residuals_per_iter = []

        for _ in range(self.n_alternations):
            aligned, transforms, total_res = [], [], 0.0
            for i in range(m):
                T, warped = pairwise_align(self.aligner, template, images[i], self.device)
                warped_img = warp_image(images[i], T, self.device)
                aligned.append(warped_img)
                transforms.append(T)
                total_res += (template - warped_img).abs().mean().item()

            residuals_per_iter.append(total_res / m)
            template = median_template(aligned)
            templates_per_iter.append(template.clone())

        return {
            'aligned': aligned, 'transforms': transforms, 'template': template,
            'templates_per_iter': templates_per_iter, 'residuals_per_iter': residuals_per_iter,
        }
