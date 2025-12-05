"""
Multiscale Inverse Compositional Algorithm (Algorithm 4)

Briand, Facciolo, Sánchez - "Improvements of the Inverse Compositional Algorithm 
for Parametric Motion Estimation", IPOL 2018
"""

import torch
import torch.nn.functional as F

from .single_scale import InverseCompositional
from .transformations import PlanarTransform
from .gaussian_pyramid import GaussianPyramid


class MultiscaleIC:
    """
    Multiscale modified inverse compositional algorithm (Algorithm 4).
    
    Coarse-to-fine estimation using Gaussian pyramid.
    """
    
    def __init__(self,
                 singleScaleIC: InverseCompositional,
                 gaussianPyramid: GaussianPyramid,
                 first_scale: int = 0,
                 grayscale: bool = True):
        """
        Args:
            singleScaleIC: Single-scale inverse compositional solver
            gaussianPyramid: Gaussian pyramid builder
            first_scale: First (finest) scale to use, s0 ∈ {0, ..., N-1} (Section 3.4)
                         0 = use all scales, 1 = skip finest scale, etc.
            grayscale: Convert to grayscale before processing (Section 3.1)
        """
        self.ic = singleScaleIC
        self.pyramid = gaussianPyramid
        self.first_scale = first_scale
        self.grayscale = grayscale
        
        # Get eta from pyramid for scaling transforms
        self.eta = gaussianPyramid.eta
        
        # Get transform type from IC for creating identity transforms
        self.transform_type = singleScaleIC.transform_type
    
    def _to_grayscale(self, img):
        """Convert RGB to grayscale (Section 3.1)."""
        if img.shape[1] == 1:
            return img
        # Standard luminance weights
        return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    
    def run(self, I1, I2, p_init=None, return_all_scales=False):
        """
        Run multiscale inverse compositional algorithm (Algorithm 4).
        
        Args:
            I1: Reference image (B, C, H, W) or (C, H, W)
            I2: Target image (B, C, H, W) or (C, H, W)
            p_init: Initial transformation (PlanarTransform) or None
            return_all_scales: If True, return dict with per-scale results
            
        Returns:
            Estimated transformation (PlanarTransform)
            If return_all_scales: dict with 'transform', 'transforms_per_scale', 'pyramid_sizes'
        """
        # Handle dimensions
        if I1.dim() == 3: I1 = I1.unsqueeze(0)
        if I2.dim() == 3: I2 = I2.unsqueeze(0)
        
        B, C, H, W = I1.shape
        device, dtype = I1.device, I1.dtype
        
        # Convert to grayscale (Section 3.1)
        if self.grayscale and C > 1:
            I1_proc = self._to_grayscale(I1)
            I2_proc = self._to_grayscale(I2)
        else:
            I1_proc = I1
            I2_proc = I2
        
        # Build Gaussian pyramids (Lines 1-2 of Algorithm 4)
        pyr1 = self.pyramid(I1_proc)  # [fine, ..., coarse]
        pyr2 = self.pyramid(I2_proc)
        
        n_scales = len(pyr1)
        
        # Initialize transformation at coarsest scale (Line 3)
        if p_init is not None:
            # Scale initial transform to coarsest level
            T = p_init
            for _ in range(n_scales - 1):
                T = T.scale(self.eta)
        else:
            T = PlanarTransform(self.transform_type, device=device, dtype=dtype)
        
        # Storage for per-scale results
        transforms_per_scale = []
        pyramid_sizes = [(p.shape[2], p.shape[3]) for p in pyr1]
        
        # Coarse-to-fine iteration (Lines 4-8 of Algorithm 4)
        # Start from coarsest (index n_scales-1) to first_scale
        for s in range(n_scales - 1, self.first_scale - 1, -1):
            I1_s = pyr1[s]
            I2_s = pyr2[s]
            
            # Run single-scale IC at this level (Line 6)
            T = self.ic.run(I1_s, I2_s, p_init=T)
            
            transforms_per_scale.append(T)
            
            # Scale transform to next finer level (Line 7)
            if s > self.first_scale:
                T = T.scale(1.0 / self.eta)
        
        if return_all_scales:
            return {
                'transform': T,
                'transforms_per_scale': transforms_per_scale[::-1],  # Fine to coarse order
                'pyramid_sizes': pyramid_sizes,
                'n_scales': n_scales
            }
        
        return T
