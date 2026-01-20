import torch
import torch.nn as nn
import numpy as np

from ..character_linking.params import GradientParameters
import cv2
import pandas as pd

from ..utils import connectedComponent


class GradientComputer:
    """Computes image gradients using various kernels."""
    
    KERNELS = {
        'central_differences': {
            'k': np.array([0, 1, 0]),
            'd': np.array([-.5, 0, .5])
        },
        'hypomode': {
            'k': np.array([0, 0.5, 0.5]),
            'd': np.array([0, -1, 1])
        },
        'farid_3x3': {
            'k': np.array([0.229879, 0.540242, 0.229879]),
            'd': np.array([-0.425287, 0, 0.425287])
        },
        'farid_5x5': {
            'k': np.array([0.037659, 0.249153, 0.426375, 0.249153, 0.037659]),
            'd': np.array([-0.109604, -0.276691, 0, 0.276691, 0.109604])
        }
    }
    
    def __init__(self, params: GradientParameters):
        self.params = params
        self._build_kernels()
    
    def _build_kernels(self):
        """Build convolution kernels for gradient computation."""
        device = self.params.device
        C = self.params.C
        
        if self.params.method == 'gaussian':
            ksize = int(self.params.grdt_sigma * self.params.ksize_factor) // 2 * 2 + 1
            k = cv2.getGaussianKernel(ksize, self.params.grdt_sigma)
            d, _ = cv2.getDerivKernels(dx=1, dy=0, ksize=ksize, normalize=True)
        elif self.params.method in self.KERNELS:
            k = self.KERNELS[self.params.method]['k']
            d = self.KERNELS[self.params.method]['d']
            ksize = len(k)
        else:
            raise ValueError(f'Unknown method: {self.params.method}')
        
        # Build convolution layers
        self.dx_conv = self._make_separable_conv(k, d, ksize, C, device)
        self.dy_conv = self._make_separable_conv(k, d, ksize, C, device, transpose=True)
        self.pre_conv = self._make_smoothing_conv(k, ksize, C, device)
    
    def _make_separable_conv(self, k, d, ksize, C, device, transpose=False):
        """Create separable 2D convolution for gradient."""
        if transpose:
            conv = nn.Sequential(
                nn.Conv2d(C, C, (ksize, 1), padding=(ksize//2, 0), bias=False, 
                         groups=C, padding_mode='reflect'),
                nn.Conv2d(C, C, (1, ksize), padding=(0, ksize//2), bias=False, 
                         groups=C, padding_mode='reflect')
            )
        else:
            conv = nn.Sequential(
                nn.Conv2d(C, C, (1, ksize), padding=(0, ksize//2), bias=False, 
                         groups=C, padding_mode='reflect'),
                nn.Conv2d(C, C, (ksize, 1), padding=(ksize//2, 0), bias=False, 
                         groups=C, padding_mode='reflect')
            )
        
        conv = conv.to(device).eval()
        
        with torch.no_grad():
            for i in range(C):
                if transpose:
                    conv[0].weight[i, 0, :, 0] = torch.tensor(k.flatten(), dtype=torch.float32, device=device)
                    conv[1].weight[i, 0, 0, :] = torch.tensor(d.flatten(), dtype=torch.float32, device=device)
                else:
                    conv[0].weight[i, 0, 0, :] = torch.tensor(k.flatten(), dtype=torch.float32, device=device)
                    conv[1].weight[i, 0, :, 0] = torch.tensor(d.flatten(), dtype=torch.float32, device=device)
        
        return conv
    
    def _make_smoothing_conv(self, k, ksize, C, device):
        """Create separable 2D convolution for smoothing."""
        conv = nn.Sequential(
            nn.Conv2d(C, C, (ksize, 1), padding=(ksize//2, 0), bias=False, 
                     groups=C, padding_mode='reflect'),
            nn.Conv2d(C, C, (1, ksize), padding=(0, ksize//2), bias=False, 
                     groups=C, padding_mode='reflect')
        ).to(device).eval()
        
        with torch.no_grad():
            for i in range(C):
                conv[0].weight[i, 0, :, 0] = torch.tensor(k.flatten(), dtype=torch.float32, device=device)
                conv[1].weight[i, 0, 0, :] = torch.tensor(k.flatten(), dtype=torch.float32, device=device)
        
        return conv
    
    def __call__(self, group_df: pd.DataFrame, database=None) -> pd.Series:
        """
        Compute gradients for all patches in a group (same image).
        
        Args:
            group_df: DataFrame subset containing patches from the same image
            database: PatchDatabase instance to retrieve the full image (optional)
                     If None, attempts to retrieve from group_df's parent if it's a PatchDatabase
        
        Returns:
            pd.Series with gradient information for each patch
        """
        device = self.params.device
        
        # Get image_idx from the first row (all rows in group have same image_idx)
        image_idx = group_df.iloc[0]['image_idx']
        
        # Retrieve full image from database
        if database is not None:
            # Database explicitly passed
            full_image = database.get_image(image_idx)
        elif hasattr(group_df, '_images'):
            # group_df is a PatchDatabase or has access to _images
            full_image = group_df._images[image_idx]
        elif hasattr(group_df, 'get_image'):
            # group_df has get_image method
            full_image = group_df.get_image(image_idx)
        else:
            # Fallback: try to get from parent (for grouped DataFrames)
            # This might not work in all cases
            raise ValueError(
                "Cannot retrieve full image. Please pass the PatchDatabase as the 'database' parameter, "
                "or ensure group_df has access to the full images via _images or get_image()."
            )
        
        # Convert to tensor
        image_tensor = torch.from_numpy(full_image).to(device)
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        elif image_tensor.ndim == 3:
            # Handle (C, H, W) format
            image_tensor = image_tensor.unsqueeze(0)
        
        # Compute gradients on full image
        with torch.no_grad():
            smoothed = self.pre_conv(image_tensor)
            dx = self.dx_conv(smoothed)
            dy = self.dy_conv(smoothed)
            magnitude = torch.sqrt(dx**2 + dy**2)
            orientation = (torch.atan2(dy, dx) % np.pi) / np.pi
        
        # Extract patches using bboxes
        results = []
        for _, row in group_df.iterrows():
            h1, w1, h2, w2 = row['bbox']
            results.append({
                'dx': dx[..., h1:h2, w1:w2].squeeze(0).cpu(),
                'dy': dy[..., h1:h2, w1:w2].squeeze(0).cpu(),
                'magnitude': magnitude[..., h1:h2, w1:w2].squeeze(0).cpu(),
                'orientation': orientation[..., h1:h2, w1:w2].squeeze(0).cpu(),
            })
        
        return pd.Series(results, index=group_df.index)