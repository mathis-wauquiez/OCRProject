
from dataclasses import dataclass
from typing import List, Optional

from torch import Tensor
import torch.nn as nn
import torch

import cv2
import numpy as np

from einops import rearrange
from ..utils import connectedComponent, GRADIENT_KERNELS, get_gradient_kernel
from .params import HOGParameters, fullHOGOutput


class HOG:

    def __init__(self, params: HOGParameters, custom_preprocess_fn = None):
        self._params = params
        self.custom_preprocess_fn = custom_preprocess_fn

        self._instantiate_kernels()

    def _instantiate_kernels(self):

        device = self._params.device
        C = self._params.C
        padding_mode = self._params.padding_mode

        if self._params.method == 'gaussian':
            ksize = int(self._params.grdt_sigma*self._params.ksize_factor) //2 * 2 + 1
            k = cv2.getGaussianKernel(ksize, self._params.grdt_sigma)
            d, _ = cv2.getDerivKernels(dx=1, dy=0, ksize=ksize, normalize=True)
        else:
            kernel = get_gradient_kernel(self._params.method)
            k = kernel['k']
            d = kernel['d']
            ksize = len(k)


        dx_conv = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=(1, ksize), padding=(0, ksize//2), bias=False, groups=C, padding_mode=padding_mode),
            nn.Conv2d(C, C, kernel_size=(ksize, 1), padding=(ksize//2, 0), bias=False, groups=C, padding_mode=padding_mode)
        ).to(device).eval()

        dy_conv = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=(ksize, 1), padding=(ksize//2, 0), bias=False, groups=C, padding_mode=padding_mode),
            nn.Conv2d(C, C, kernel_size=(1, ksize), padding=(0, ksize//2), bias=False, groups=C, padding_mode=padding_mode)
        ).to(device).eval()

        pre_gradient_conv = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=(ksize, 1), padding=(ksize//2, 0), bias=False, groups=C, padding_mode=padding_mode),
            nn.Conv2d(C, C, kernel_size=(1, ksize), padding=(0, ksize//2), bias=False, groups=C, padding_mode=padding_mode)
        ).to(device).eval()

        with torch.no_grad():
            for i in range(C):
                dx_conv[0].weight[i, 0, 0, :] = torch.tensor(k.flatten(), dtype=torch.float32, device=device)
                dx_conv[1].weight[i, 0, :, 0] = torch.tensor(d.flatten(), dtype=torch.float32, device=device)

                dy_conv[0].weight[i, 0, :, 0] = torch.tensor(k.flatten(), dtype=torch.float32, device=device)
                dy_conv[1].weight[i, 0, 0, :] = torch.tensor(d.flatten(), dtype=torch.float32, device=device)

                pre_gradient_conv[0].weight[i, 0, :, 0] = torch.tensor(k.flatten(), dtype=torch.float32, device=device)
                pre_gradient_conv[1].weight[i, 0, 0, :] = torch.tensor(k.flatten(), dtype=torch.float32, device=device)

        self.dx_conv = dx_conv
        self.dy_conv = dy_conv
        self.pre_gradient_conv = pre_gradient_conv

    def preprocess(self, input):
        preprocessed = self.pre_gradient_conv(input)
        return self.dx_conv(preprocessed), self.dy_conv(preprocessed)

    def normalize(self, patches):
         # === UNUSED AT THE MOMENT ===
        raise NotImplementedError
        f = lambda patch: nn.functional.interpolate(patch, (self._params.psize, self._params.psize), mode='bilinear')

        patches = [f(patch.unsqueeze(0)).squeeze(0) for patch in patches]
        return torch.stack(patches)

    def compute_hog_histograms_trilinear(self, magn_patches, ori_patches):
        """
        Compute HOG histograms using trilinear interpolation with built-in functions.
        
        Args:
            magn_patches: shape (..., H, W)
            ori_patches: shape (..., H, W), 0-1
            num_bins: Number of orientation bins
        
        Returns:
            Histograms of shape (..., num_patches, num_bins)
        """
        
        # magn_patches = self.normalize(magn_patches)
        # ori_patches = self.normalize(ori_patches)

        device = self._params.device
        dtype = magn_patches.dtype

        cell_height = self._params.cell_height
        cell_width = self._params.cell_width
        num_bins = self._params.num_bins
        sigma = self._params.sigma

        # Divide the patch into cells

        magn_patches = rearrange(magn_patches, '... (h ch) (w cw) -> ... (h w) ch cw', 
                        ch=cell_height, cw=cell_width)

        ori_patches = rearrange(ori_patches, '... (h ch) (w cw) -> ... (h w) ch cw', 
                            ch=cell_height, cw=cell_width)
        
        assert ori_patches.max() <= 1 and ori_patches.min() >= 0, "The orientation must be between 0 and 1"

            
        # Normalize orientations to [0, num_bins) range
        ori_normalized = ori_patches * num_bins
        

        if sigma is not None:        
            # Spatial distance weighting from cell center
            y = torch.linspace(-1, 1, cell_height, device=device, dtype=dtype)
            x = torch.linspace(-1, 1, cell_width, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            spatial_weight = torch.exp(-(yy**2 + xx**2) / 2 / sigma)
        
            # Apply spatial weights to magnitudes
            weighted_magnitudes = magn_patches * spatial_weight
        
        else:
            weighted_magnitudes = magn_patches
        
        # Flatten for bin assignment
        weighted_mag_flat = rearrange(weighted_magnitudes, '... h w -> ... (h w)')
        ori_flat = rearrange(ori_normalized, '... h w -> ... (h w)')
        
        # Orientation bin interpolation
        ori_floor = torch.floor(ori_flat).long() % num_bins
        ori_ceil = (ori_floor + 1) % num_bins
        ori_weight_ceil = ori_flat - torch.floor(ori_flat)
        ori_weight_floor = 1 - ori_weight_ceil
        
        # Split weighted magnitudes between adjacent bins
        contrib_floor = weighted_mag_flat * ori_weight_floor
        contrib_ceil = weighted_mag_flat * ori_weight_ceil
        
        # Use scatter_add to add to the histograms
        batch_shape = weighted_mag_flat.shape[:-1]
        histograms = torch.zeros(*batch_shape, num_bins, device=device, dtype=dtype)
        
        histograms.scatter_add_(dim=-1, index=ori_floor, src=contrib_floor)
        histograms.scatter_add_(dim=-1, index=ori_ceil, src=contrib_ceil)
        
        return histograms
    
    def normalize_histograms(self, histograms):
        if self._params.normalize is True or self._params.normalize == "patch":
            norm_dims = [-2, -1]
        elif self._params.normalize == 'cell':
            norm_dims = [-1]
        elif self._params.normalize is None or self._params.normalize is False:
            return histograms
        else:
            raise ValueError(f"Normalization must be a boolean, 'patch' or 'cell'. Received {self._params.normalize}.")
        
        histograms = histograms / torch.norm(histograms, dim=norm_dims, keepdim=True)
        histograms.clip_(-self._params.threshold, self._params.threshold)
        histograms /= torch.norm(histograms, dim=norm_dims, keepdim=True)
        return histograms
    

    def __call__(self, patches: Tensor):
        with torch.no_grad():
            dx, dy = self.preprocess(patches) # B, C, H, W
            magn = torch.sqrt(dx**2 + dy**2)
            # angle = torch.arctan2(dy, dx) / np.pi / 2 + .5

            # USE UNSIGNED GRADIENTS
            angle = (torch.atan2(dy, dx) % np.pi) / np.pi
            
            histograms = self.compute_hog_histograms_trilinear(magn, angle)
            if self._params.normalize:
                histograms = self.normalize_histograms(histograms)
            
            if self._params.partial_output:
                del dx, dy 
                return histograms
            
            return fullHOGOutput(
                dx=dx,
                dy=dy,
                patches_grdt_magnitude=magn,
                patches_grdt_orientation=angle,
                histograms=histograms
            )




# =========================
# ==== Caching utility ====
# =========================
import pickle
import hashlib
import json
from dataclasses import asdict
from PIL import Image

def get_params_hash(params):
    """Create hash from HOG parameters, excluding device."""
    params_dict = {k: v for k, v in asdict(params).items() if k != 'device'}
    return hashlib.md5(json.dumps(params_dict, sort_keys=True).encode()).hexdigest()

def setup_cache(cache_folder, params):
    """Setup cache folder and check if params changed."""
    cache_folder.mkdir(exist_ok=True)
    params_file = cache_folder / 'params.json'
    current_hash = get_params_hash(params)
    
    # Check and clear cache if params changed
    if params_file.exists():
        with open(params_file, 'r') as f:
            if json.load(f).get('hash') != current_hash:
                print("Parameters changed - clearing cache")
                for f in cache_folder.glob('*.pkl'):
                    f.unlink()
    
    # Save current params
    with open(params_file, 'w') as f:
        json.dump({'hash': current_hash}, f)

def load_or_compute_hog(file, cache_folder, hog, image_folder, comps_folder):
    """Load cached HOG output or compute if not available."""
    cache_file = cache_folder / f"{file}.pkl"
    
    # if cache_file.exists():
    #     with open(cache_file, 'rb') as f:
    #         return pickle.load(f)
    
    # Compute HOG
    img_np = np.array(Image.open(image_folder / file))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)[..., None]
    img_torch = torch.tensor(img_np, device="cuda").permute(2,0,1).float() / 255
    img_torch.requires_grad = False
    img_comp = connectedComponent.load(comps_folder / (str(file) + '.npz'))
    
    hog_output = hog(img_torch, img_comp)
    
    # Cache result
    # with open(cache_file, 'wb') as f:
    #     pickle.dump(hog_output, f)
    
    return hog_output

