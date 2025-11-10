

from torch import Tensor
import torch.nn as nn
import torch

import cv2
import numpy as np

from einops import rearrange
from ..utils import connectedComponent, extract_patches
from .params import HOGParameters, fullHOGOutput, featureMatchingOutputs, featureMatchingParameters

kernels = {
    'central_differences':{
        'k': np.array([0,1,0]),
        'd': np.array([-.5,0,.5])
    },
    'hypomode':{
        'k': np.array([0,0.5,0.5]),
        'd': np.array([0,-1,1])
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


class HOG:

    def __init__(self, params: HOGParameters):
        self._params = params

        self._instantiate_kernels()

    def _instantiate_kernels(self):

        device = self._params.device
        C = self._params.C

        if self._params.method == 'gaussian':
            ksize = int(self._params.grdt_sigma*self._params.ksize_factor) //2 * 2 + 1
            k = cv2.getGaussianKernel(ksize, self._params.grdt_sigma)
            d, _ = cv2.getDerivKernels(dx=1, dy=0, ksize=ksize, normalize=True)
        elif self._params.method in kernels:
            k = kernels[self._params.method]['k']
            d = kernels[self._params.method]['d']
            ksize = len(k)
        else:
            raise ValueError(f'Gradient computation method not recognized: {self._params.method}')


        dx_conv = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=(1, ksize), padding=(0, ksize//2), bias=False, groups=C, padding_mode='reflect'),
            nn.Conv2d(C, C, kernel_size=(ksize, 1), padding=(ksize//2, 0), bias=False, groups=C, padding_mode='reflect')
        ).to(device).eval()

        dy_conv = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=(ksize, 1), padding=(ksize//2, 0), bias=False, groups=C, padding_mode='reflect'),
            nn.Conv2d(C, C, kernel_size=(1, ksize), padding=(0, ksize//2), bias=False, groups=C, padding_mode='reflect')
        ).to(device).eval()

        pre_gradient_conv = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=(ksize, 1), padding=(ksize//2, 0), bias=False, groups=C, padding_mode='reflect'),
            nn.Conv2d(C, C, kernel_size=(1, ksize), padding=(0, ksize//2), bias=False, groups=C, padding_mode='reflect')
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
        
        # Spatial distance weighting from cell center
        y = torch.linspace(-1, 1, cell_height, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, cell_width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        spatial_weight = torch.exp(-(yy**2 + xx**2) / 2 / sigma)
        
        # Apply spatial weights to magnitudes
        weighted_magnitudes = magn_patches * spatial_weight
        
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
        histograms = histograms / torch.norm(histograms)
        histograms.clip_(-0.2, 0.2)
        histograms /= torch.norm(histograms)
        return histograms
    

    def __call__(self, image: Tensor, comps: connectedComponent):
        with torch.no_grad():
            dx, dy = self.preprocess(image)
            magn = torch.sqrt(dx**2 + dy**2)
            # angle = torch.arctan2(dy, dx) / np.pi / 2 + .5

            # USE UNSIGNED GRADIENTS
            angle = (torch.atan2(dy, dx) % np.pi) / np.pi
            
            magn_patches, ori_patches, img_patches = extract_patches(
                comps, (magn, angle, image)
            )
            magn_patches = self.normalize(magn_patches)
            ori_patches = self.normalize(ori_patches)
            img_patches = self.normalize(img_patches)
            
            histograms = self.compute_hog_histograms_trilinear(magn_patches, ori_patches)
            histograms = self.normalize_histograms(histograms)
            
            if self._params.partial_output:
                del dx, dy 
                return histograms, img_patches
            
            return fullHOGOutput(
                dx=dx,
                dy=dy,
                patches_grdt_magnitude=magn_patches,
                patches_grdt_orientation=ori_patches,
                patches_image=img_patches,
                histograms=histograms
            )


class featureMatching:

    def __init__(self, params: featureMatchingParameters):

        self._params = params

    def match(self, query_histograms, key_histograms):

        from torch.distributions import Normal
        import psutil

        device = query_histograms.device
        epsilon = self._params.epsilon  # Expected number of false detections over the dataset

        N1, Nh, Nbins = query_histograms.shape
        N2, Nh, Nbins = key_histograms.shape

        standard_normal = Normal(0, 1)

        if not self._params.partial_output:
            deltas = torch.zeros((N1,), device=device)
            total_dissimilarities = torch.zeros((N1, N2))

        available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0) if device == 'cuda' else psutil.virtual_memory().available
        available_memory -= 100 * 2**20 # remove 100 MB to be sure everything fits in memory
        element_size = query_histograms.element_size()

        slice_memory = N2 * Nh * Nbins * element_size * 30

        # Compute the batch size as the maximum amount of (N2, Nh, Nbins) slices we can fit in memory
        batch_size = max(1, available_memory // slice_memory)

        print('Batch size : ', batch_size)

        matches_list = []

        for idx_start in range(0, N1, batch_size):
            queries = query_histograms[idx_start:idx_start+batch_size]

            # Compute the N_cells dissimilarities for the batch of queries
            dissimilarities = self.compute_dissimilarities(queries, key_histograms) # N1, N2, N_cells

            # Compute the sum of their mean/std
            mu_tot   = dissimilarities.mean(dim=1).sum(-1)  # (N1,)
            var_tot  = dissimilarities.var(dim=1).sum(-1)   # (N1,)
            
            # Compute the threshold and store it
            delta = mu_tot + var_tot**.5 * standard_normal.icdf(torch.tensor(epsilon / (N1 * N2)))

            # Compute total dissimilarities D(a^i, b^j) for all pairs
            D = dissimilarities.sum(dim=-1)  # (N1, N2)

            if not self._params.partial_output:
                deltas[idx_start:idx_start+batch_size] = delta
                total_dissimilarities[idx_start:idx_start+batch_size] = D

            # Find eps-meaningful matches: D(a^i, b^j) <= delta_i(eps)
            matches = D <= delta.unsqueeze(1)  # (N1, N2) boolean mask

            # Get match indices (query_idx, candidate_idx)
            match_indices = torch.nonzero(matches, as_tuple=False)  # (num_matches, 2)
            match_indices[:, 0] += idx_start
            matches_list.append(match_indices)

        matches = torch.cat(matches_list, dim=0)

        if self._params.partial_output:
            return matches
        
        return matches, deltas, total_dissimilarities
    
    def keep_reciprocal(self, matches1, matches2):
        matches2 = matches2[:, [1, 0]]
        
        # Use hashing approach: combine indices into unique keys
        # key = query_idx * max_key_idx + key_idx
        max_idx = max(matches1[:, 1].max(), matches2[:, 1].max()) + 1
        
        keys1 = matches1[:, 0] * max_idx + matches1[:, 1]
        keys2 = matches2[:, 0] * max_idx + matches2[:, 1]
        
        # Find which keys from matches1 are also in matches2
        mask = torch.isin(keys1, keys2)
        
        # Get reciprocal matches
        reciprocal_matches = matches1[mask]
        
        return reciprocal_matches

    def __call__(self, query_histograms, key_histograms):
        # Forward matching
        match1_output = self.match(query_histograms, key_histograms)
        
        # Handle partial output mode
        if self._params.partial_output:
            match1 = match1_output
            
            if self._params.reciprocal_only:
                match2 = self.match(key_histograms, query_histograms)
                return self.keep_reciprocal(match1, match2)
            
            return match1
        
        # Full output mode
        match1, deltas1, dissim1 = match1_output
        
        if self._params.reciprocal_only:
            match2, deltas2, dissim2 = self.match(key_histograms, query_histograms)
            matches = self.keep_reciprocal(match1, match2)
            
            return featureMatchingOutputs(
                match_indices=matches,
                deltas=deltas1,
                total_dissimilarities=dissim1,
                deltas2=deltas2,
                total_dissimilarities2=dissim2
            )
        
        return featureMatchingOutputs(
            match_indices=match1,
            deltas=deltas1,
            total_dissimilarities=dissim1
        )
    
    def compute_dissimilarities(self, queries, keys):
        if self._params.metric == "L2":
            return torch.pow(queries[:, None] - keys[None, :], 2).sum(-1)
        elif self._params.metric == 'CEMD':
            X = queries.cumsum(-1)[:, None] - keys.cumsum(-1)[None, :] # (n_q, n_k, Nh, Nbins)

            # Compute ||X_k||_1 for each k
            X_padded = torch.cat([torch.zeros_like(X[..., :1]), X], dim=-1)  # ..., N+1

            
            l1_norms = []
            for starting_bin in range(X.shape[-1]):
                X_k = X - X_padded[..., starting_bin:starting_bin+1]  # Subtract X[k-1] from all positions
                l1_norm = X_k.abs().sum(dim=-1) / X.shape[-1]
                l1_norms.append(l1_norm)    

            cemd = torch.stack(l1_norms, dim=-1).min(dim=-1).values
            return cemd







