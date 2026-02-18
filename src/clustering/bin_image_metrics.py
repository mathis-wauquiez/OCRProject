from ..registration.single_scale import InverseCompositional
from ..registration.gradients import Gradients
from ..registration.gaussian_pyramid import GaussianPyramid
from ..registration.multiscale_registration import MultiscaleIC


import torch

import numpy as np
import tqdm


# At the end of this file there is:

# metrics_dict = {
#     'dice': dice_coefficient,
#     'jaccard': jaccard_index,
#     'hamming': compute_hamming,
#     'hausdorff': compute_hausdorff,
# }

# # Initialize the registered metric
# reg_metric = registeredMetric(metrics=metrics_dict, sym=True)

def compute_hamming(img1_bin, img2_bin):
    result = (img1_bin != img2_bin).sum() / img1_bin.numel()
    return result.item() if torch.is_tensor(result) else result

def dice_coefficient(img1, img2):
    """Dice coefficient (F1 score)"""
    intersection = (img1 & img2).sum()
    result = 2.0 * intersection / (img1.sum() + img2.sum() + 1e-8)
    return result.item() if torch.is_tensor(result) else result


def jaccard_index(img1, img2):
    """Jaccard Index (IoU)"""
    intersection = (img1 & img2).sum()
    union = (img1 | img2).sum()
    result = intersection / (union + 1e-8)
    return result.item() if torch.is_tensor(result) else result


# def compute_hausdorff(img1_bin, img2_bin):
#     """
#     Proper Hausdorff distance using KD-tree.

#     ---- EDWIN'S CODE ----
#     """
#     # Get coordinates
#     coords1 = torch.nonzero(img1_bin, as_tuple=False).float()
#     coords2 = torch.nonzero(img2_bin, as_tuple=False).float()
    
#     if len(coords1) == 0 or len(coords2) == 0:
#         return float('inf')
    
#     # Convert to numpy for KD-tree (scipy doesn't work with torch)
#     coords1_np = coords1.cpu().numpy()
#     coords2_np = coords2.cpu().numpy()
    
#     from scipy.spatial import cKDTree
    
#     tree1 = cKDTree(coords1_np)
#     tree2 = cKDTree(coords2_np)
    
#     # Query nearest neighbors
#     dist1, _ = tree2.query(coords1_np, workers=-1)
#     dist2, _ = tree1.query(coords2_np, workers=-1)
    
#     return float(max(dist1.max(), dist2.max()))

def compute_hausdorff(img1_bin, img2_bin):
    """
    Hausdorff distance using distorch (Rony & Kervadec, MIDL 2025).
    
    Supports both single masks (H, W) and batched masks (B, H, W).
    Returns float for single pair, (B,) tensor for batched.
    """
    import distorch

    m1 = img1_bin.bool()
    m2 = img2_bin.bool()

    # Handle empty masks
    if m1.dim() == 2:
        if not m1.any() or not m2.any():
            return float('inf')
    
    metrics = distorch.boundary_metrics(m1, m2)
    
    if m1.dim() == 2:
        return float(metrics.Hausdorff)
    else:
        return metrics.Hausdorff  # (B,) tensor


class registeredMetric:

    def __init__(self, metrics: dict, sym=True, device='cuda', lazy_metric=None, lazy_threshold=None):
        """
        Class to compute the metrics between two images.
        Args:
            metrics: dict{metric_name: metric_function}, with metric_function taking torch (H, W) Tensors
            sym: wether to symmetrize the distance by default or not
        """

        gradient_method = Gradients(method='farid5', C=1, device=device)
        single_scale_ic = InverseCompositional(
            transform_type='homography',
            gradient_method=gradient_method,
            error_function='lorentzian',
            delta=5,
            epsilon=1e-3,
            max_iter=5,
        )

        gaussian_pyramid = GaussianPyramid(
            eta=0.5,                        # unzooming factor
            sigma_0=0.6,                    # initial std of the gaussian kernel
            ksize_factor=8,                 # kernel size = 2 * sigma * ksize_factor | 1
            min_size=32                     # size of the coarsest image in the pyramid
        )

        # Create the multiscale registration
        self.ic = MultiscaleIC(
            singleScaleIC=single_scale_ic,
            gaussianPyramid=gaussian_pyramid
        )

        self.device = device

        self.metrics = metrics
        self.sym = sym

        self.lazy_metric=lazy_metric
        self.lazy_threshold = lazy_threshold

    def __call__(self, I1, I2, sym=True):
        """
        Compute the metrics for this class.
        Args:
            I1: torch.Tensor
            I2: torch.Tensor
            sym: wether to symmetrize the distance or not
        Returns:
            dict[metric_name: metric_value]
        """

        do_warp = True

        if self.lazy_metric is not None:
            if self.lazy_metric(I1>.5, I2>.5) > self.lazy_threshold:
                do_warp = False


        if do_warp:
            I1 = I1.to(self.device)
            I2 = I2.to(self.device)
            I1 = I1.unsqueeze(0); I2 = I2.unsqueeze(0)

            try:
                T = self.ic.run(I1, I2)
            except:
                return {metric_name: 999 for metric_name in self.metrics.keys()}
            I2 = T.warp(I2.unsqueeze(0))[0, 0]

            mask = T.visibility_mask(I1.shape[1], I1.shape[2], delta=0).squeeze()
            I2[~mask] = 0

        I2_bin = I2 > 0.5
        I1_bin = I1.squeeze() > 0.5

        distances = {
            metric_name: metric(I1_bin, I2_bin)
            for metric_name, metric in self.metrics.items()
        }

        if sym and self.sym:
            distances_sym = self(I2.squeeze(), I1.squeeze(), sym=False)
            distances = {
                metric_name: (distances[metric_name] + distances_sym[metric_name]) / 2
                for metric_name in distances.keys()
            }

        return distances


metrics_dict = {
    'dice': dice_coefficient,
    'jaccard': jaccard_index,
    'hamming': compute_hamming,
    'hausdorff': compute_hausdorff,
}

# Initialize the registered metric
reg_metric = registeredMetric(metrics=metrics_dict, sym=True)





""" BELOW THIS LINE IS AI SLOP """

#! ===============================

def compute_distance_matrices_batched(
    reg_metric, renderer, subdf,
    batch_size=64,   # pairs per batch — tune to your VRAM
    device='cuda',
    use_tqdm=False,
    sym_registration=None,
):
    """
    Batched pairwise distance matrix computation.

    Instead of N² sequential IC registrations, we batch pairs of images
    through the multiscale IC and compute all metrics in parallel.

    Always computes only the upper triangle (i < j) and mirrors results,
    since distance matrices are symmetric.

    Args:
        sym_registration: Whether to run reverse IC registration and average
            forward + reverse metrics.  Defaults to ``reg_metric.sym``.
            Set to ``False`` to halve GPU time by using forward-only
            registration (Hausdorff is naturally symmetric on point sets;
            the small asymmetry from warp direction is negligible for
            same-cluster glyphs).
    """
    if sym_registration is None:
        sym_registration = reg_metric.sym

    indices = subdf.index.tolist()
    N = len(indices)
    metric_names = list(reg_metric.metrics.keys())

    distance_matrices = {key: np.zeros((N, N)) for key in metric_names}

    # ── Build upper-triangle pairs (i < j) ─────────────────────────────
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    n_pairs = len(pairs)

    # ── Process in batches ──────────────────────────────────────────────
    it = range(0, n_pairs, batch_size)
    if use_tqdm:
        it = tqdm.tqdm(it)
    for start in it:
        batch_pairs = pairs[start : start + batch_size]
        B = len(batch_pairs)

        # Stack image pairs → (B, 1, H, W)
        I1_batch = torch.stack([renderer[indices[i]].to(device) for i, j in batch_pairs]).unsqueeze(1)
        I2_batch = torch.stack([renderer[indices[j]].to(device) for i, j in batch_pairs]).unsqueeze(1)

        # ── Batched registration ────────────────────────────────────
        try:
            T = reg_metric.ic.run(I1_batch, I2_batch)  # (B,) transforms
        except Exception:
            # Fill with sentinel on failure
            for i_loc, j_loc in batch_pairs:
                for key in metric_names:
                    distance_matrices[key][i_loc, j_loc] = 999
                    distance_matrices[key][j_loc, i_loc] = 999
            continue

        # Warp I2 → (B, 1, H, W)
        I2_warped = T.warp(I2_batch)[:, 0]          # (B, H, W)

        # Visibility mask → (B, H, W)
        H, W = I1_batch.shape[2], I1_batch.shape[3]
        mask = T.visibility_mask(H, W, delta=0)      # (B, H, W)
        I2_warped[~mask] = 0

        # Binarise
        I1_bin = (I1_batch[:, 0] > 0.5)              # (B, H, W)
        I2_bin = (I2_warped > 0.5)                    # (B, H, W)

        # ── Batched metrics ─────────────────────────────────────────
        batch_metrics = _compute_metrics_batched(I1_bin, I2_bin, metric_names)

        # ── Symmetrise: also register I2→I1 ─────────────────────────
        if sym_registration:
            try:
                T_rev = reg_metric.ic.run(I2_batch, I1_batch)
            except Exception:
                T_rev = None

            if T_rev is not None:
                I1_warped = T_rev.warp(I1_batch)[:, 0]
                mask_rev = T_rev.visibility_mask(H, W, delta=0)
                I1_warped[~mask_rev] = 0
                I2_bin_rev = (I2_batch[:, 0] > 0.5)
                I1_bin_rev = (I1_warped > 0.5)
                batch_metrics_rev = _compute_metrics_batched(I2_bin_rev, I1_bin_rev, metric_names)

                # Average forward + reverse
                for key in metric_names:
                    batch_metrics[key] = (batch_metrics[key] + batch_metrics_rev[key]) / 2

        # ── Write to distance matrices (always mirror) ──────────────
        for k, (i_loc, j_loc) in enumerate(batch_pairs):
            for key in metric_names:
                val = batch_metrics[key][k].item() if torch.is_tensor(batch_metrics[key]) else batch_metrics[key][k]
                distance_matrices[key][i_loc, j_loc] = val
                distance_matrices[key][j_loc, i_loc] = val

    # ── Diagonal ────────────────────────────────────────────────────────
    for key in metric_names:
        if key in ('dice', 'jaccard'):
            np.fill_diagonal(distance_matrices[key], 1.0)
        # hamming=0, hausdorff=0 on diagonal (already zero from np.zeros)

    return distance_matrices


def _compute_metrics_batched(I1_bin, I2_bin, metric_names):
    """
    Compute all metrics for B pairs at once.
    
    I1_bin, I2_bin: (B, H, W) bool tensors.
    Returns: dict[str → (B,) tensor or array]
    """
    B = I1_bin.shape[0]
    results = {}
    
    if 'dice' in metric_names:
        inter = (I1_bin & I2_bin).flatten(1).sum(1).float()
        denom = I1_bin.flatten(1).sum(1).float() + I2_bin.flatten(1).sum(1).float() + 1e-8
        results['dice'] = 2.0 * inter / denom
    
    if 'jaccard' in metric_names:
        inter = (I1_bin & I2_bin).flatten(1).sum(1).float()
        union = (I1_bin | I2_bin).flatten(1).sum(1).float() + 1e-8
        results['jaccard'] = inter / union
    
    if 'hamming' in metric_names:
        n_pixels = I1_bin.shape[1] * I1_bin.shape[2]
        results['hamming'] = (I1_bin != I2_bin).flatten(1).sum(1).float() / n_pixels
    
    if 'hausdorff' in metric_names:
        import distorch
        hd = distorch.boundary_metrics(I1_bin, I2_bin).Hausdorff  # (B,)
        results['hausdorff'] = hd
    
    return results

