from ..registration.single_scale import InverseCompositional
from ..registration.gradients import Gradients
from ..registration.gaussian_pyramid import GaussianPyramid
from ..registration.multiscale_registration import MultiscaleIC


import torch



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


def compute_hausdorff(img1_bin, img2_bin):
    """
    Proper Hausdorff distance using KD-tree.

    ---- EDWIN'S CODE ----
    """
    # Get coordinates
    coords1 = torch.nonzero(img1_bin, as_tuple=False).float()
    coords2 = torch.nonzero(img2_bin, as_tuple=False).float()
    
    if len(coords1) == 0 or len(coords2) == 0:
        return float('inf')
    
    # Convert to numpy for KD-tree (scipy doesn't work with torch)
    coords1_np = coords1.cpu().numpy()
    coords2_np = coords2.cpu().numpy()
    
    from scipy.spatial import cKDTree
    
    tree1 = cKDTree(coords1_np)
    tree2 = cKDTree(coords2_np)
    
    # Query nearest neighbors
    dist1, _ = tree2.query(coords1_np, workers=-1)
    dist2, _ = tree1.query(coords2_np, workers=-1)
    
    return float(max(dist1.max(), dist2.max()))



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

            mask = T.visibility_mask(I1.shape[1], I1.shape[2], delta=0)
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
