from .hog import HOG
from .params import HOGParameters

from .renderer import Renderer

def get_hog_cfg(cell_size, grdt_sigma=5, num_bins=8, normalize='patch', device='cuda'):
    renderer = Renderer(
        scale           = 1.0,
        dpi             = 256,
        bin_thresh      = 128,
        pad_to_multiple = cell_size,
        verbose         = True
    )

    hog_params = HOGParameters(
        device = device,
        C = 1,                           # Use grayscale images
        partial_output = False,          # Also output the resized images, their gradient orientation and magnitude
        method = "gaussian",             # Use gaussian smoothing to compute the gradients
        grdt_sigma = grdt_sigma,         # Std of the smoothing
        ksize_factor = 6,                # Size of the smoothing kernel = factor * std
        cell_height = cell_size,         # Size of the cells to compute the histograms
        cell_width= cell_size,
        num_bins = num_bins,             # Number of bins
        threshold = 0.2,                 # Clip the values of the descriptor
        normalize = normalize            # Normalize at patch-level. SIFT uses cell-level descriptor normalization
    )

    return renderer, hog_params