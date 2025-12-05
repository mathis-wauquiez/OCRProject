
import torch
from dataclasses import dataclass
from torch import Tensor
from typing import Optional, List


@dataclass
class GradientParameters:
    device: str = "cuda" if torch.cuda.is_available() else 'cpu'
    C: int = 1
    method: str = 'gaussian'
    grdt_sigma: float = 3.5
    ksize_factor: float = 8


@dataclass
class HOGParameters:
    device: str = "cuda" if torch.cuda.is_available() else 'cpu'
    C: int      = 1
    partial_output: bool = True

    #! Gradients computation
    method: str         = 'gaussian'
    """ Either gaussian, farid_3x3, farid_5x5, hypomode or central differences """
    grdt_sigma: float   = 3.5
    ksize_factor: float = 8


    #! HOG Parameters
    cell_height: int    = 16
    """ Height of the cells used to compute histograms """
    cell_width: int     = 16
    """ Width of the cells used to compute histograms """
    psize:      int     = 128
    """ The size of the patches """
    num_bins: int       = 8
    """ Number of bins in the histogram """
    sigma: float | None = None
    """ Parameter for the gaussian kernel """
    normalize: bool = False

    threshold: float | None = 0.2
    """ Threshold the values of the normalized histograms to 0.2 """

@dataclass
class fullHOGOutput:
    dx: Tensor          # C, H, W
    dy: Tensor          # C, H, W
    patches_grdt_magnitude: Tensor   # N, C, h, w
    patches_grdt_orientation: Tensor # N, C, h, w
    patches_image           : Tensor # N, C, h, w
    histograms: Tensor               # N, C, N_cells, N_bins tensor with the normalized histograms



@dataclass
class featureMatchingParameters:
    #! Feature matching
    metric: str = "CEMD"
    """ Metric used for the matching. Either CEMD, L1, L2 or EMD """
    epsilon: float = 0.005
    """ Expectancy of false positives on all the dataset under the normality hypothesis """
    reciprocal_only: bool = True
    """ Wether or not the matches should be reciprocal. When true, the distances have to be computed twice. """
    partial_output: bool = True
    """ Return everything or only the matches """
    distribution: str = 'normal'
    """ Distribution of the total distance. Either normal or gamma """
    two_pass: bool = False
    """ If true, two passes will be done - one using epsilon with L2, another using epsilon_2 using the specified metric"""
    epsilon_2: Optional[float] = None

@dataclass
class featureMatchingOutputs:
    match_indices: Tensor
    """ (N_matches, 2) Tensor matching queries (match_indices[:, 0]) to their keys (match_indices[:, 1])"""
    nlfa: Tensor
    """ The threshold values. If the distances are inferior to this, we reject the feature independance hypothesis and match the patches. """
    dissimilarities: Tensor
    """ (N1, N2) aggregated dissimilarities for forward matching """
    nlfa_threshold: float

    nlfa2: Optional[Tensor] = None
    """ The threshold values for backward matching (when reciprocal_only=True) """
