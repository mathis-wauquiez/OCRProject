
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
    padding_mode: str = 'reflect'


    #! HOG Parameters
    cell_height: int    = 16
    """ Height of the cells used to compute histograms """
    cell_width: int     = 16
    """ Width of the cells used to compute histograms """
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
    histograms: Tensor               # N, C, N_cells, N_bins tensor with the normalized histograms


