
import torch
from dataclasses import dataclass
from torch import Tensor
from typing import Optional, List



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
