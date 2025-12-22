

import torch

from dataclasses import dataclass
from dataclasses import replace


from collections import OrderedDict
from typing import NamedTuple, Any, Tuple, List
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Updateable(object):
    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class craftParams(Updateable):
    mag_ratio: float = 5
    canvas_size: int = 1280
    interpolation: str = 'bilinear'

    mean = torch.tensor((0.485, 0.456, 0.406))*255
    std  = torch.tensor((0.229, 0.224, 0.225))*255

    chckpt: str = 'models/craft/craft_mlt_25k.pth'
    """CRAFT model path"""

@dataclass
class craftComponentsParams(Updateable):
    # Connected components parameters

    text_threshold: float = 0.6
    """Threshold used on the CRAFT output"""
    connectivity: int = 8
    """Connectivity for the connected components"""

    # Components filtering
    min_area: int | None = 10
    min_aspect_ratio: float | None = .5
    max_aspect_ratio: float | None = 2

    # Components merging
    min_dist: float | None = 8.
    """"Blobs whose centroids are < 8 pixels away will be merged"""

    # Unary potential parameters
    # UNUSED AT THE MOMENT
    characteristic_distance: float = 5. # characteristic n° of px for the
                                        # influence region of components 
    neighborhood_radius: int = 50       # outside, potential is null

    method: str = "watershed" # | 'cc'


@dataclass
class imageComponentsParams(Updateable):

    threshold: float | str = 'otsu'
    """Method for binarizing the image"""


    # Image components filtering
    # -> Delete long, elongated components that ressembles lines
    # -> Delete image components far away from characters

    min_image_component_aspect_ratio: float = 20
    """Definition of elongated"""
    min_image_component_axis_major_length_criterion: float = 130
    """Definition of long"""

    similarity_threshold: float = -15. # pixels further than 30 pixels (in Mahalanobis distance) away will be considered background
    """Definition of far from a character"""

    similarity_metric: str = 'mahalanobis'
    """Similarity metric. Can be euclidian or mahalanobis at the moment."""

    # Characters filtering

    # w, h
    min_box_size: Tuple[int] = (30, 30)
    max_box_size: Tuple[int] = (250, 250)

    max_aspect_ratio: float = 10

    max_filled_area_portion: float = 0.9

    min_area: float = 700

    cc_filtering:          bool  = True
    cc_distance_threshold: float = 50
    cc_min_comp_size:      float = 10000

from PIL import Image
from ..utils import connectedComponent

@dataclass
class PipelineOutput:

    img_pil:    Image
    "PIL input image"
    
    preprocessed:   Any
    "Image preprocessed to be fed to CRAFT"

    binary_img: np.ndarray
    "Binarized image, background = 0, foreground = 1"

    score_text: torch.Tensor
    "Score map output for texts from the CRAFT model"

    score_text_components: connectedComponent
    "Connected components of the text score map"

    filtered_text_components: connectedComponent
    "Text score map components after filtering small and elongated components"

    merged_text_components: connectedComponent
    "Filtered components after merging the components that are too close"

    image_components: connectedComponent
    "The connected components of the binarized image"

    filtered_image_components: connectedComponent
    "The connected components of the binarized image after filtering big lines"

    character_components: connectedComponent
    "The characters' components"

    cc_filtered: connectedComponent
    "The components filtered not to touch the borders etc"

    filteredCharacters: connectedComponent
    "Characters after filtering"