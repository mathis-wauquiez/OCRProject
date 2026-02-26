

import json

import torch

from dataclasses import dataclass
from dataclasses import replace
from PIL import Image
from ..utils import connectedComponent
from pathlib import Path


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
    """Threshold used on the CRAFT output for seed detection (peak_local_max)"""
    connectivity: int = 8
    """Connectivity for the connected components"""

    mask_threshold: float | None = None
    """Lower threshold for watershed basin expansion mask. When set, watershed
    seeds are detected at text_threshold (high) but basins expand into areas
    with score > mask_threshold (lower), capturing radicals of composite
    characters (e.g. 蒸) that have moderate CRAFT scores. Set to None to use
    text_threshold for both (original behavior). Recommended: 0.2-0.4."""

    link_threshold: float | None = None
    """Threshold for CRAFT link score combination. When set, text and link
    scores are combined before watershed: combined = clip(score_text +
    (score_link > link_threshold), 0, 1). This bridges gaps between
    sub-components of composite characters. Set to None to disable.
    Recommended: 0.3-0.7."""

    # Watershed seed spacing (peak_local_max min_distance)
    min_dist: float = 8.
    """Minimum distance between watershed seeds (pixels)."""

    # Components filtering
    min_area: int | None = 10

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

    min_area: float = 700

    cc_filtering:          bool  = True
    cc_distance_threshold: float = 50
    cc_min_comp_size:      float = 10000


@dataclass
class PipelineOutput:

    img_pil: Image
    "PIL input image"
    
    preprocessed: Any
    "Image preprocessed to be fed to CRAFT"

    binary_img: np.ndarray
    "Binarized image, background = 0, foreground = 1"

    score_text: torch.Tensor
    "Score map output for texts from the CRAFT model"

    craft_components: connectedComponent
    "CRAFT text components after filtering and merging (check deletion/merge history for intermediate stages)"

    image_components: connectedComponent
    "The connected components of the binarized image"

    filtered_image_components: connectedComponent
    "The connected components of the binarized image after filtering big lines"

    characters: connectedComponent
    "Final character components after all filtering (proximity, size, area)"

    score_link: torch.Tensor = None
    "Score map output for link/affinity regions from the CRAFT model"

    similarity_matrix: np.ndarray = None
    "Similarity matrix between filtered image components and CRAFT components"
    
    characters_before_contour_filter: connectedComponent = None
    "Character components before contour proximity filtering (for visualization)"

    
    def save(self, save_dir: Path | str, save_intermediates: bool = False):
        """Save pipeline results"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save essentials
        self.img_pil.save(save_dir / 'input.png')
        Image.fromarray((self.binary_img * 255).astype(np.uint8)).save(save_dir / 'binary.png')
        
        # Save final characters
        self.characters.save(save_dir / 'characters.npz')
        Image.fromarray(self.characters.segm_img).save(save_dir / 'characters_viz.png')
        
        # Save deletion visualization
        deletion_viz, colors = self.characters.deletion_viz()
        Image.fromarray(deletion_viz).save(save_dir / 'deletion_viz.png')
        with open(save_dir / 'deletion_legend.json', 'w') as f:
            json.dump({k: list(v) for k, v in colors.items()}, f, indent=2)
        
        if save_intermediates:
            torch.save(self.score_text, save_dir / 'score_text.pt')
            self.craft_components.save(save_dir / 'craft_components.npz')
            self.image_components.save(save_dir / 'image_components.npz')
            self.filtered_image_components.save(save_dir / 'filtered_image_components.npz')
            
            # Save CRAFT deletion viz
            craft_viz, craft_colors = self.craft_components.deletion_viz()
            Image.fromarray(craft_viz).save(save_dir / 'craft_deletion_viz.png')
            with open(save_dir / 'craft_deletion_legend.json', 'w') as f:
                json.dump({k: list(v) for k, v in craft_colors.items()}, f, indent=2)
    
    @classmethod
    def load(cls, save_dir: Path | str):
        """Load saved characters from pipeline results"""
        save_dir = Path(save_dir)
        return connectedComponent.load(save_dir / 'characters.npz')
