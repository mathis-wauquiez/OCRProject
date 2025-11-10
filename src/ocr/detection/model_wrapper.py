import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from skimage.measure import _regionprops, regionprops

from dataclasses import dataclass
from dataclasses import replace

from .craft.craft import CRAFT
from .craft import craft_utils

from collections import OrderedDict
from typing import NamedTuple, Any, Tuple, List

from ...utils import connectedComponent
from ... import utils

from ..params import craftParams





@dataclass
class craftComponents:
    filtered_components: connectedComponent
    merged_components: connectedComponent


class craftWrapper(nn.Module):

    def __init__(self, params=craftParams(), **kwargs):
        super().__init__()

        params = replace(params, **kwargs)

        self.model = CRAFT()
        self.params = params
        self.model.load_state_dict(utils.copyStateDict(torch.load(params.chckpt)))

    def process_batch(self, x):
        """
        Args:
            x, (B, C, H, W) preprocessed tensor
        Returns:
            score_text, (B, C, H/2, W/2) tensor
            score_link, (B, C, H/2, W/2) tensor
        """
        y, feature = self.model(x)

        score_text = y[..., 0]
        score_link = y[..., 1]
        
        return score_text, score_link


    def preprocess_batch(self, x, square_size=None, interpolation=None, mag_ratio=None):
        """
        Args:
            x: (B, C, H, W) tensor with 0-255 range
        Returns:
            preprocessed tensor
        
        """
        B, C, H, W = x.shape

        mag_ratio = mag_ratio or self.params.mag_ratio
        square_size = square_size or self.params.canvas_size
        interpolation = interpolation or self.params.interpolation

        ratio = min(mag_ratio * max(H, W), square_size) / max(H, W)
        target_h, target_w = int(H * ratio), int(W * ratio)
        target_h_32, target_w_32 = utils.nearest_32(target_h), utils.nearest_32(target_w)

        processed = torch.zeros((B, C, target_h_32, target_w_32), device=x.device)
        processed[..., :target_h, :target_w] = F.interpolate(x, size=(target_h, target_w), mode=interpolation, align_corners=False)
        
        # this is very weird, but the authors normalize after the zero-padding
        processed = self.normalize(processed)

        return processed
    
    def normalize(self, x):
        x = x - self.params.mean.view(1, -1, 1, 1).to(x.device)
        x /= self.params.std.view(1, -1, 1, 1).to(x.device)
        return x

    def unnormalize(self, x):
        x = x * self.params.std.view(1, -1, 1, 1).to(x.device)
        x += self.params.mean.view(1, -1, 1, 1).to(x.device)
        return x

    def map_preprocessed_to_original(self, coords_preprocessed, original_shape, square_size=None, mag_ratio=None):
        H, W = original_shape
        
        mag_ratio = mag_ratio or self.params.mag_ratio
        square_size = square_size or self.params.canvas_size
        
        # Calculate the same ratio used in preprocessing
        ratio = min(mag_ratio * max(H, W), square_size) / max(H, W)
        
        # Reverse the scaling: divide by ratio
        coords_original = coords_preprocessed / ratio
        
        return coords_original
    
    def map_original_to_preprocessed(self, coords_original, original_shape, square_size=None, mag_ratio=None):
        H, W = original_shape
        
        mag_ratio = mag_ratio or self.params.mag_ratio
        square_size = square_size or self.params.canvas_size
        
        # Calculate the same ratio used in preprocessing
        ratio = min(mag_ratio * max(H, W), square_size) / max(H, W)
        
        # The image was scaled by 'ratio', so multiply by ratio
        coords_preprocessed = coords_original * ratio
        
        return coords_preprocessed
        
    def forward(self, x):
        preprocessed = self.preprocess_batch(x)
        score_text, score_link = self.process_batch(preprocessed)

        return preprocessed, score_text, score_link

        
    def filter_image_components(self, components: connectedComponent):
        # Get unique labels from the component
        unique_labels = np.unique(components.labels)
        
        # Create a set of labels to keep
        labels_to_keep = set()
        
        # Always keep the background component (label 0)
        labels_to_keep.add(0)
        
        # Filter by aspect ratio
        for i, region in enumerate(components.regions):
            # Get the actual label for this region
            # Assuming regions correspond to unique_labels[1:] (excluding background)
            if i + 1 < len(unique_labels):
                label = unique_labels[i + 1]
            else:
                continue
                
            aspect_ratio = region.axis_major_length / max(region.axis_minor_length, 1)
            if aspect_ratio > self.params.min_image_component_aspect_ratio and region.axis_major_length > self.params.min_image_component_axis_major_length_criterion:
                # Don't keep this component
                pass
            else:
                labels_to_keep.add(label)
            # if region.eccentricity < 0.9:
            #     labels_to_keep.add(label)
        
        # Apply filter: set filtered components to background
        labels = components.labels.copy()
        labels[~np.isin(labels, list(labels_to_keep))] = 0
        return connectedComponent.from_labels(labels, intensity_image=components.intensity_image)




