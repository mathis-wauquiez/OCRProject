
from dataclasses import dataclass
from typing import List, Optional

from torch import Tensor
import torch.nn as nn
import torch

import cv2
import numpy as np

from einops import rearrange
from ..utils import connectedComponent


def extract_patches(characterComponents: connectedComponent, images, return_bin = False, border = None):
    """
    Extracts patches arround each character in characterComponents.
    """

    patches_list = []
    
    if return_bin:
        images.insert(0, characterComponents.labels)

    for i, image in enumerate(images):
        if border is not None:
            if type(image) == np.ndarray:
                image = np.pad(image, border, mode='constant', constant_values=0)
            else:
                image = nn.functional.pad(image, (border, border), mode='constant', value=0)
        else:
            border = 0

        patches = []

        for region in characterComponents.regions:
            h1, w1, h2, w2 = region.bbox
            h2+=2*border; w2+=2*border

            patch = image[h1:h2, w1:w2] if type(image) == np.ndarray else image[..., h1:h2, w1:w2]
            
            if i == 0 and return_bin:
                patch = patch == region.label

            patches.append(patch)

        patches_list.append(patches)
    
    return patches_list