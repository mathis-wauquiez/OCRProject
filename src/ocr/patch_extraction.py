

from .params import craftComponentsParams
from ..utils import connectedComponent

import cv2
import numpy as np
import networkx as nx

from scipy.spatial import distance_matrix

def extract_patches(characterComponents: connectedComponent, border = None, image = None):
    """
    Extracts patches arround each character in characterComponents.
    """

    if border is not None:
        labels = np.pad(characterComponents.labels.copy(), border, mode='constant', constant_values=0)
        if image is not None:
            image = np.pad(image.copy(), ((border, border), (border, border), (0,0)), mode='median')
    else:
        labels = characterComponents.labels

    patches = []

    for region in characterComponents.regions:
        h1, w1, h2, w2 = region.bbox
        h2+=2*border; w2+=2*border

        patch = labels[h1:h2, w1:w2]
        patch = (patch == region.label).astype(np.float32)
        patches.append(patch)
    
    if image is not None:

        image_patches = []
        for region in characterComponents.regions:
            h1, w1, h2, w2 = region.bbox
            h2+=2*border; w2+=2*border
            patch = image[h1:h2, w1:w2]
            image_patches.append(patch)

        return patches, image_patches

    return patches


# def normalize_patches(patches, target_shape):
    