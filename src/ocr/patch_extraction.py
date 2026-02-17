
from ..utils import connectedComponent
from ..patch_processing.patch_extraction import extract_patches as _extract_patches_general

import numpy as np


def extract_patches(characterComponents: connectedComponent, border=None, image=None):
    """
    Extracts patches around each character in characterComponents.

    Thin wrapper around patch_processing.patch_extraction.extract_patches
    for backward compatibility with the simpler OCR-specific API.
    """
    images = []
    if image is not None:
        if border is not None:
            image = np.pad(image.copy(), ((border, border), (border, border), (0, 0)), mode='median')
        images.append(image)

    result = _extract_patches_general(characterComponents, images=images, return_bin=True, border=border)

    bin_patches = result[0]

    # Cast bin patches to float32 to match original behavior
    bin_patches = [p.astype(np.float32) for p in bin_patches]

    if image is not None:
        image_patches = result[1]
        return bin_patches, image_patches

    return bin_patches
