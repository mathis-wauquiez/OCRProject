"""
This files handles the different operations on the components extracted from CRAFT's output:
- combine_text_link_scores -> combine text and link score maps before watershed
- filter_craft_components  -> filter the components by area
"""

from .params import craftComponentsParams
from ..utils import connectedComponent

import numpy as np


def combine_text_link_scores(score_text: np.ndarray,
                             score_link: np.ndarray,
                             link_threshold: float) -> np.ndarray:
    """Combine CRAFT text and link score maps before watershed segmentation.

    The link score bridges gaps between sub-components of composite characters
    (e.g., the body and water radical of 蒸). By adding the thresholded link
    score to the text score, watershed segmentation produces unified blobs for
    composite characters instead of separate fragments.

    This follows the canonical CRAFT approach from craft_utils.py:29:
        text_score_comb = np.clip(text_score + link_score, 0, 1)

    Parameters
    ----------
    score_text : np.ndarray, shape (H, W)
        CRAFT text confidence map (values in [0, 1]).
    score_link : np.ndarray, shape (H, W)
        CRAFT link/affinity confidence map (values in [0, 1]).
    link_threshold : float
        Threshold for binarizing the link score. Pixels with
        score_link > link_threshold contribute to the combined score.

    Returns
    -------
    combined : np.ndarray, shape (H, W)
        Combined score map, clipped to [0, 1].
    """
    link_binary = (score_link > link_threshold).astype(score_text.dtype)
    return np.clip(score_text + link_binary, 0, 1)


def filter_craft_components(params: craftComponentsParams, components: connectedComponent):
    """Filter CRAFT components by area, marking deletions with reasons."""

    for region in components.regions:
        label = region.label
        area = region.area

        if params.min_area is not None and area < params.min_area:
            components.delete(label, reason="area_too_small")

    return components
