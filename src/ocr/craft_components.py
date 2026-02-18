"""
This files handles the different operations on the components extracted from CRAFT's output:
- combine_text_link_scores -> combine text and link score maps before watershed
- filter_craft_components  -> filter the components by aspect ratio and area
- merge_craft_components   -> merge the components that are too close to each other (composite characters usually)
"""

from .params import craftComponentsParams
from ..utils import connectedComponent

import cv2
import numpy as np
import networkx as nx

from scipy.spatial import distance_matrix


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
    """Filter CRAFT components by area and aspect ratio, marking deletions with reasons."""
    
    for region in components.regions:
        label = region.label
        
        # Get component stats
        bbox = region.bbox
        width = bbox[3] - bbox[1]
        height = bbox[2] - bbox[0]
        area = region.area
        aspect_ratio = width / height if height > 0 else 0
        
        # Check criteria and delete with specific reason
        if params.min_area is not None and area < params.min_area:
            components.delete(label, reason="area_too_small")
        elif params.min_aspect_ratio is not None and aspect_ratio < params.min_aspect_ratio:
            components.delete(label, reason="aspect_ratio_too_low")
        elif params.max_aspect_ratio is not None and aspect_ratio > params.max_aspect_ratio:
            components.delete(label, reason="aspect_ratio_too_high")
    
    return components

def merge_craft_components(params: craftComponentsParams, components: connectedComponent):
    """Merge CRAFT components that are too close to each other."""
    
    # Get all non-deleted labels
    unique_labels = np.unique(components.labels)
    active_labels = unique_labels[unique_labels != 0].tolist()
    
    if len(active_labels) <= 1:
        return components
    
    # Build a mapping from label to region
    label_to_region = {region.label: region for region in components.regions}
    
    # Get centroids for active regions
    centroids = np.array([label_to_region[label].centroid for label in active_labels])
    
    # Calculate distance matrix and linking matrix
    dist_matrix = distance_matrix(centroids, centroids)
    link_matrix = dist_matrix < params.min_dist
    
    # Merge the components
    components.merge(link_matrix, labels_list=active_labels)
    
    return components