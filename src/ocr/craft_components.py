"""
This files handles the different operations on the components extracted from CRAFT's output.
It is only two functions:
- filter_craft_components -> filter the components by aspect ratio and area
- merge_craft_components  -> merge the components that are too close to each other (composite characters usually)


"""

from .params import craftComponentsParams
from ..utils import connectedComponent

import cv2
import numpy as np
import networkx as nx

from scipy.spatial import distance_matrix

def filter_craft_components(params: craftComponentsParams, components: connectedComponent):
    filter_mask = np.ones(components.nLabels, dtype=bool)
    
    # Filter by minimum area
    areas = components.stats[:, cv2.CC_STAT_AREA]
    filter_mask &= areas >= params.min_area

    # Filter by aspect ratio
    widths = components.stats[:, cv2.CC_STAT_WIDTH]
    heights = components.stats[:, cv2.CC_STAT_HEIGHT]
    aspect_ratios = widths / heights
    filter_mask &= (aspect_ratios >= params.min_aspect_ratio)
    filter_mask &= (aspect_ratios <= params.max_aspect_ratio)

    # Always keep the background component (index 0)
    filter_mask[0] = True

    # Apply filter: set filtered components to background
    labels = components.labels.copy()
    labels[~np.isin(labels, np.where(filter_mask)[0])] = 0
    return connectedComponent.from_labels(labels, intensity_image=components.intensity_image)

def merge_craft_components(params: craftComponentsParams, components: connectedComponent):
    # Get all unique labels except background
    unique_labels = np.unique(components.labels)
    unique_labels = unique_labels[unique_labels != 0]
    
    # Build a mapping from label to region index
    label_to_region_idx = {}
    for idx, region in enumerate(components.regions):
        label_to_region_idx[region.label] = idx
    
    # Get centroids for non-background regions
    centroids = np.array([region.centroid for region in components.regions])
    
    # Calculate distance matrix
    dist_matrix = distance_matrix(centroids, centroids)
    link_matrix = dist_matrix < params.min_dist
    
    # Create a graph using labels as nodes
    G = nx.Graph()
    G.add_nodes_from(unique_labels)
    
    # Add edges between labels that are too close
    # We need to map back from region indices to labels
    edges_idx = np.argwhere(np.triu(link_matrix, k=1))
    edges_labels = []
    for i, j in edges_idx:
        label_i = components.regions[i].label
        label_j = components.regions[j].label
        edges_labels.append((label_i, label_j))
    
    G.add_edges_from(edges_labels)
    
    # Find connected components (groups to merge)
    connected_groups = list(nx.connected_components(G))
    
    # Create mapping from old labels to new labels
    old_to_new_label = {0: 0}  # Keep background as 0
    
    # Assign new labels to merged groups
    new_label = 1
    for group in connected_groups:
        for old_label in group:
            old_to_new_label[old_label] = new_label
        new_label += 1
    
    # Apply the mapping to create new labels array
    new_labels = np.zeros_like(components.labels)
    for old_label, new_label_value in old_to_new_label.items():
        new_labels[components.labels == old_label] = new_label_value
    
    return connectedComponent.from_labels(new_labels, intensity_image=components.intensity_image)