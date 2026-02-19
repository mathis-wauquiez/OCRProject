"""
This file defines all the operations that are done on the image directly, including:
- binarization
- connected components extraction
- filtering the connected components by aspect ratio
- associate the components from the image with CRAFT's detections based on a distance metric
- filter the characters by aspect ratio and filled area percentage
"""

from skimage.filters import *
import cv2
import numpy as np
import torch

from ..utils import connectedComponent, Timer
from .distances import compute_mahalanobis_distance, compute_mahalanobis_distance_batched, l2
from .params import imageComponentsParams
from .detection.model_wrapper import craftWrapper
from . import params

from typing import NamedTuple

class compsResult(NamedTuple):
    binary_img: np.ndarray
    img_components: connectedComponent
    filtered_img_components: connectedComponent
    characters: connectedComponent
    similarity_matrix: np.ndarray
    characters_before_contour_filter: connectedComponent


class imageComponentsPipeline:

    def __init__(self, params: imageComponentsParams, craftDetector: craftWrapper):
        self.params = params
        self.craftDetector = craftDetector

    def binarize(self, im_pil):
        """Binarize image using specified threshold method."""
        im_np = np.array(im_pil)
        im_np_g = cv2.cvtColor(im_np, cv2.COLOR_RGB2GRAY)

        methods = {
            'otsu': threshold_otsu,
            'li': threshold_li,
            'isodata': threshold_isodata,
            'mean': threshold_mean,
            'triangle': threshold_triangle,
            'min': threshold_minimum,
            'minimum': threshold_minimum,
            'yen': threshold_yen
        }

        if isinstance(self.params.threshold, str):
            if self.params.threshold not in methods:
                raise ValueError(f'Binarization method {self.params.threshold} not known.')
            thresh = methods[self.params.threshold](im_np_g)
        else:
            thresh = self.params.threshold

        return im_np_g < thresh

    def filter_image_components(self, components: connectedComponent):
        """Vectorized component filtering by aspect ratio."""
        regions = components.regions
        if not regions:
            return components
        
        # Extract properties in single pass
        labels = np.array([r.label for r in regions])
        major_lengths = np.array([r.axis_major_length for r in regions])
        minor_lengths = np.array([r.axis_minor_length for r in regions])
        
        aspect_ratios = major_lengths / np.maximum(minor_lengths, 1)
        
        # Vectorized filter
        delete_mask = ((aspect_ratios > self.params.min_image_component_aspect_ratio) & 
                    (major_lengths > self.params.min_image_component_axis_major_length_criterion))
        
        for label in labels[delete_mask]:
            components.delete(int(label), reason="high_aspect_ratio")
        
        return components
    
    def merge_from_similarities(self, imageComponents: connectedComponent, 
                            charactersComponents: connectedComponent, similarities):
        """Merge image components with character components based on similarity scores.
        
        Assigns CRAFT component labels to image components based on best similarity match.
        """
        # Get active labels from image components
        image_labels = np.array([r.label for r in imageComponents.regions 
                                if not imageComponents.is_deleted(r.label)])
        
        # Get the actual CRAFT labels (not character component labels!)
        # These are the labels from charactersComponents which ARE the CRAFT components
        craft_regions = [r for r in charactersComponents.regions 
                        if not charactersComponents.is_deleted(r.label)]
        craft_labels = np.array([r.label for r in craft_regions])
        
        # Find best matching CRAFT component for each image component
        best_craft_idx = similarities.argmax(axis=0)  # For each image component
        best_similarities = similarities.max(axis=0)
        
        # Get the CRAFT label for each best match
        best_craft_labels = craft_labels[best_craft_idx]
        
        # Set to background where similarity is below threshold
        best_craft_labels[best_similarities < self.params.similarity_threshold] = 0

        # Create label mapping: image_component_label -> craft_label
        max_label = imageComponents._labels.max()
        lookup = np.zeros(max_label + 1, dtype=best_craft_labels.dtype)
        lookup[image_labels] = best_craft_labels
        
        # Apply mapping: replace image labels with their matched CRAFT labels
        new_labels = lookup[imageComponents.labels]

        return connectedComponent.from_labels(new_labels, 
                                            intensity_image=imageComponents.intensity_image)
    
    def compute_similarities(self, imageComponents: connectedComponent,
                            charactersComponents: connectedComponent):
        """Compute similarity matrix between image and character components."""
        # Get active image component centroids
        active_img_regions = [r for r in imageComponents.regions
                             if not imageComponents.is_deleted(r.label)]

        img_centroids = torch.tensor([r.centroid for r in active_img_regions], dtype=torch.float32).to(params.device)
        img_centroids = torch.flip(img_centroids, dims=[1])
        img_centroids = self.craftDetector.map_original_to_preprocessed(
            img_centroids, original_shape=imageComponents._labels.shape) / 2

        # Collect active CRAFT regions
        active_craft_regions = charactersComponents.regions  # already filters deleted
        n_chars = len(active_craft_regions)
        n_imgs = len(active_img_regions)

        if n_chars == 0 or n_imgs == 0:
            return np.zeros((n_chars, n_imgs), dtype=np.float32)

        # Vectorised path — one batched call instead of N individual ones
        craft_centroids = torch.tensor(
            np.array([r.centroid[::-1] for r in active_craft_regions]),
            device=params.device, dtype=torch.float32,
        )

        if self.params.similarity_metric == 'mahalanobis':
            VIs = torch.tensor(
                np.stack([r.inertia_tensor for r in active_craft_regions]),
                device=params.device, dtype=torch.float32,
            )  # (N, 2, 2)
            similarities = -compute_mahalanobis_distance_batched(
                craft_centroids, img_centroids, VIs,
            )
        elif self.params.similarity_metric == 'euclidian':
            similarities = -torch.cdist(craft_centroids.unsqueeze(0),
                                        img_centroids.unsqueeze(0)).squeeze(0)
        else:
            similarities = torch.zeros((n_chars, n_imgs), device=params.device)

        return similarities.cpu().numpy()

    def filter_bad_characters(self, characterComponents: connectedComponent):
        """Vectorized character filtering."""
        regions = [r for r in characterComponents.regions 
                if not characterComponents.is_deleted(r.label)]
        if not regions:
            return characterComponents
        
        # Extract all properties at once (single pass through regions)
        n = len(regions)
        labels = np.empty(n, dtype=np.int32)
        bboxes = np.empty((n, 4), dtype=np.int32)
        areas_filled = np.empty(n, dtype=np.float32)
        areas_bbox = np.empty(n, dtype=np.float32)
        
        for i, r in enumerate(regions):
            labels[i] = r.label
            bboxes[i] = r.bbox
            areas_filled[i] = r.area_filled
            areas_bbox[i] = r.area_bbox
        
        # Vectorized computations
        h = bboxes[:, 2] - bboxes[:, 0]
        w = bboxes[:, 3] - bboxes[:, 1]
        aspect_ratios = np.maximum(h / np.maximum(w, 1), w / np.maximum(h, 1))
        filled_portions = areas_filled / np.maximum(areas_bbox, 1)
        
        # Vectorized condition checks (order matters - first match wins)
        delete_mask = np.zeros(n, dtype=bool)
        reasons = np.empty(n, dtype=object)
        
        m = filled_portions > self.params.max_filled_area_portion
        reasons[m & ~delete_mask] = "filled_area_too_high"
        delete_mask |= m
        
        m = (h < self.params.min_box_size[1]) | (w < self.params.min_box_size[0])
        reasons[m & ~delete_mask] = "too_small"
        delete_mask |= m
        
        m = (h > self.params.max_box_size[1]) | (w > self.params.max_box_size[0])
        reasons[m & ~delete_mask] = "too_large"
        delete_mask |= m
        
        m = aspect_ratios > self.params.max_aspect_ratio
        reasons[m & ~delete_mask] = "aspect_ratio_too_high"
        delete_mask |= m
        
        m = areas_filled < self.params.min_area
        reasons[m & ~delete_mask] = "area_too_small"
        delete_mask |= m
        
        # delete
        for label, reason in zip(labels[delete_mask], reasons[delete_mask]):
            characterComponents.delete(int(label), reason=str(reason))
        
        return characterComponents

    def forward(self, im_pil, craftComponents, return_intermediate=False):
        """Main processing pipeline."""
        timer = Timer()

        binary_img = self.binarize(im_pil)
        timer('#|1 Binary image: {:.2f}')

        img_components = connectedComponent.from_image(binary_img.astype(np.uint8) * 255)
        filtered_img_components = self.filter_image_components(img_components)
        timer('#|2 Filtering components: {:.2f}')
        
        similarities = self.compute_similarities(filtered_img_components, craftComponents)
        timer('#|3 Computing similarities: {:.2f}')

        characters = self.merge_from_similarities(filtered_img_components, craftComponents, similarities)
        timer('#|4 Merging: {:.2f}')
        
        # Save state before contour filtering if we need intermediate results
        if return_intermediate:
            import copy
            characters_before_contour = copy.deepcopy(characters)
        else:
            characters_before_contour = None
        
        characters = self.filter_centroids_by_contour_proximity(characters, img_components)
        timer('#|5 Filtering close contours: {:.2f}')

        characters = self.filter_bad_characters(characters)
        timer('#|6 Filtering bad characters: {:.2f}')

        return compsResult(
            binary_img=binary_img,
            img_components=img_components,
            filtered_img_components=filtered_img_components,
            characters=characters,
            similarity_matrix=similarities,
            characters_before_contour_filter=characters_before_contour
        )

    def filter_centroids_by_contour_proximity(
        self, 
        centroids_components,
        reference_components,
        distance_threshold=None,
        min_component_size=None
    ):
        """Filter out centroid components that are too close to large component contours."""
        distance_threshold = distance_threshold or self.params.cc_distance_threshold
        min_component_size = min_component_size or self.params.cc_min_comp_size
        
        # Get active centroids
        active_regions = [r for r in centroids_components.regions 
                        if not centroids_components.is_deleted(r.label)]
        if not active_regions:
            return centroids_components
            
        active_labels = np.array([r.label for r in active_regions])
        centroids = np.array([r.centroid for r in active_regions])
        centroids = torch.tensor(centroids, device=params.device, dtype=torch.float32).flip(-1)
        
        ref_labels = reference_components.labels
        
        # Find large components
        counts = np.bincount(ref_labels.ravel())
        large_indices = np.where((counts > min_component_size) & (np.arange(len(counts)) != 0))[0]
        
        if len(large_indices) == 0:
            return centroids_components
        
        # Batch extract ALL contours at once
        all_lines_list = []
        component_indices = []
        
        for idx in large_indices:
            bin_image = (ref_labels == idx).astype(np.uint8)  # Uses cached array
            lines = extract_contour_lines(bin_image)
            if len(lines) > 0:
                all_lines_list.append(lines)
                component_indices.extend([idx] * len(lines))
        
        if not all_lines_list:
            return centroids_components
        
        # Single batched distance computation
        all_lines = np.concatenate(all_lines_list, axis=0)
        all_lines_t = torch.tensor(all_lines, device=params.device, dtype=torch.float32)
        all_distances = point_to_line_segment_distance(centroids, all_lines_t)  # (n_centroids, n_lines)
        
        component_indices = np.array(component_indices)
        labels_to_delete = set()
        
        # Process each component's results
        for idx in np.unique(component_indices):
            line_mask = component_indices == idx
            comp_distances = all_distances[:, line_mask]
            min_dists = comp_distances.min(dim=1).values
            
            close_mask = min_dists < distance_threshold
            num_close = close_mask.sum().item()
            
            if num_close > 1:
                closest_idx = min_dists.argmin()
                close_mask[closest_idx] = False
                labels_to_delete.update(active_labels[close_mask.cpu().numpy()].tolist())
        
        # Batch delete at the end
        for label in labels_to_delete:
            centroids_components.delete(label, reason="too_close_to_contour")
        
        return centroids_components

# Utility functions for contour distance computation

def extract_contour_lines(bin_image):
    """Extract line segments from binary image contours."""
    image = (bin_image.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.empty((0, 2, 2))
    
    line_segments = []
    for contour in contours:
        contour = contour.squeeze(1)  # (N, 1, 2) → (N, 2)
        if len(contour) < 2:
            continue
        # Close the contour
        contour = np.vstack([contour, contour[0:1]])
        lines = np.stack((contour[:-1], contour[1:]), axis=1)
        line_segments.append(lines)
    
    return np.concatenate(line_segments, axis=0) if line_segments else np.empty((0, 2, 2))


def point_to_line_segment_distance(points, lines):
    """Calculate minimum distances from points to line segments."""
    p1, p2 = lines[:, 0], lines[:, 1]
    v = p2 - p1
    w = points[:, None, :] - p1[None, :, :]
    
    # Project points onto line segments
    c1 = torch.einsum('ijk,jk->ij', w, v)
    c2 = torch.einsum('jk,jk->j', v, v)[None, :]
    t = (c1 / (c2 + 1e-10)).clamp(0, 1)
    
    # Find closest points on segments
    closest = p1[None, :, :] + t[:, :, None] * v[None, :, :]
    diff = points[:, None, :] - closest
    
    return torch.sqrt(torch.einsum('ijk,ijk->ij', diff, diff))


def get_contour_distances(bin_image, points):
    """Get minimum distance from each point to image contours."""
    lines = extract_contour_lines(bin_image)
    
    if len(lines) == 0:
        return torch.full((points.shape[0],), float('inf'),
                         device=points.device, dtype=points.dtype)
    
    lines = torch.tensor(lines, device=points.device, dtype=points.dtype)
    distances = point_to_line_segment_distance(points, lines)
    return distances.min(dim=1).values