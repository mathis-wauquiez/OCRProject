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

from ..utils import connectedComponent
from .distances import compute_mahalanobis_distance, l2
from .params import imageComponentsParams
from .detection.model_wrapper import craftWrapper

from . import params

class imageComponentsPipeline:

    def __init__(self, params: imageComponentsParams, craftDetector: craftWrapper):
        self.params = params
        self.craftDetector = craftDetector

    def binarize(self, im_pil):
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

        if type(self.params.threshold) == str:
            if not self.params.threshold in methods:
                raise ValueError(f'Binarization method f{self.params.threshold} not known.')
            thresh = methods[self.params.threshold](im_np_g)
        else:
            thresh = self.params.threshold

        binary_img = im_np_g < thresh
        return binary_img
    

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
                pass
            else:
                labels_to_keep.add(label)
        
        # Apply filter: set filtered components to background
        labels = components.labels.copy()
        labels[~np.isin(labels, list(labels_to_keep))] = 0
        return connectedComponent.from_labels(labels, intensity_image=components.intensity_image)


    def merge_from_similarities(self, imageComponents: connectedComponent, charactersComponents: connectedComponent, similarities):
                
        uniqueImageComponents = np.unique(imageComponents.labels)
        uniqueCharsComponents = np.unique(charactersComponents.labels)

        # Remove background label (0) from consideration for mapping
        non_bg_image_components = uniqueImageComponents[uniqueImageComponents != 0]
        
        bestIdx = similarities.argmax(axis=0)  # similarities is (nCharsComponents, nImageComponents)
        bestSimilarities = similarities.max(axis=0)  # Get the max similarity value for each image component
        
        # bestChars maps each non-background image component to a character region
        bestChars = uniqueCharsComponents[bestIdx]
        
        # Set to background (0) where similarity is below threshold
        bestChars[bestSimilarities < self.params.similarity_threshold] = 0

        old_labels = imageComponents.labels

        # Create lookup table - background (0) stays as background
        max_label = uniqueImageComponents.max()
        lookup = np.zeros(max_label + 1, dtype=bestChars.dtype)
        
        # Background remains 0 (already set by zeros)
        # Map non-background components
        lookup[non_bg_image_components] = bestChars
        
        # Apply the mapping
        new_labels = lookup[old_labels]

        return connectedComponent.from_labels(new_labels, intensity_image=imageComponents.intensity_image)

    def compute_similarities(self, imageComponents: connectedComponent, charactersComponents: connectedComponent):
        # Pre-compute other_centroids ONCE before the loop
        other_centroids = torch.tensor([region.centroid for region in imageComponents.regions]).to(params.device)
        other_centroids = torch.flip(other_centroids, dims=[1])
        other_centroids = self.craftDetector.map_original_to_preprocessed(
            other_centroids, 
            original_shape=imageComponents.labels.shape
        ) / 2

        similarities = torch.zeros((len(charactersComponents.regions), len(imageComponents.regions))).to(params.device)

        for i, region in enumerate(charactersComponents.regions):
            centroid = torch.tensor(region.centroid[::-1]).to(params.device)

            if self.params.similarity_metric == 'mahalanobis':
                VI = torch.from_numpy(region.inertia_tensor).to(params.device)
                similarities[i, :] = - compute_mahalanobis_distance(centroid, other_centroids, VI=VI)
            elif self.params.similarity_metric == 'euclidian':
                similarities[i, :] = - l2(centroid, other_centroids) ** .5

        return similarities.cpu().numpy()


    def filter_bad_characters(self, characterComponents: connectedComponent):

        unique_labels = np.unique(characterComponents.labels)
        labels_to_keep = {0}
        
        # Filter by aspect ratio
        for i, region in enumerate(characterComponents.regions):
            label = unique_labels[i + 1]

            bbox = region.bbox
            filled_area_portion = region.area_filled / region.area_bbox
            
            h, w = bbox[2] - bbox[0], bbox[3] - bbox[1]
            aspect_ratio = max(h/w, w/h)

            if filled_area_portion > self.params.max_filled_area_portion or\
               h < self.params.min_box_size[1] or w < self.params.min_box_size[0] or\
               h > self.params.max_box_size[1] or w > self.params.max_box_size[0] or\
               aspect_ratio > self.params.max_aspect_ratio or\
                region.area_filled < self.params.min_area:
                pass

            else:
                labels_to_keep.add(label)
        
        # Apply filter: set filtered components to background
        labels = characterComponents.labels.copy()
        labels[~np.isin(labels, list(labels_to_keep))] = 0
        return connectedComponent.from_labels(labels, intensity_image=characterComponents.intensity_image, compute_stats=True)



    def forward(self, im_pil, craftComponents):
        binary_img = self.binarize(im_pil)
        img_components = connectedComponent.from_image(binary_img.astype(np.uint8)*255)
        filtered_img_components = self.filter_image_components(img_components)
        similarities = self.compute_similarities(filtered_img_components, craftComponents)
        associated_components = self.merge_from_similarities(filtered_img_components, craftComponents, similarities)

        cc_filtered = self.filter_centroids_by_contour_proximity(
            centroids_components=associated_components,  # Characters to filter
            reference_components=img_components,         # Large components to check against
        )

        
        filtered_characters = self.filter_bad_characters(cc_filtered)


        return binary_img, img_components, filtered_img_components, associated_components, cc_filtered, filtered_characters
    
    def filter_centroids_by_contour_proximity(
        self, 
        centroids_components,  # connectedComponent
        reference_components,  # connectedComponent
        distance_threshold=None,
        min_component_size=None
    ):
        """
        Filter out centroid components that are too close to large component contours.
        
        Args:
            centroids_components: Components to filter (e.g., character detections)
            reference_components: Components whose contours to check against
            distance_threshold: Max distance in pixels (centroids closer = excluded)
            min_component_size: Only check components with area > this value
        
        Returns:
            Filtered connectedComponent with excluded centroids set to background
        
        Example:
            filtered = pipeline.filter_centroids_by_contour_proximity(
                centroids_components=character_components,
                reference_components=image_components,
                distance_threshold=50,
                min_component_size=10000
            )
        """
        distance_threshold = distance_threshold or self.params.cc_distance_threshold
        min_component_size = min_component_size or self.params.cc_min_comp_size

        unique_centroids_labels = np.unique(centroids_components.labels)
        to_keep = set(unique_centroids_labels)
        
        # Extract centroids and convert from (y,x) to (x,y) format
        centroids = np.array([region.centroid for region in centroids_components.regions])
        centroids = torch.tensor(centroids, device=params.device, dtype=torch.float32).flip(-1)
        
        # Find large components
        counts = np.bincount(reference_components.labels.reshape(-1))
        large_mask = counts > min_component_size
        large_indices = np.arange(len(counts))[large_mask]
        
        
        # Check each large component
        for idx in large_indices:
            if idx == 0:  # Skip background
                continue
                
            bin_image = (reference_components.labels == idx).astype(np.uint8)
            min_dists = self._get_contour_distances(bin_image, centroids)
            
            # Find centroids too close
            close_mask = min_dists < distance_threshold
            num_close = close_mask.sum().item()
            
            if num_close > 0:
                closest_idx = min_dists.argmin()
                close_mask[closest_idx] = False  # Don't exclude the closest one
                
                # Recalculate number of centroids to exclude (after keeping closest)
                num_close = close_mask.sum().item()
                
                if num_close > 0:
                    # Map boolean mask to actual labels (skip background at index 0)
                    close_labels = unique_centroids_labels[1:][close_mask.cpu().numpy()]
                    to_keep -= set(close_labels.tolist())
        
        # Apply filter
        filtered_labels = centroids_components.labels.copy()
        to_exclude = set(unique_centroids_labels) - to_keep
        filtered_labels[np.isin(filtered_labels, list(to_exclude))] = 0
                
        return connectedComponent.from_labels(
            filtered_labels,
            intensity_image=centroids_components.intensity_image
        )


    def _extract_contour_lines(self, bin_image):
        """Extract line segments from binary image contours."""
        image = bin_image.astype(np.uint8) * 255
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return np.empty((0, 2, 2))
        
        line_segments = []
        for contour in contours:
            contour = contour.squeeze(1)  # (N, 1, 2) → (N, 2)
            contour = np.vstack([contour, contour[0:1]])  # Close contour
            lines = np.stack((contour[:-1], contour[1:]), axis=1)
            line_segments.append(lines)
        
        return np.concatenate(line_segments, axis=0)


    def _point_to_line_segment_distance(self, points, lines):
        """Calculate point-to-line-segment distances."""
        p1, p2 = lines[:, 0], lines[:, 1]
        v = p2 - p1
        w = points[:, None, :] - p1[None, :, :]
        
        c1 = torch.einsum('ijk,jk->ij', w, v)
        c2 = torch.einsum('jk,jk->j', v, v)[None, :]
        t = (c1 / (c2 + 1e-10)).clamp(0, 1)
        
        closest = p1[None, :, :] + t[:, :, None] * v[None, :, :]
        diff = points[:, None, :] - closest
        return torch.sqrt(torch.einsum('ijk,ijk->ij', diff, diff))


    def _get_contour_distances(self, bin_image, points):
        """Get minimum distance from each point to image contours."""
        lines = self._extract_contour_lines(bin_image)
        
        if len(lines) == 0:
            return torch.full((points.shape[0],), float('inf'),
                            device=points.device, dtype=points.dtype)
        
        lines = torch.tensor(lines, device=points.device, dtype=points.dtype)
        distances = self._point_to_line_segment_distance(points, lines)
        return distances.min(dim=1).values



def extract_lines(bin_image):
    """Extract line segments from contours."""
    image = bin_image.astype(np.uint8) * 255
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.empty((0, 2, 2))
    
    line_segments = []
    for contour in contours:
        contour = contour.squeeze(1)
        contour = np.vstack([contour, contour[0:1]])
        lines = np.stack((contour[:-1], contour[1:]), axis=1)
        line_segments.append(lines)
    
    return np.concatenate(line_segments, axis=0)


def point_to_line_segment_distance(points, lines):
    """Calculate distances from points to line segments."""
    p1, p2 = lines[:, 0], lines[:, 1]
    v = p2 - p1
    w = points[:, None, :] - p1[None, :, :]
    
    c1 = torch.einsum('ijk,jk->ij', w, v)
    c2 = torch.einsum('jk,jk->j', v, v)[None, :]
    t = (c1 / (c2 + 1e-10)).clamp(0, 1)
    
    closest = p1[None, :, :] + t[:, :, None] * v[None, :, :]
    diff = points[:, None, :] - closest
    return torch.sqrt(torch.einsum('ijk,ijk->ij', diff, diff))


def get_distances(bin_image, points):
    """Get minimum distances from points to contours."""
    lines = extract_lines(bin_image)
    
    if len(lines) == 0:
        inf_dists = torch.full((points.shape[0],), float('inf'),
                              device=points.device, dtype=points.dtype)
        zero_idx = torch.zeros((points.shape[0],), dtype=torch.long, 
                              device=points.device)
        return inf_dists, zero_idx
    
    lines = torch.tensor(lines, device=points.device, dtype=points.dtype)
    distances = point_to_line_segment_distance(points, lines)
    min_distances, min_indices = distances.min(dim=1)
    
    return min_distances, min_indices
