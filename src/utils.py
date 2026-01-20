from dataclasses import dataclass, field

from collections import OrderedDict
from typing import NamedTuple, Any, Tuple, List

import numpy as np
from skimage.measure import _regionprops, regionprops
from pathlib import Path
from PIL import Image

import torch.nn as nn

def to_numpy(tensor):
    return tensor.permute(1,2,0).detach().cpu().numpy()


@dataclass
class connectedComponent:
    _nLabels: int
    _labels: np.ndarray
    _regions: List[_regionprops.RegionProperties]
    _stats: np.ndarray | None = None
    _intensity_image: np.ndarray | None = None
    _colors: np.ndarray = None

    _deleted_labels: List[int] = field(default_factory=list)
    _delete_reason: List[str] = field(default_factory=list)
    
    # New: track label merging
    _merge_mapping: dict = field(default_factory=dict)  # old_label -> new_label

    @property
    def labels(self):
        """Label array with deleted labels set to 0 and merged labels remapped"""
        l = self._labels.copy()
        
        # First, set deleted labels to 0
        l[np.isin(l, self._deleted_labels)] = 0
        
        # Then, apply merge mapping
        if self._merge_mapping:
            # Create lookup table for fast remapping
            max_label = l.max()
            lookup = np.arange(max_label + 1)
            
            # Update lookup with merge mapping
            for old_label, new_label in self._merge_mapping.items():
                if old_label <= max_label:
                    lookup[old_label] = new_label
            
            # Apply the mapping
            l = lookup[l]
        
        return l
    
    def merge(self, link_matrix, labels_list=None):
        """
        Merge labels based on a linking matrix using connected components.
        
        Parameters
        ----------
        link_matrix : np.ndarray
            Boolean matrix where link_matrix[i, j] = True means labels at 
            positions i and j in labels_list should be merged together.
        labels_list : list or np.ndarray, optional
            List of labels corresponding to the rows/columns of link_matrix.
            If None, uses all non-deleted, non-background labels in sorted order.
        
        Returns
        -------
        self : connectedComponent
            Returns self for method chaining
        
        Notes
        -----
        - Labels within the same connected component are merged to the smallest label in the group
        - Deleted labels are excluded from merging
        - The merge mapping is stored and applied when accessing the .labels property
        """
        import networkx as nx
        
        # Default to all active labels if not specified
        if labels_list is None:
            labels_list = np.unique(self.labels)
            labels_list = labels_list[labels_list != 0]  # Exclude background
        else:
            labels_list = np.array(labels_list)
        
        if len(labels_list) == 0:
            return self
        
        # Validate link_matrix shape
        n = len(labels_list)
        if link_matrix.shape != (n, n):
            raise ValueError(f"link_matrix shape {link_matrix.shape} doesn't match labels_list length {n}")
        
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(labels_list)
        
        # Add edges from link_matrix
        edges_idx = np.argwhere(np.triu(link_matrix, k=1))
        edges = [(labels_list[i], labels_list[j]) for i, j in edges_idx]
        G.add_edges_from(edges)
        
        # Find connected components (groups to merge)
        connected_groups = list(nx.connected_components(G))
        
        # Create merge mapping: all labels in a group map to the smallest label
        for group in connected_groups:
            if len(group) > 1:
                representative = min(group)  # Use smallest label as representative
                for label in group:
                    if label != representative:
                        self._merge_mapping[label] = representative
        
        return self
    
    def get_merge_representative(self, label: int) -> int:
        """
        Get the representative label for a merged label.
        
        Parameters
        ----------
        label : int
            The label to query
        
        Returns
        -------
        int
            The representative label (may be the same as input if not merged)
        """
        # Follow the chain of merges to find the final representative
        representative = label
        while representative in self._merge_mapping:
            representative = self._merge_mapping[representative]
        return representative
    
    def is_merged(self, label: int) -> bool:
        """Check if a label has been merged into another."""
        return label in self._merge_mapping
    
    def get_merged_groups(self):
        """
        Get all groups of merged labels.
        
        Returns
        -------
        dict
            Dictionary mapping representative labels to lists of merged labels
        """
        groups = {}
        
        # Add all merged labels
        for old_label, new_label in self._merge_mapping.items():
            representative = self.get_merge_representative(old_label)
            if representative not in groups:
                groups[representative] = [representative]
            if old_label not in groups[representative]:
                groups[representative].append(old_label)
        
        return groups
    
    def unmerge(self, label: int):
        """
        Unmerge a label, restoring it to its original state.
        
        Parameters
        ----------
        label : int
            The label to unmerge
        """
        if label not in self._merge_mapping:
            raise ValueError(f"Label {label} is not merged")
        
        del self._merge_mapping[label]
    
    def clear_merges(self):
        """Clear all merge mappings."""
        self._merge_mapping.clear()
    
    @property
    def nLabels(self):
        """Number of undeleted labels (including background)"""
        return len(np.unique(self.labels))
    
    @property
    def regions(self):
        """List of regions for undeleted labels only"""
        return [r for r in self._regions if r.label not in self._deleted_labels]
    
    @property
    def stats(self):
        """Stats array with only non-deleted labels (excludes deleted labels entirely)"""
        if self._stats is None:
            return None
        
        # Get all unique labels in the original array
        unique_labels = np.unique(self._labels)
        
        # Filter to keep only non-deleted labels that are within stats array bounds
        valid_labels = [label for label in unique_labels 
                       if label not in self._deleted_labels and label < len(self._stats)]
        
        # Return stats only for valid labels
        return self._stats[valid_labels]
    
    @stats.setter
    def stats(self, value):
        """Set the stats array (sets private _stats)"""
        self._stats = value
    
    @property
    def intensity_image(self):
        """Intensity image (unchanged by deletions)"""
        return self._intensity_image
    
    @intensity_image.setter
    def intensity_image(self, value):
        """Set the intensity image (sets private _intensity_image)"""
        self._intensity_image = value

    @property
    def colors(self):
        """Color array for visualization"""
        if self._colors is None:
            self._colors = np.random.randint(30, 255, (self._nLabels, 3))
        return self._colors

    @classmethod
    def from_labels(cls, labels, stats=None, intensity_image=None, compute_stats=False):
        if compute_stats:
            stats = cls._compute_stats_from_labels(labels)
        return cls(
            _nLabels=len(np.unique(labels)),
            _labels=labels,
            _regions=regionprops(labels, intensity_image),
            _intensity_image=intensity_image,
            _stats=stats
        )
    
    @classmethod
    def from_image(cls, image, connectivity=4):
        import cv2
        image_umat = cv2.UMat(image)
        nLabels, labels_umat, stats_umat, centroids_umat = cv2.connectedComponentsWithStats(
            image_umat, connectivity=connectivity
        )
        labels = labels_umat.get()
        stats = stats_umat.get()
        return cls.from_labels(labels, stats, image)

    @staticmethod
    def _compute_stats_from_labels(labels):
        from scipy import ndimage
        
        unique_labels = np.unique(labels)
        max_label = int(unique_labels.max())
        
        # Create stats array indexed by label value (like OpenCV does)
        # stats[label] = stats for that label
        stats = np.zeros((max_label + 1, 5), dtype=np.int32)
        
        # Get bounding boxes for all objects at once
        slices = ndimage.find_objects(labels)
        
        for label in unique_labels:
            if label == 0 or label > len(slices) or slices[label-1] is None:
                stats[label] = [0, 0, 0, 0, 0]
                continue
            
            # Get the slice for this label
            slice_obj = slices[label-1]
            
            # Extract bounding box from slices
            y_min, y_max = slice_obj[0].start, slice_obj[0].stop
            x_min, x_max = slice_obj[1].start, slice_obj[1].stop
            
            # Calculate area by counting pixels in the region
            region_mask = labels[slice_obj] == label
            area = np.count_nonzero(region_mask)
            
            stats[label] = [
                x_min,                    # CC_STAT_LEFT
                y_min,                    # CC_STAT_TOP
                x_max - x_min,            # CC_STAT_WIDTH
                y_max - y_min,            # CC_STAT_HEIGHT
                area                      # CC_STAT_AREA
            ]
        
        return stats

    @classmethod
    def from_image_watershed(cls, image, min_distance=10, connectivity=1, compute_stats=False,
                            binary_threshold=None, use_intensity=False):
        from scipy import ndimage
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        from skimage.filters import threshold_otsu
                
        # Convert to float for processing
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        min_distance = int(min_distance)
        
        # Create binary mask
        if binary_threshold is not None:
            binary = image > binary_threshold
        else:
            thresh = threshold_otsu(image)
            binary = image > thresh
        
        if use_intensity:
            elevation = -image
            coords = peak_local_max(image, min_distance=min_distance, labels=binary)
        else:
            distance = ndimage.distance_transform_edt(binary)
            elevation = -distance
            coords = peak_local_max(distance, min_distance=min_distance, labels=binary)
        
        # Create markers from peaks
        mask = np.zeros(binary.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask)
        
        # Apply watershed
        labels = watershed(elevation, markers, mask=binary, connectivity=connectivity)
        
        return cls.from_labels(labels, intensity_image=image, compute_stats=compute_stats)


    @property
    def segm_img(self):
        unique_labels = np.unique(self.labels)
        
        # Map label values to color indices
        label_indices = np.searchsorted(unique_labels, self.labels)
        
        # Create color map (background stays black)
        color_map = np.zeros((len(unique_labels), 3), dtype=np.uint8)
        
        # Generate colors only for non-background, non-deleted labels
        if len(unique_labels) > 1:
            # Use colors from the original color array, mapped to unique labels
            for i, label in enumerate(unique_labels[1:], start=1):
                if label < len(self.colors):
                    color_map[i] = self.colors[label]
                else:
                    # Fallback: generate random color if label is out of bounds
                    color_map[i] = np.random.randint(30, 255, 3)
        
        return color_map[label_indices]
        
    def deletion_viz(self, kept_color=(0, 255, 0), deleted_colors=None):
        """
        Create a visualization showing kept labels vs deleted labels by reason.
        
        Parameters
        ----------
        kept_color : tuple, optional
            RGB color for kept labels (default: green)
        deleted_colors : dict or list, optional
            Dictionary mapping deletion reasons to RGB colors, or list of colors.
            If None, generates random colors.
        
        Returns
        -------
        vis : np.ndarray
            RGB visualization image
        color_map : dict
            Dictionary mapping category names to RGB colors.
        """
        vis = np.zeros((*self._labels.shape, 3), dtype=np.uint8)
        
        # Get unique deletion reasons
        unique_reasons = list(set(self._delete_reason)) if self._delete_reason else []
        
        # Create reason to color mapping
        reason_to_color = {}
        if deleted_colors is None:
            # Generate random colors
            for reason in unique_reasons:
                reason_to_color[reason] = tuple(np.random.randint(30, 255, 3).tolist())
        elif isinstance(deleted_colors, dict):
            # Use provided dictionary, generate random for missing reasons
            for reason in unique_reasons:
                if reason in deleted_colors:
                    reason_to_color[reason] = deleted_colors[reason]
                else:
                    # Generate consistent color based on hash for unknown reasons
                    np.random.seed(hash(reason) % (2**32))
                    reason_to_color[reason] = tuple(np.random.randint(30, 255, 3).tolist())
        else:
            # Assume it's a list
            if len(deleted_colors) < len(unique_reasons):
                raise ValueError(f"Need at least {len(unique_reasons)} colors for deletion reasons")
            reason_to_color = {reason: deleted_colors[i] for i, reason in enumerate(unique_reasons)}
        
        # Build complete color map for legend
        color_map = {'kept': kept_color}
        color_map.update(reason_to_color)
        
        # Color kept labels
        kept_mask = np.isin(self._labels, self._deleted_labels, invert=True) & (self._labels > 0)
        vis[kept_mask] = kept_color
        
        # Color deleted labels by reason
        for label, reason in zip(self._deleted_labels, self._delete_reason):
            color = reason_to_color[reason]
            vis[self._labels == label] = color
        
        return vis, color_map
    
    def __len__(self):
        return len(np.unique(self.labels))

    
    def save(self, filepath):
        """Save using numpy's compressed format."""
        filepath = Path(filepath)
        
        # Save all arrays and metadata in a single .npz file
        np.savez_compressed(
            filepath,
            _nLabels=self._nLabels,
            _labels=self._labels,
            _deleted_labels=self._deleted_labels,
            _delete_reason=self._delete_reason,
            _merge_mapping_keys=list(self._merge_mapping.keys()),
            _merge_mapping_values=list(self._merge_mapping.values()),
            _stats=self._stats if self._stats is not None else np.array([]),
            _intensity_image=self._intensity_image if self._intensity_image is not None else np.array([]),
            _colors=self._colors if self._colors is not None else np.array([]),
            has_stats=self._stats is not None,
            has_intensity=self._intensity_image is not None,
            has_colors=self._colors is not None
        )

    @classmethod
    def load(cls, filepath):
        """Load from numpy compressed file."""
        data = np.load(filepath, allow_pickle=False)
        
        # Reconstruct regions from labels
        intensity_img = data['_intensity_image'] if data['has_intensity'] else None
        regions = regionprops(data['_labels'], intensity_img)
        
        instance = cls(
            _nLabels=int(data['_nLabels']),
            _labels=data['_labels'],
            _regions=regions,
            _stats=data['_stats'] if data['has_stats'] else None,
            _intensity_image=intensity_img
        )
        
        # Load deletion tracking
        instance._deleted_labels = data['_deleted_labels'].tolist()
        instance._delete_reason = data['_delete_reason'].tolist()
        
        # Load merge mapping
        if '_merge_mapping_keys' in data:
            merge_keys = data['_merge_mapping_keys'].tolist()
            merge_values = data['_merge_mapping_values'].tolist()
            instance._merge_mapping = dict(zip(merge_keys, merge_values))
        
        if data['has_colors']:
            instance._colors = data['_colors']
        
        return instance
    
    def delete(self, label: int, reason: str = ""):
        """
        Mark a label as deleted.
        
        Parameters
        ----------
        label : int
            The label to delete
        reason : str, optional
            Reason for deletion (default: "")
        """
        if label == 0:
            raise ValueError("Cannot delete background label (0)")
        
        if label not in np.unique(self._labels):
            raise ValueError(f"Label {label} does not exist in the component")
        
        if label in self._deleted_labels:
            print(f"Warning: Label {label} is already deleted")
            return
        
        self._deleted_labels.append(label)
        self._delete_reason.append(reason)

    def is_deleted(self, label: int) -> bool:
        """Check if a label is deleted."""
        return label in self._deleted_labels

    def undelete(self, label: int):
        """
        Restore a previously deleted label.
        
        Parameters
        ----------
        label : int
            The label to restore
        """
        if label not in self._deleted_labels:
            raise ValueError(f"Label {label} is not deleted")
        
        idx = self._deleted_labels.index(label)
        self._deleted_labels.pop(idx)
        self._delete_reason.pop(idx)

    def get_delete_reason(self, label: int) -> str:
        """Get the deletion reason for a label."""
        if label not in self._deleted_labels:
            raise ValueError(f"Label {label} is not deleted")
        idx = self._deleted_labels.index(label)
        return self._delete_reason[idx]

    def __deepcopy__(self, memo):
        """Custom deepcopy that regenerates regions instead of copying them."""
        # Create new instance without calling __init__
        cls = self.__class__
        result = cls.__new__(cls)
        
        # Add to memo to handle circular references
        memo[id(self)] = result
        
        # Deep copy everything except _regions
        result._nLabels = self._nLabels
        result._labels = self._labels.copy()
        result._deleted_labels = self._deleted_labels.copy()
        result._delete_reason = self._delete_reason.copy()
        result._merge_mapping = self._merge_mapping.copy()
        result._stats = self._stats.copy() if self._stats is not None else None
        result._intensity_image = self._intensity_image.copy() if self._intensity_image is not None else None
        result._colors = self._colors.copy() if self._colors is not None else None
        
        # Regenerate regions instead of copying
        result._regions = regionprops(result._labels, result._intensity_image)
        
        return result
        
def torch_to_pil(tensor, max_size=None, max_normalize=False, mean_normalize=True):
    if max_normalize:
        tensor = tensor - tensor.min()
        tensor /= tensor.max()
    elif mean_normalize:
        tensor -= tensor.mean()
        tensor /= tensor.std() * 2
        tensor += .5
        tensor.clip_(0, 1)

    np_array = (tensor.permute(1,2,0)*255).detach().cpu().numpy().astype(np.uint8)
    if np_array.shape[-1] == 1:
        np_array = np_array[..., 0]
    
    img = Image.fromarray(
        np_array
    )
    if max_size is None:
        return img
    max_length = max(img.width, img.height)
    scaling_factor = min(max_size / max_length, 1)
    return img.resize((int(img.width * scaling_factor), int(img.height * scaling_factor)))


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def nearest_multiple(a: int, div: int):
    if a%div != 0:
        return a + (div - a %div)
    return a

nearest_32 = lambda a: nearest_multiple(a, 32)