from dataclasses import dataclass

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
    nLabels: int
    labels: np.ndarray
    regions: List[_regionprops.RegionProperties]
    stats: np.ndarray | None = None
    intensity_image: np.ndarray | None = None
    _colors: np.ndarray = None

    @property
    def colors(self):
        if self._colors is None:
            self._colors = np.random.randint(30, 255, (self.nLabels, 3))
        return self._colors

    @classmethod
    def from_labels(cls, labels, stats=None, intensity_image=None, compute_stats=False):
        if compute_stats:
            stats = cls._compute_stats_from_labels(labels)
        return cls(
            nLabels=len(np.unique(labels)),
            labels=labels,
            regions=regionprops(labels, intensity_image),
            intensity_image=intensity_image,
            stats=stats
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
        stats = np.zeros((len(unique_labels), 5), dtype=np.int32)
        
        # Get bounding boxes for all objects at once
        slices = ndimage.find_objects(labels)
        
        for i, label in enumerate(unique_labels):
            if label == 0 or label > len(slices) or slices[label-1] is None:
                stats[i] = [0, 0, 0, 0, 0]
                continue
            
            # Get the slice for this label
            slice_obj = slices[label-1]
            
            # Extract bounding box from slices
            y_min, y_max = slice_obj[0].start, slice_obj[0].stop
            x_min, x_max = slice_obj[1].start, slice_obj[1].stop
            
            # Calculate area by counting pixels in the region
            region_mask = labels[slice_obj] == label
            area = np.count_nonzero(region_mask)
            
            stats[i] = [
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
        # unique_labels = np.unique(self.labels)
        # vis = np.zeros((*self.labels.shape, 3), dtype=np.uint8)
        # for i, lbl in enumerate(unique_labels):
        #     if lbl == 0:
        #         continue
        #     color = self.colors[i]
        #     vis[self.labels == lbl] = color
        # return vis
        unique_labels = np.unique(self.labels)
        
        # Map label values to color indices
        label_indices = np.searchsorted(unique_labels, self.labels)
        
        # Create color map (background stays black)
        color_map = np.zeros((len(unique_labels), 3), dtype=np.uint8)
        color_map[1:] = self.colors[1:]  # Skip background
        
        return color_map[label_indices]
    
    def __len__(self):
        return len(np.unique(self.labels))

    
    def save(self, filepath):
        """Save using numpy's compressed format."""
        filepath = Path(filepath)
        
        # Save all arrays and metadata in a single .npz file
        np.savez_compressed(
            filepath,
            nLabels=self.nLabels,
            labels=self.labels,
            stats=self.stats if self.stats is not None else np.array([]),
            intensity_image=self.intensity_image if self.intensity_image is not None else np.array([]),
            _colors=self._colors if self._colors is not None else np.array([]),
            has_stats=self.stats is not None,
            has_intensity=self.intensity_image is not None,
            has_colors=self._colors is not None
        )

    @classmethod
    def load(cls, filepath):
        """Load from numpy compressed file."""
        data = np.load(filepath, allow_pickle=False)
        
        instance = cls.from_labels(
            labels=data['labels'],
            stats=data['stats'] if data['has_stats'] else None,
            intensity_image=data['intensity_image'] if data['has_intensity'] else None
        )
        
        if data['has_colors']:
            instance._colors = data['_colors']
        
        return instance

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



def extract_patches(characterComponents: connectedComponent, images, border = None):
    """
    Extracts patches arround each character in characterComponents.
    """

    patches_list = []

    for image in images:
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
            patches.append(patch)

        patches_list.append(patches)
    
    return patches_list



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