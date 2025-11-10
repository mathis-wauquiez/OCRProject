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
    def from_labels(cls, labels, stats=None, intensity_image=None):
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

    @property
    def segm_img(self):
        unique_labels = np.unique(self.labels)
        vis = np.zeros((*self.labels.shape, 3), dtype=np.uint8)
        for i, lbl in enumerate(unique_labels):
            if lbl == 0:
                continue
            color = self.colors[i]
            vis[self.labels == lbl] = color
        return vis
    
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