"""
Patch database built on pandas DataFrame for efficient storage and querying.
Safe: does not store skimage RegionProperties in the DataFrame pickle.
If a corrupted pickle exists, load() will automatically rebuild from saved components/images/labels.
"""

from typing import Optional, Dict, Any, Union, List
from pathlib import Path

import pandas as pd
import numpy as np
from torch import Tensor
from PIL import Image
from skimage.measure import regionprops

from ..utils import connectedComponent


class PatchDatabase(pd.DataFrame):
    """
    DataFrame-based patch database. Each row is a patch (one region).
    """

    _metadata = ['_components', '_metadata_list', '_images', '_labels']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, '_components'):
            self._components: List[connectedComponent] = []
        if not hasattr(self, '_metadata_list'):
            self._metadata_list: List[Dict[str, Any]] = []
        if not hasattr(self, '_images'):
            self._images: List[np.ndarray] = []
        if not hasattr(self, '_labels'):
            self._labels: List[np.ndarray] = []   # store connected-component maps

    @property
    def _constructor(self):
        return PatchDatabase

    def add_image(self,
                  img: Union[Image.Image, np.ndarray, Tensor],
                  components: connectedComponent,
                  image_name: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Add an image together with its connected components.
        IMPORTANT: we store region index and bbox only; RegionProperties objects are NOT stored.
        """
        img = self._to_numpy(img)
        image_idx = len(self._components)

        if image_name is None:
            image_name = f'image_{image_idx}'

        # store components and supporting data
        self._components.append(components)
        self._metadata_list.append(metadata or {})
        self._images.append(img)
        # store the label image (connected-component map) so regions can be reconstructed on load
        self._labels.append(components.labels)

        rows = []
        for region_idx, region in enumerate(components.regions):
            h1, w1, h2, w2 = region.bbox
            patch = img[h1:h2, w1:w2]
            bbox = (int(h1), int(w1), int(h2), int(w2))

            rows.append({
                'image_idx': int(image_idx),
                'image_name': image_name,
                'region_idx': int(region_idx),
                'region': bbox,     # lightweight, serializable bbox tuple
                'bbox': bbox,
                'height': int(h2 - h1),
                'width': int(w2 - w1),
                'area': int((h2 - h1) * (w2 - w1)),
                'image': patch,
            })

        new_df = pd.DataFrame(rows)
        result = pd.concat([self, new_df], ignore_index=True)
        # preserve subclass internals
        self._mgr = result._mgr
        return self

    def _to_numpy(self, img: Union[Image.Image, np.ndarray, Tensor]) -> np.ndarray:
        """
        Convert PIL / torch Tensor / numpy to float32 numpy in [0,1].
        """
        if isinstance(img, Image.Image):
            arr = np.array(img, dtype=np.float32)
            return arr / 255.0 if arr.max() > 1.0 else arr.astype(np.float32)
        if isinstance(img, Tensor):
            arr = img.detach().cpu().numpy()
            arr = arr.astype(np.float32)
            return arr / 255.0 if arr.max() > 1.0 else arr
        if isinstance(img, np.ndarray):
            arr = img.astype(np.float32)
            return arr / 255.0 if arr.max() > 1.0 else arr
        # fallback: return as-is (rare)
        return np.array(img, dtype=np.float32)

    def save(self, filepath: str, columns: Optional[List[str]] = None):
        """
        Save DB components, metadata, images, labels and DataFrame (pickle).
        This will DROP any non-serializable column (like raw RegionProperties) before pickling.
        """
        filepath = Path(filepath)

        # 1) components -> folder of .npz files (using connectedComponent.save)
        components_dir = filepath.parent / (filepath.stem + '_components')
        components_dir.mkdir(exist_ok=True, parents=True)
        for idx, comp in enumerate(self._components):
            comp.save(str(components_dir / f'comp_{idx}.npz'))

        # 2) metadata
        pd.to_pickle(self._metadata_list, filepath.parent / (filepath.stem + '_metadata.pkl'))

        # 3) images
        if len(self._images) > 0:
            np.savez_compressed(filepath.parent / (filepath.stem + '_images.npz'), *self._images)

        # 4) labels (connected-component maps)
        if len(self._labels) > 0:
            np.savez_compressed(filepath.parent / (filepath.stem + '_labels.npz'), *self._labels)

        # 5) DataFrame - but drop any truly unsafe columns first
        df_to_save = self[columns] if columns else self
        df_to_save = df_to_save.copy()

        # Common unsafe column names that could contain unserializable objects:
        drop_if_present = [c for c in ['regionprops', 'RegionProperties', 'region_obj', 'region'] if c in df_to_save.columns]
        # Note: we DO keep the 'region' column when it contains bbox tuples; this line only drops if it's not pickle-friendly.
        # To be conservative, try to detect non-serializable entries: if any cell in 'region' is not a tuple/list of ints, drop.
        if 'region' in df_to_save.columns:
            try:
                # simple check: every entry must be a tuple/list of 4 ints
                col_ok = all(isinstance(x, (tuple, list)) and len(x) == 4 and all(isinstance(int(i), int) for i in x)
                             for x in df_to_save['region'])
                if not col_ok:
                    drop_if_present.append('region')
            except Exception:
                drop_if_present.append('region')

        if drop_if_present:
            cols_to_drop = [c for c in drop_if_present if c in df_to_save.columns]
            if cols_to_drop:
                df_to_save = df_to_save.drop(columns=cols_to_drop)

        df_to_save.to_pickle(filepath)
        print(f"Saved database: {len(self)} patches, {len(self.columns)} columns, {len(self._images)} images")

    @classmethod
    def load(cls, filepath: str) -> 'PatchDatabase':
        """
        Load database. Behavior:
          - Try pd.read_pickle normally.
          - If pd.read_pickle raises RecursionError (old corrupted pickle that contains RegionProperties),
            automatically rebuild the DataFrame from components/images/labels saved on disk and return it.
        """
        filepath = Path(filepath)

        try:
            df = pd.read_pickle(filepath)
        except RecursionError:
            # Corrupted pickle (likely RegionProperties stored). Rebuild from saved files silently.
            return cls.rebuild_from_saved(str(filepath))

        # load components
        components_dir = filepath.parent / (filepath.stem + '_components')
        components = []
        idx = 0
        while (components_dir / f'comp_{idx}.npz').exists():
            components.append(connectedComponent.load(str(components_dir / f'comp_{idx}.npz')))
            idx += 1

        # load metadata
        metadata_file = filepath.parent / (filepath.stem + '_metadata.pkl')
        metadata_list = pd.read_pickle(metadata_file) if metadata_file.exists() else []

        # load images
        images = []
        images_file = filepath.parent / (filepath.stem + '_images.npz')
        if images_file.exists():
            data = np.load(images_file)
            for key in data.files:
                images.append(data[key])

        # load labels
        labels = []
        labels_file = filepath.parent / (filepath.stem + '_labels.npz')
        if labels_file.exists():
            data = np.load(labels_file)
            for key in data.files:
                labels.append(data[key])

        db = cls(df)
        db._components = components
        db._metadata_list = metadata_list
        db._images = images
        db._labels = labels
        print(f"Loaded database: {len(db)} patches, {len(db.columns)} columns, {len(images)} images")
        return db

    @classmethod
    def rebuild_from_saved(cls, filepath: str) -> 'PatchDatabase':
        """
        Reconstruct the DataFrame from components/*.npz and <stem>_images.npz and <stem>_labels.npz.
        This recovers from corrupted old pickles.
        """
        filepath = Path(filepath)
        components_dir = filepath.parent / (filepath.stem + '_components')
        if not components_dir.exists():
            raise FileNotFoundError(f"Components directory not found: {components_dir}")

        # load components
        components = []
        idx = 0
        while (components_dir / f'comp_{idx}.npz').exists():
            components.append(connectedComponent.load(str(components_dir / f'comp_{idx}.npz')))
            idx += 1

        # load images
        images = []
        images_file = filepath.parent / (filepath.stem + '_images.npz')
        if images_file.exists():
            data = np.load(images_file)
            for key in data.files:
                images.append(data[key])
        else:
            raise FileNotFoundError(f"Images file not found: {images_file}")

        # load labels
        labels = []
        labels_file = filepath.parent / (filepath.stem + '_labels.npz')
        if labels_file.exists():
            data = np.load(labels_file)
            for key in data.files:
                labels.append(data[key])
        else:
            # If labels are missing we can still try to reconstruct minimal rows from components,
            # but regionprops will be called without intensity_image.
            labels = [comp.labels for comp in components]

        rows = []
        for image_idx, comp in enumerate(components):
            img = images[image_idx]
            lbl = labels[image_idx] if image_idx < len(labels) else comp.labels
            for region_idx, region in enumerate(comp.regions):
                h1, w1, h2, w2 = region.bbox
                patch = img[h1:h2, w1:w2]
                bbox = (int(h1), int(w1), int(h2), int(w2))
                rows.append({
                    'image_idx': int(image_idx),
                    'image_name': f'image_{image_idx}',
                    'region_idx': int(region_idx),
                    'region': bbox,
                    'bbox': bbox,
                    'height': int(h2 - h1),
                    'width': int(w2 - w1),
                    'area': int((h2 - h1) * (w2 - w1)),
                    'image': patch,
                })

        df = pd.DataFrame(rows)
        db = cls(df)
        db._components = components
        db._images = images
        db._labels = labels
        db._metadata_list = []
        print(f"Rebuilt database from saved files: {len(db)} patches, {len(components)} components")
        return db

    def get_component(self, image_idx: int) -> connectedComponent:
        return self._components[image_idx]

    def get_metadata(self, image_idx: int) -> Dict:
        return self._metadata_list[image_idx]

    def get_image(self, image_idx: int):
        return self._images[image_idx]

    def get_region(self, row_or_index):
        """
        Return a skimage RegionProperties object for the given row.
        Reconstructs regionprops from the stored label map and intensity image.
        """
        if isinstance(row_or_index, int):
            row = self.iloc[row_or_index]
        else:
            row = row_or_index

        # prefer labels that were explicitly saved; fallback to component's labels
        if len(self._labels) > int(row.image_idx):
            labels = self._labels[int(row.image_idx)]
        else:
            labels = self._components[int(row.image_idx)].labels

        img = self._images[int(row.image_idx)] if len(self._images) > int(row.image_idx) else None
        props = regionprops(labels, intensity_image=img)
        return props[int(row.region_idx)]

    def __repr__(self):
        return f"PatchDatabase({len(self)} patches, {len(self.columns)} columns)"
