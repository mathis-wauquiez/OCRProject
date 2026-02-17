import sys
import os
from pathlib import Path

os.environ["HYDRA_FULL_ERROR"] = "1"

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hydra.utils import instantiate

from operator import itemgetter
from PIL import Image

from notebook_utils.parquet_utils import save_dataframe
from notebook_utils.viz import show_random_sample
from notebook_utils.descriptor import compute_hog, visualize_hog

from src.utils import connectedComponent

from .ink_filter import InkFilter
from .hog import HOG
from .params import HOGParameters
from .svg import SVG, rotation
from .patch_extraction import extract_patches
from ..layout_analysis.skew import get_document_orientation

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import contextmanager
from queue import Queue
from threading import Thread
from dataclasses import dataclass, field

import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# type-hinting
from src.vectorization.wrapper import BinaryShapeVectorizer
# from src.ocr.wrappers import OCRModel          # old Qwen/wrapper-based OCR — disabled
# from src.patch_processing.renderer import Renderer  # only needed for old OCR renderer
from src.patch_processing.renderer import Renderer   # still used for HOG renderer
from typing import List, Optional

from src.layout_analysis.parsing import ReadingOrder, split_to_rectangles, break_into_subcols
from src.ocr.chat import ModelWrapper


# ------------------------------------------------------------------ #
#  Pipeline data transfer                                              #
# ------------------------------------------------------------------ #

@dataclass
class PagePayload:
    """Data handed off from the CPU stage to the GPU stage."""
    page_df: pd.DataFrame
    svg_imgs: list
    # Raw grayscale page image (H, W) uint8 — used by the CHAT model
    image: np.ndarray = field(default=None)
    # Subcolumn layout returned by split_to_rectangles — used for CHAT grouping
    rectangles: pd.DataFrame = field(default=None)


SENTINEL = None  # signals end of the producer stream


# ------------------------------------------------------------------ #
#  Main class                                                          #
# ------------------------------------------------------------------ #

class PatchPreprocessing:

    def __init__(self,
                 reading_order: ReadingOrder,
                 ink_filter: InkFilter,
                 vectorizer: BinaryShapeVectorizer,
                 chat_model: ModelWrapper,
                 # ocr_model_configs: List[dict],   # old Qwen/wrapper approach — disabled
                 # ocr_renderer: Renderer,           # old SVG renderer for OCR — disabled
                 hog_renderer: Renderer,
                 hog_params: HOGParameters,
                 output_viz: None | Path = None,
                 verbose=True):

        self.ink_filter = ink_filter
        self.vectorizer = vectorizer
        self.verbose = verbose
        self.output_viz = Path(output_viz) if output_viz is not None else None
        self.chat_model = chat_model
        # self.ocr_model_configs = ocr_model_configs  # disabled
        # self.ocr_renderer = ocr_renderer              # disabled
        self.hog_renderer = hog_renderer
        self.hog = HOG(hog_params)
        self.reading_order = reading_order

    def _print(self, *args, **kwargs):
        if self.verbose:
            return print(*args, **kwargs)

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def __call__(self, image_folder, comps_folder):
        files = sorted(next(os.walk(image_folder))[2])
        queue: Queue[Optional[PagePayload]] = Queue(maxsize=2)

        # CHAT model is instantiated once and reused across all pages
        # (no context manager needed — lifecycle managed by the caller / Hydra)

        # --- Producer thread (CPU-bound work) ---
        def cpu_worker():
            for page_idx, file in enumerate(files):
                payload = self._cpu_stage(
                    page_idx, file, image_folder, comps_folder
                )
                queue.put(payload)
            queue.put(SENTINEL)

        producer = Thread(target=cpu_worker, daemon=True)
        producer.start()

        # --- Consumer (GPU work, main thread) ---
        page_dataframes = []
        pbar = tqdm(total=len(files), desc="Pages")

        while True:
            payload = queue.get()
            if payload is SENTINEL:
                break
            self._gpu_stage(payload)
            page_dataframes.append(payload.page_df)
            pbar.update(1)

        producer.join()
        pbar.close()

        # Concatenate and sort globally
        result = pd.concat(page_dataframes, ignore_index=True)
        result.sort_values(
            by=['page', 'reading_order'], inplace=True, na_position='last'
        )
        result.reset_index(drop=True, inplace=True)
        return result

    # ------------------------------------------------------------------ #
    #  CPU stage: extract → filter → vectorize → deskew                    #
    # ------------------------------------------------------------------ #

    def _cpu_stage(self, page_idx, file, image_folder, comps_folder
                   ) -> PagePayload:
        """All CPU-bound work for a single page."""
        page_df, image, rectangles = self._extract_page(
            page_idx, file, image_folder, comps_folder
        )

        # Ink filter
        self._print(f'  [{file}] Applying ink filter')
        ink_filtered = self.ink_filter(page_df['bin_patch'])
        ink_filtered = [patch < .5 for patch in ink_filtered]

        # Vectorize
        vectorization_output = sorted(
            list(self.vectorizer(ink_filtered)), key=itemgetter(0)
        )
        svg_imgs = [svg for _, svg in vectorization_output]
        page_df['svg'] = svg_imgs
        del ink_filtered

        # Deskew SVGs
        page_skew = page_df['page_skew'].iloc[0]
        for svg in svg_imgs:
            svg.apply_homography(rotation(-page_skew))

        return PagePayload(
            page_df=page_df,
            svg_imgs=svg_imgs,
            image=image,
            rectangles=rectangles,
        )

    # ------------------------------------------------------------------ #
    #  GPU stage: OCR → HOG                                                #
    # ------------------------------------------------------------------ #

    def _gpu_stage(self, payload: PagePayload):
        """All GPU-bound work for a single page."""
        page_df = payload.page_df
        svg_imgs = payload.svg_imgs
        image = payload.image          # (H, W) uint8 grayscale
        rectangles = payload.rectangles
        file = page_df['file'].iloc[0]

        # ---- CHAT OCR (replaces old Qwen / wrapper renderer approach) ----
        #
        # Old approach (disabled):
        #   ocr_rendered = self.ocr_renderer(svg_imgs)
        #   for ocr_model in ocr_models:
        #       chars, uncertainties = ocr_model.predict_with_scores(ocr_rendered)
        #       page_df[f'unc_{ocr_model.name}'] = uncertainties
        #       page_df[f'char_{ocr_model.name}'] = chars
        #
        # New approach:
        #   Feed each subcolumn's *raw image crop* plus the per-character
        #   bounding-boxes and CRAFT barycenters (centroids from page_df)
        #   directly to the CHAT ModelWrapper.

        self._print(f'  [{file}] Running CHAT OCR')
        self._run_chat_ocr(page_df, image, rectangles)

        # HOG
        self._print(f'  [{file}] Computing HOG')
        self._compute_hog(page_df, svg_imgs)

    # ------------------------------------------------------------------ #
    #  CHAT OCR (one page)                                                 #
    # ------------------------------------------------------------------ #

    def _run_chat_ocr(
        self,
        page_df: pd.DataFrame,
        image: np.ndarray,
        rectangles: pd.DataFrame,
    ):
        """Feed each subcolumn image + CRAFT barycenters to the CHAT model.

        The subcolumn structure comes from `rectangles` (split_to_rectangles).
        For every subcolumn we:
          1. Crop the grayscale page image to the union of character bboxes.
          2. Build per-character bbox and centroid lists (relative to the crop).
          3. Call `self.chat_model.predict(image_crop, bboxes, centers)`.
          4. Write `char_chat` / `conf_chat` columns back into page_df.

        Centroids (barycenters) are taken from `page_df['centroid']`, which
        stores the CRAFT-component centroids as (cy, cx) tuples.
        """
        if rectangles is None or len(rectangles) == 0:
            return

        # Pre-build a label → DataFrame-row-index lookup for O(1) access
        label_to_idx: dict[int, int] = {
            int(row['label']): i for i, row in page_df.iterrows()
        }

        # Initialise output columns
        page_df['char_chat'] = None
        page_df['conf_chat'] = None

        for col_idx, col_rects in rectangles.groupby('col_idx'):
            subcolumns = break_into_subcols(col_rects)

            for subcolumn in subcolumns:
                n_subcols = len(subcolumn['labels'].iloc[0])

                for subcol_idx in range(n_subcols):
                    # Collect character labels for this subcolumn, in reading order
                    subcol_labels = []
                    for _, row in subcolumn.iterrows():
                        labels_in_row = row['labels']
                        if subcol_idx < len(labels_in_row):
                            lbl = int(labels_in_row[subcol_idx])
                            if lbl != 0 and lbl in label_to_idx:
                                subcol_labels.append(lbl)

                    if not subcol_labels:
                        continue

                    # DataFrame rows for this subcolumn
                    row_indices = [label_to_idx[l] for l in subcol_labels]
                    subcol_df = page_df.loc[row_indices]

                    # Subcolumn bounding box (pixel coords in full page)
                    x0 = int(subcol_df['left'].min())
                    y0 = int(subcol_df['top'].min())
                    x1 = int((subcol_df['left'] + subcol_df['width']).max())
                    y1 = int((subcol_df['top'] + subcol_df['height']).max())

                    # Crop grayscale image to subcolumn
                    subcol_image = image[y0:y1, x0:x1]
                    if subcol_image.size == 0:
                        continue

                    # Per-character bboxes relative to the crop: (x0,y0,x1,y1)
                    bboxes = [
                        (
                            int(r['left']) - x0,
                            int(r['top']) - y0,
                            int(r['left']) - x0 + int(r['width']),
                            int(r['top']) - y0 + int(r['height']),
                        )
                        for _, r in subcol_df.iterrows()
                    ]

                    # Per-character CRAFT barycenters relative to crop: (cy, cx)
                    # centroid stored as (cy, cx) in page_df
                    centers = [
                        (
                            float(r['centroid'][0]) - y0,
                            float(r['centroid'][1]) - x0,
                        )
                        for _, r in subcol_df.iterrows()
                    ]

                    # Run CHAT model on this subcolumn
                    result = self.chat_model.predict(subcol_image, bboxes, centers)

                    # Write results back into page_df
                    for lbl, char, conf in zip(
                        subcol_labels, result.char, result.confidence
                    ):
                        idx = label_to_idx[lbl]
                        page_df.at[idx, 'char_chat'] = char
                        page_df.at[idx, 'conf_chat'] = conf

    # ------------------------------------------------------------------ #
    #  Extraction (one page)                                               #
    # ------------------------------------------------------------------ #

    def _extract_page(self, page_idx, file, image_folder, comps_folder
                      ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Load components, extract patches, apply reading order.

        Returns
        -------
        page_df : pd.DataFrame
        image : np.ndarray  shape (H, W) uint8 — grayscale page image
        rectangles : pd.DataFrame  — subcolumn layout from split_to_rectangles
        """
        self._print(f'  [{file}] Extracting patches')

        img_np = np.array(Image.open(image_folder / file))
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)[..., None]

        img_comp = connectedComponent.load(
            comps_folder / 'components' / (str(file) + '.npz')
        )
        img_comp._stats = img_comp._compute_stats_from_labels(img_comp._labels)

        craft_comp = connectedComponent.load(
            comps_folder / 'craft_components' / (str(file) + '.npz')
        )

        _bin_patches, _img_patches = extract_patches(
            characterComponents=img_comp,
            images=[img_np],
            return_bin=True
        )

        lbls = [r.label for r in img_comp.regions
                 if not img_comp.is_deleted(r.label)]

        centroids = [r.centroid for r in img_comp.regions
                     if not img_comp.is_deleted(r.label)]

        stats = img_comp.stats[1:]

        page_skew = get_document_orientation(img_np)

        page_df = pd.DataFrame({
            'bin_patch': _bin_patches,
            'img_patch': _img_patches,
            'page': page_idx,
            'file': file,
            'left': stats[:, 0],
            'top': stats[:, 1],
            'width': stats[:, 2],
            'height': stats[:, 3],
            'label': lbls,
            'centroid': centroids,
            'page_skew': page_skew
        })

        # Reading order + optional visualisation
        # split_to_rectangles is called here (same params as inside ReadingOrder)
        # so we can reuse the subcolumn structure in the CHAT OCR stage.
        rectangles = split_to_rectangles(
            craft_comp.labels,
            min_col_area=self.reading_order.min_col_area,
        )

        if self.output_viz is not None:
            canvas1 = craft_comp.segm_img
            canvas2 = np.array(Image.open(image_folder / file))
            fig = self.reading_order(craft_comp.labels, page_df,
                                     canvas1, canvas2)
            self._save_viz(fig, canvas1, canvas2, file)
        else:
            self.reading_order(craft_comp.labels, page_df)

        # Return (H, W) uint8 image — squeeze the trailing channel added above
        image_hw = img_np[:, :, 0]

        return page_df, image_hw, rectangles

    # ------------------------------------------------------------------ #
    #  HOG (one page)                                                      #
    # ------------------------------------------------------------------ #

    def _compute_hog(self, page_df: pd.DataFrame, svg_imgs):
        hog_device = self.hog._params.device

        hog_renderer = self.hog_renderer(svg_imgs)
        dataloader = DataLoader(
            hog_renderer,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        # Preallocate
        first_batch = next(iter(dataloader))
        sample_out = self.hog(
            first_batch[:1].unsqueeze(1).to(
                dtype=torch.float32, device=hog_device
            )
        )
        hist_shape = sample_out.histograms[0, 0].shape
        histograms = torch.zeros(
            (len(svg_imgs), *hist_shape), device=hog_device
        )

        start_idx = 0
        for batch in dataloader:
            hog_out = self.hog(
                batch.unsqueeze(1).to(
                    dtype=torch.float32, device=hog_device
                )
            )
            h = hog_out.histograms[:, 0]
            histograms[start_idx:start_idx + h.shape[0]] = h
            start_idx += h.shape[0]

        page_df['histogram'] = list(histograms.cpu().numpy())

    # ------------------------------------------------------------------ #
    #  OCR model lifecycle (old Qwen / wrapper approach — disabled)       #
    # ------------------------------------------------------------------ #

    # @contextmanager
    # def _load_all_ocr_models(self):
    #     """Load all OCR models for the duration of the run, then clean up."""
    #     models = []
    #     try:
    #         for ocr_partial in self.ocr_model_configs:
    #             model = ocr_partial()
    #             self._print(f"Loaded OCR model: {model.name}")
    #             models.append(model)
    #         yield models
    #     finally:
    #         for model in models:
    #             self._print(f"Unloading {model.name}")
    #             del model
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    #  Visualisation helpers                                               #
    # ------------------------------------------------------------------ #

    def _save_viz(self, fig, canvas1, canvas2, file):
        for canvas, subfolder in [(canvas1, "craft_reading_order"),
                                  (canvas2, "reading_order")]:
            folder = self.output_viz / subfolder
            folder.mkdir(exist_ok=True, parents=True)
            Image.fromarray(canvas).save(folder / file, quality=100)

        folder = self.output_viz / "plt_reading_order"
        folder.mkdir(exist_ok=True)
        fig.savefig(folder / file)