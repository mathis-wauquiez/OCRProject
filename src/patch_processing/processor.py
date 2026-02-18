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
from queue import Queue
from threading import Thread
from dataclasses import dataclass, field

import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from src.vectorization.wrapper import BinaryShapeVectorizer
from src.patch_processing.renderer import Renderer
from src.layout_analysis.parsing import ReadingOrder, split_to_rectangles, break_into_subcols
from src.ocr.chat import ModelWrapper
from src.auto_report import AutoReport

from kraken import rpred as kraken_rpred
from kraken.containers import Segmentation, BaselineLine

UNKNOWN_CHAR = "▯"


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
                 hog_renderer: Renderer,
                 hog_params: HOGParameters,
                 output_viz: None | Path = None,
                 verbose=True,
                 viz_report: AutoReport | None = None,
                 max_viz_per_page: int = 5):

        self.ink_filter = ink_filter
        self.vectorizer = vectorizer
        self.verbose = verbose
        self.output_viz = Path(output_viz) if output_viz is not None else None
        self.chat_model = chat_model
        self.hog_renderer = hog_renderer
        self.hog = HOG(hog_params)
        self.reading_order = reading_order
        self.viz_report = viz_report
        self.max_viz_per_page = max_viz_per_page
        self._viz_counter = 0  # Counter for limiting visualizations

    def _print(self, *args, **kwargs):
        if self.verbose:
            return print(*args, **kwargs)

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def __call__(self, image_folder, comps_folder):
        files = sorted(next(os.walk(image_folder))[2])
        queue: Queue[PagePayload | None] = Queue(maxsize=2)

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
        file = payload.page_df['file'].iloc[0]

        self._print(f'  [{file}] Running CHAT OCR')
        self._run_chat_ocr(payload.page_df, payload.image, payload.rectangles)

        self._print(f'  [{file}] Computing HOG')
        self._compute_hog(payload.page_df, payload.svg_imgs)

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

        For every subcolumn (from ``split_to_rectangles``):
          1. Crop the grayscale page image to the union bbox of its characters.
          2. Binarize the crop (PIL mode ``"1"``).
          3. Build a synthetic baseline from CRAFT centroids.
          4. Run ``kraken.rpred.rpred()`` for proper line extraction + recognition.
          5. Write ``char_chat`` / ``conf_chat`` back into *page_df*.
        """
        if rectangles is None or len(rectangles) == 0:
            return

        # Reset visualization counter for this page
        self._viz_counter = 0

        # label → DataFrame-row-index for O(1) write-back
        label_to_idx: dict[int, int] = {
            int(row['label']): i for i, row in page_df.iterrows()
        }

        page_df['char_chat'] = None
        page_df['conf_chat'] = None

        for _, col_rects in rectangles.groupby('col_idx'):
            for subcolumn in break_into_subcols(col_rects):
                n_subcols = len(subcolumn['labels'].iloc[0])

                for subcol_idx in range(n_subcols):
                    self._predict_subcol(
                        page_df, image, subcolumn, subcol_idx, label_to_idx,
                    )

    def _predict_subcol(
        self,
        page_df: pd.DataFrame,
        image: np.ndarray,
        subcolumn: pd.DataFrame,
        subcol_idx: int,
        label_to_idx: dict[int, int],
    ):
        """Run the CHAT model on a single vertical subcolumn lane.

        Instead of calling ``ModelWrapper.predict()`` directly (which feeds
        the raw image to ``net.predict()`` and gets 0 detections), we use
        ``kraken.rpred.rpred()`` — the proper Kraken recognition pipeline
        that handles line extraction, dewarping, and inference.

        A synthetic baseline is built from the CRAFT barycenters so that
        no separate segmentation model is needed.
        """
        # Collect labels present in page_df for this lane
        subcol_labels = []
        for _, row in subcolumn.iterrows():
            labels_in_row = row['labels']
            if subcol_idx < len(labels_in_row):
                lbl = int(labels_in_row[subcol_idx])
                if lbl != 0 and lbl in label_to_idx:
                    subcol_labels.append(lbl)

        if not subcol_labels:
            return

        subcol_df = page_df.loc[[label_to_idx[l] for l in subcol_labels]]

        # Tight bounding box around all characters (full-page coords)
        x0 = int(subcol_df['left'].min())
        y0 = int(subcol_df['top'].min())
        x1 = int((subcol_df['left'] + subcol_df['width']).max())
        y1 = int((subcol_df['top'] + subcol_df['height']).max())

        subcol_image = image[y0:y1, x0:x1]
        if subcol_image.size == 0:
            return
        h, w = subcol_image.shape[:2]

        # Binarize → PIL mode "1" (as expected by the CHAT model — see demo)
        pil_img = Image.fromarray(subcol_image).point(
            lambda x: 0 if x < 128 else 255, "1"
        )

        # CRAFT barycenters relative to crop: (cy, cx)
        centers_rel = []
        for _, r in subcol_df.iterrows():
            cy, cx = r['centroid']
            centers_rel.append((float(cy) - y0, float(cx) - x0))

        # Synthetic baseline: straight vertical line at the median x-coord,
        # spanning the full crop height.  This tells rpred where the text
        # column runs without needing a separate segmentation model.
        cx_median = float(np.median([cx for _, cx in centers_rel]))
        baseline = [(int(round(cx_median)), 0), (int(round(cx_median)), h)]
        boundary = [(0, 0), (w, 0), (w, h), (0, h)]

        line = BaselineLine(id="0", baseline=baseline, boundary=boundary)
        seg = Segmentation(
            type="baselines",
            imagename="",
            text_direction="vertical-rl",
            script_detection=False,
            lines=[line],
        )

        # Run Kraken recognition via rpred (line extraction + dewarping + inference)
        pred_it = kraken_rpred.rpred(
            self.chat_model.net, pil_img, seg, pad=self.chat_model.pad,
        )

        pred_chars, pred_confs = [], []
        for record in pred_it:
            pred_chars = list(record._prediction)
            pred_confs = (
                list(record._confidences)
                if record._confidences else []
            )

        # Ensure pred_confs is always aligned with pred_chars.
        # When the model returns no confidences, default to 1.0 so the
        # zip in the write-back loop is never cut short.
        if len(pred_confs) < len(pred_chars):
            pred_confs += [1.0] * (len(pred_chars) - len(pred_confs))

        # Align predictions to CRAFT centroids
        M = len(subcol_labels)
        N = len(pred_chars)
        if N > M:
            pred_chars = pred_chars[:M]
            pred_confs = pred_confs[:M]
        elif N < M:
            pred_chars += [UNKNOWN_CHAR] * (M - N)
            pred_confs += [0.0] * (M - N)

        # Optional visualization
        if self.viz_report is not None:
            page_name = page_df['file'].iloc[0] if 'file' in page_df.columns else 'page'
            self._visualize_subcol_pipeline(
                subcol_image, pil_img, baseline, boundary, centers_rel,
                pred_chars, pred_confs, page_name
            )

        for lbl, char, conf in zip(subcol_labels, pred_chars, pred_confs):
            idx = label_to_idx[lbl]
            page_df.at[idx, 'char_chat'] = char
            page_df.at[idx, 'conf_chat'] = conf

    def _visualize_subcol_pipeline(
        self,
        subcol_image: np.ndarray,
        pil_img: Image.Image,
        baseline: list,
        boundary: list,
        centers_rel: list,
        pred_chars: list,
        pred_confs: list,
        page_name: str,
    ):
        """Create visualizations showing the CHAT preprocessing pipeline steps.

        Generates 3 subplots:
          1. Grayscale crop (raw subcolumn extraction)
          2. Binarized image (PIL mode "1" after threshold 128)
          3. Predictions overlaid with baseline + confidence colors
        """
        # Limit visualizations per page to keep reports manageable
        self._viz_counter += 1
        if self._viz_counter > self.max_viz_per_page:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        fig.suptitle(f'CHAT Pipeline: {page_name} (subcolumn {self._viz_counter})',
                     fontsize=14, fontweight='bold')

        # 1. Grayscale crop
        axes[0].imshow(subcol_image, cmap='gray')
        axes[0].set_title('1. Grayscale Crop', fontsize=12)
        axes[0].axis('off')

        # Plot CRAFT centroids
        if centers_rel:
            cy_vals, cx_vals = zip(*centers_rel)
            axes[0].scatter(cx_vals, cy_vals, c='red', s=20, marker='x',
                          alpha=0.7, label='CRAFT centroids')
            axes[0].legend(fontsize=8)

        # 2. Binarized image
        # Convert PIL "1" mode back to numpy for display
        bin_array = np.array(pil_img)
        axes[1].imshow(bin_array, cmap='gray')
        axes[1].set_title('2. Binarized (threshold=128)', fontsize=12)
        axes[1].axis('off')

        # 3. Predictions + baseline
        axes[2].imshow(bin_array, cmap='gray')
        axes[2].set_title('3. Baseline + Predictions', fontsize=12)
        axes[2].axis('off')

        # Draw synthetic baseline (vertical line)
        if baseline and len(baseline) >= 2:
            bl_x, bl_y = zip(*baseline)
            axes[2].plot(bl_x, bl_y, 'b-', linewidth=2, label='Baseline', alpha=0.8)

        # Draw boundary
        if boundary and len(boundary) >= 3:
            bd_x, bd_y = zip(*boundary)
            bd_x = bd_x + (bd_x[0],)  # close polygon
            bd_y = bd_y + (bd_y[0],)
            axes[2].plot(bd_x, bd_y, 'g--', linewidth=1, label='Boundary', alpha=0.5)

        # Draw predictions with confidence-based colors
        if pred_chars and centers_rel:
            for (cy, cx), char, conf in zip(centers_rel, pred_chars, pred_confs):
                # Color: green (high conf) → yellow (medium) → red (low)
                if conf >= 0.8:
                    color = 'green'
                elif conf >= 0.5:
                    color = 'orange'
                else:
                    color = 'red'

                axes[2].text(cx, cy, char, fontsize=10, color=color,
                           ha='center', va='center', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white', alpha=0.7, edgecolor=color))

        axes[2].legend(fontsize=8, loc='upper right')

        plt.tight_layout()
        self.viz_report.report_figure(fig, title=f'Subcolumn {self._viz_counter} - {page_name}')

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