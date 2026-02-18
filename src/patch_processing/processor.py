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
from scipy.optimize import linear_sum_assignment


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


@dataclass
class _SubcolVizData:
    """Visualization payload collected by _predict_subcol for deferred rendering."""
    subcol_image: np.ndarray          # grayscale crop (H, W) uint8
    pil_img: Image.Image              # binarized PIL image (mode "1")
    centers_rel: list                 # [(cy, cx), …] CRAFT centroids in crop coords
    char_boxes_rel: list              # [(left, top, w, h), …] CRAFT bboxes in crop
    pred_chars: list                  # matched prediction per CRAFT centroid
    pred_confs: list                  # matched confidence per CRAFT centroid
    # Raw Kraken output + spatial info from cuts
    cut_polys: list = field(default_factory=list)   # 4-point quads in crop coords
    raw_pred_chars: list = field(default_factory=list)
    raw_pred_confs: list = field(default_factory=list)


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

        # label → DataFrame-row-index for O(1) write-back
        label_to_idx: dict[int, int] = {
            int(row['label']): i for i, row in page_df.iterrows()
        }

        page_df['char_chat'] = None
        page_df['conf_chat'] = None

        # Collect viz data per column for deferred, structured report rendering
        page_viz: dict[int, list[_SubcolVizData]] = {}

        for col_idx, col_rects in rectangles.groupby('col_idx'):
            col_subcols: list[_SubcolVizData] = []
            for subcolumn in break_into_subcols(col_rects):
                n_subcols = len(subcolumn['labels'].iloc[0])
                for subcol_idx in range(n_subcols):
                    viz = self._predict_subcol(
                        page_df, image, subcolumn, subcol_idx, label_to_idx,
                    )
                    if viz is not None:
                        col_subcols.append(viz)
            if col_subcols:
                page_viz[int(col_idx)] = col_subcols

        if self.viz_report is not None and page_viz:
            page_name = page_df['file'].iloc[0] if 'file' in page_df.columns else 'page'
            self._add_page_to_report(page_name, page_viz)

    def _predict_subcol(
        self,
        page_df: pd.DataFrame,
        image: np.ndarray,
        subcolumn: pd.DataFrame,
        subcol_idx: int,
        label_to_idx: dict[int, int],
    ) -> '_SubcolVizData | None':
        """Run the CHAT model on a single vertical subcolumn lane.

        Instead of calling ``ModelWrapper.predict()`` directly (which feeds
        the raw image to ``net.predict()`` and gets 0 detections), we use
        ``kraken.rpred.rpred()`` — the proper Kraken recognition pipeline
        that handles line extraction, dewarping, and inference.

        A synthetic baseline is built from the CRAFT barycenters so that
        no separate segmentation model is needed.

        Returns a ``_SubcolVizData`` when ``self.viz_report`` is set,
        otherwise returns ``None``.
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
            return None

        subcol_df = page_df.loc[[label_to_idx[l] for l in subcol_labels]]

        # Tight bounding box around all characters (full-page coords)
        x0 = int(subcol_df['left'].min())
        y0 = int(subcol_df['top'].min())
        x1 = int((subcol_df['left'] + subcol_df['width']).max())
        y1 = int((subcol_df['top'] + subcol_df['height']).max())

        subcol_image = image[y0:y1, x0:x1]
        if subcol_image.size == 0:
            return None
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

        # Character bounding boxes relative to crop: (left, top, width, height)
        char_boxes_rel = [
            (int(r['left']) - x0, int(r['top']) - y0,
             int(r['width']), int(r['height']))
            for _, r in subcol_df.iterrows()
        ]

        # Synthetic baseline: straight vertical line at the median x-coord,
        # spanning the full crop height.  This tells rpred where the text
        # column runs without needing a separate segmentation model.
        cx_median = float(np.median([cx for _, cx in centers_rel]))
        baseline = [(int(round(cx_median)), 0), (int(round(cx_median)), h - 1)]
        boundary = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]

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

        raw_pred_chars, raw_pred_confs, cut_polys = [], [], []
        for record in pred_it:
            raw_pred_chars = list(record._prediction)
            raw_pred_confs = (
                list(record._confidences)
                if record._confidences else []
            )
            # record.cuts is a property returning 4-point polygons in the
            # same coordinate system as the baseline/boundary (= crop coords).
            try:
                cut_polys = [list(poly) for poly in record.cuts]
            except Exception:
                cut_polys = []

        # Ensure confs aligned with chars
        if len(raw_pred_confs) < len(raw_pred_chars):
            raw_pred_confs += [1.0] * (len(raw_pred_chars) - len(raw_pred_confs))

        # --- Spatial matching via Kraken cuts --------------------------------
        # Each cut_polys[j] is a 4-point polygon [[x,y], …] in crop coords.
        # Compute its center-y and match to the nearest CRAFT centroid-y.
        M = len(subcol_labels)
        N = len(raw_pred_chars)

        matched_chars = [UNKNOWN_CHAR] * M
        matched_confs = [0.0] * M

        if cut_polys and len(cut_polys) == N and N > 0 and M > 0:
            # Center-y of each Kraken cut polygon
            cut_cy = np.array([
                float(np.mean([pt[1] for pt in poly]))
                for poly in cut_polys
            ])
            # CRAFT centroid y-positions
            craft_cy = np.array([cy for cy, _cx in centers_rel])

            # Cost matrix: |craft_cy[i] - cut_cy[j]|
            cost = np.abs(craft_cy[:, None] - cut_cy[None, :])

            # Reject matches farther than 1.5× average character height
            avg_h = float(np.mean([bh for _, _, _, bh in char_boxes_rel])) \
                if char_boxes_rel else 50.0
            max_dist = avg_h * 1.5

            row_ind, col_ind = linear_sum_assignment(cost)
            for ci, pi in zip(row_ind, col_ind):
                if cost[ci, pi] <= max_dist:
                    matched_chars[ci] = raw_pred_chars[pi]
                    matched_confs[ci] = raw_pred_confs[pi]
        else:
            # Fallback: naive positional alignment (no cuts available)
            for i in range(min(M, N)):
                matched_chars[i] = raw_pred_chars[i]
                matched_confs[i] = raw_pred_confs[i]

        # Write back to page_df
        for lbl, char, conf in zip(subcol_labels, matched_chars, matched_confs):
            idx = label_to_idx[lbl]
            page_df.at[idx, 'char_chat'] = char
            page_df.at[idx, 'conf_chat'] = conf

        if self.viz_report is not None:
            return _SubcolVizData(
                subcol_image=subcol_image,
                pil_img=pil_img,
                centers_rel=centers_rel,
                char_boxes_rel=char_boxes_rel,
                pred_chars=matched_chars,
                pred_confs=matched_confs,
                cut_polys=cut_polys,
                raw_pred_chars=raw_pred_chars,
                raw_pred_confs=raw_pred_confs,
            )
        return None

    def _add_page_to_report(
        self,
        page_name: str,
        page_viz: dict,
    ):
        """Add one report section per page, with one figure per column.

        The section title is the page filename.  Inside, each column gets a
        titled figure so they appear as visual subsections.  Each figure
        contains one row per subcolumn (up to ``max_viz_per_page`` rows),
        with two panels side-by-side:

          - **Left**: clean binarized crop (no overlay)
          - **Right**: white panel with predicted characters rendered at the
            same spatial positions and approximate sizes as the originals
        """
        with self.viz_report.section(f'Page: {page_name}'):
            for col_idx, subcols in sorted(page_viz.items()):
                fig = self._make_col_figure(col_idx, subcols, page_name)
                n = min(len(subcols), self.max_viz_per_page)
                self.viz_report.report_figure(
                    fig,
                    title=f'Column {col_idx} ({n} subcolumn(s) shown)',
                )

    def _make_col_figure(
        self,
        col_idx: int,
        subcols: list,
        page_name: str,
    ) -> plt.Figure:
        """Build a figure showing all subcolumns in one column side-by-side.

        Each row corresponds to one subcolumn.  Row heights are proportional
        to the subcolumn crop height so the left and right panels are always
        the same scale.
        """
        subcols = subcols[: self.max_viz_per_page]
        n = len(subcols)

        # Row heights proportional to crop height (minimum 20 px)
        row_heights = [max(d.subcol_image.shape[0], 20) for d in subcols]
        total_px = sum(row_heights)
        # ~50 px per inch; at least 3 in tall so titles are readable
        fig_h = max(3.0, total_px / 50.0)

        gridspec_kw = {'height_ratios': row_heights} if n > 1 else {}
        fig, axes = plt.subplots(
            n, 2,
            figsize=(11, fig_h),
            gridspec_kw=gridspec_kw,
            squeeze=False,
        )
        fig.suptitle(
            f'{page_name} — Column {col_idx}',
            fontsize=11, fontweight='bold',
        )

        for i, data in enumerate(subcols):
            bin_arr = np.array(data.pil_img)
            h_img, w_img = bin_arr.shape[:2]

            # ---- Left: binarized image + cut boundaries -----------------
            axes[i][0].imshow(bin_arr, cmap='gray')
            # Overlay cut polygon boundaries as horizontal dashed lines
            for poly in data.cut_polys:
                if poly and len(poly) >= 4:
                    ys = [pt[1] for pt in poly]
                    y_min, y_max = min(ys), max(ys)
                    axes[i][0].axhline(
                        y_min, color='cyan', linewidth=0.6,
                        linestyle='--', alpha=0.7,
                    )
                    axes[i][0].axhline(
                        y_max, color='cyan', linewidth=0.6,
                        linestyle='--', alpha=0.7,
                    )
            n_kraken = len(data.raw_pred_chars)
            n_craft = len(data.centers_rel)
            axes[i][0].set_title(
                f'Subcol {i + 1}  (CRAFT {n_craft}, Kraken {n_kraken})',
                fontsize=8,
            )
            axes[i][0].axis('off')

            # ---- Right: predicted characters panel ----------------------
            self._draw_prediction_panel(
                axes[i][1], data,
            )
            matched_known = [c for c in data.pred_chars if c != UNKNOWN_CHAR]
            chars_preview = ''.join(matched_known[:12])
            if len(matched_known) > 12:
                chars_preview += '…'
            avg_conf = float(np.mean([
                c for c in data.pred_confs if c > 0
            ])) if any(c > 0 for c in data.pred_confs) else 0.0
            axes[i][1].set_title(
                f'"{chars_preview}"  avg conf {avg_conf:.0%}',
                fontsize=8,
            )

        plt.tight_layout()
        return fig

    def _draw_prediction_panel(
        self,
        ax: plt.Axes,
        data: '_SubcolVizData',
    ):
        """Fill *ax* with predicted characters on a white background.

        When Kraken cuts are available, each raw prediction is drawn at the
        y-center of its cut polygon so the spatial alignment between CRAFT
        centroids and Kraken detections is visible.  CRAFT bounding boxes
        are shown as light-blue rectangles and a thin grey line connects
        each matched prediction to the CRAFT centroid it was assigned to.

        When no cuts are available, predictions are drawn at the CRAFT
        centroid positions (fallback).
        """
        h, w = data.subcol_image.shape[:2]

        ax.set_facecolor('white')
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # image convention: y increases downward
        ax.axis('off')

        avg_bh = (float(np.mean([bh for _, _, _, bh in data.char_boxes_rel]))
                  if data.char_boxes_rel
                  else h / max(len(data.raw_pred_chars), 1))

        # Draw light CRAFT bounding boxes for reference
        for (bx, by, bw, bh) in data.char_boxes_rel:
            rect = plt.Rectangle(
                (bx, by), bw, bh,
                fill=False, edgecolor='lightsteelblue', linewidth=0.6,
            )
            ax.add_patch(rect)

        # Small CRAFT centroid markers
        for (cy, cx) in data.centers_rel:
            ax.plot(cx, cy, 'x', color='lightsteelblue', markersize=4,
                    markeredgewidth=0.8)

        mid_x = w / 2.0

        if data.cut_polys and len(data.cut_polys) == len(data.raw_pred_chars):
            # --- Draw raw predictions at their cut y-centers -------------
            # Precompute a CRAFT-centroid lookup for matched predictions
            # (matched_chars[i] != UNKNOWN_CHAR ↔ CRAFT idx i was matched)
            # Build reverse map: which raw pred idx → which CRAFT idx
            craft_cy = np.array([cy for cy, _cx in data.centers_rel])
            cut_cy = np.array([
                float(np.mean([pt[1] for pt in poly]))
                for poly in data.cut_polys
            ])
            # Re-derive the match for drawing connecting lines
            cost = np.abs(craft_cy[:, None] - cut_cy[None, :])
            avg_h = float(np.mean([
                bh for _, _, _, bh in data.char_boxes_rel
            ])) if data.char_boxes_rel else 50.0
            max_dist = avg_h * 1.5
            row_ind, col_ind = linear_sum_assignment(cost)
            match_pred_to_craft = {}
            for ci, pi in zip(row_ind, col_ind):
                if cost[ci, pi] <= max_dist:
                    match_pred_to_craft[pi] = ci

            for j, (char, conf, poly) in enumerate(zip(
                data.raw_pred_chars, data.raw_pred_confs, data.cut_polys
            )):
                cy_cut = float(np.mean([pt[1] for pt in poly]))

                # Confidence color
                if conf >= 0.8:
                    color = 'darkgreen'
                elif conf >= 0.5:
                    color = 'darkorange'
                else:
                    color = 'red'

                # Font size: use cut polygon height when meaningful,
                # otherwise fall back to average CRAFT bbox height
                ys = [pt[1] for pt in poly]
                cut_h = max(ys) - min(ys)
                effective_h = cut_h if cut_h > avg_bh * 0.4 else avg_bh
                fontsize = max(8, min(effective_h * 0.65, 36))

                ax.text(
                    mid_x, cy_cut, char,
                    ha='center', va='center',
                    color=color, fontsize=fontsize, fontweight='bold',
                )

                # Thin connecting line to matched CRAFT centroid
                if j in match_pred_to_craft:
                    ci = match_pred_to_craft[j]
                    craft_y = data.centers_rel[ci][0]
                    ax.plot(
                        [mid_x + fontsize * 0.6, w - 2],
                        [cy_cut, craft_y],
                        color='silver', linewidth=0.5, alpha=0.6,
                    )

            # Mark unmatched CRAFT centroids with a red '?'
            matched_craft_set = set(match_pred_to_craft.values())
            for ci, ((cy, cx), (_, _, _, bh)) in enumerate(zip(
                data.centers_rel, data.char_boxes_rel
            )):
                if ci not in matched_craft_set:
                    fs = max(6, min(bh * 0.5, 24))
                    ax.text(
                        mid_x, cy, '?',
                        ha='center', va='center',
                        color='red', fontsize=fs, fontstyle='italic',
                        alpha=0.6,
                    )
        else:
            # Fallback: draw matched predictions at CRAFT centroid positions
            for (cy, cx), (_, _, _, bh), char, conf in zip(
                data.centers_rel, data.char_boxes_rel,
                data.pred_chars, data.pred_confs,
            ):
                if conf >= 0.8:
                    color = 'darkgreen'
                elif conf >= 0.5:
                    color = 'darkorange'
                else:
                    color = 'red'
                fontsize = max(8, min(avg_bh * 0.65, 36))
                ax.text(
                    mid_x, cy, char,
                    ha='center', va='center',
                    color=color, fontsize=fontsize, fontweight='bold',
                )

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