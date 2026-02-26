#!/usr/bin/env python3
"""
Generate the main pipeline example figure for the paper.

Produces a multi-panel figure showing the five key stages of the
extraction → preprocessing pipeline on a single page:

  (a) Input image
  (b) Extraction result (character bounding boxes)
  (c) Reading order and columns
  (d) Line extraction + OCR matching
  (e) Final alignment (OCR vs transcription)

Usage:
    python scripts/figure_generation/generate_paper_main_figure.py \
        --image       data/datasets/book1/wdl_13516_005.jpg \
        --components  results/extraction/book1/components/wdl_13516_005.jpg.npz \
        --dataframe   results/preprocessing/book1 \
        --page        5 \
        --output      paper/figures/generated/main_pipeline.pdf
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyArrowPatch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import connectedComponent
from src.layout_analysis.parsing import find_columns, split_to_rectangles, get_reading_order


# ── Colour palette ──────────────────────────────────────────────────
_GREEN  = (0, 200, 0)
_ORANGE = (255, 165, 0)
_RED    = (220, 40, 40)
_GREY   = (130, 130, 130)
_BLUE   = (60, 120, 220)


# ====================================================================
#  Panel generators
# ====================================================================

def panel_input(ax, image):
    """(a) Raw input image."""
    ax.imshow(image)
    ax.set_title("(a) Input page", fontsize=9, fontweight="bold")
    ax.axis("off")


def panel_extraction(ax, image, components):
    """(b) Extraction result — character bounding boxes on the page."""
    canvas = image.copy()
    for region in components.regions:
        if components.is_deleted(region.label):
            continue
        y0, x0, y1, x1 = region.bbox
        cv2.rectangle(canvas, (x0, y0), (x1, y1), _GREEN, 2)
    ax.imshow(canvas)
    n = sum(1 for r in components.regions if not components.is_deleted(r.label))
    ax.set_title(f"(b) Extracted characters ({n})", fontsize=9, fontweight="bold")
    ax.axis("off")


def panel_reading_order(ax, image, components):
    """(c) Columns + reading order."""
    labels = components.labels
    bin_image = labels != 0

    # Detect columns
    left_cols, right_cols = find_columns(bin_image, threshold=1, min_col_area=2000)
    rectangles = split_to_rectangles(labels, min_col_area=2000)
    ordered_labels = get_reading_order(rectangles)

    canvas = image.copy()
    h_img = canvas.shape[0]

    # Draw column rectangles
    col_colors = plt.cm.Set2(np.linspace(0, 1, max(len(left_cols), 1)))
    for i, (l, r) in enumerate(zip(left_cols, right_cols)):
        color_bgr = tuple(int(c * 255) for c in col_colors[i % len(col_colors)][:3])
        # Semi-transparent column overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (int(l), 0), (int(r), h_img), color_bgr, -1)
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
        cv2.rectangle(canvas, (int(l), 0), (int(r), h_img), color_bgr, 3)

    # Draw reading-order numbers on character centroids
    label_to_order = {lbl: idx for idx, lbl in enumerate(ordered_labels)}
    for region in components.regions:
        if components.is_deleted(region.label):
            continue
        order = label_to_order.get(region.label)
        if order is None:
            continue
        cy, cx = int(region.centroid[0]), int(region.centroid[1])
        cv2.putText(canvas, str(order), (cx - 8, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 30, 30), 1,
                    cv2.LINE_AA)

    ax.imshow(canvas)
    ax.set_title(
        f"(c) Columns ({len(left_cols)}) + reading order",
        fontsize=9, fontweight="bold",
    )
    ax.axis("off")


def panel_ocr_matching(ax, image, page_df, n_show=3):
    """(d) Line extraction + OCR result.

    Shows a few subcolumn crops side-by-side with OCR predictions.
    """
    if page_df is None or "char_chat" not in page_df.columns:
        ax.text(0.5, 0.5, "(d) OCR matching\n[run preprocessing first]",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="grey")
        ax.axis("off")
        return

    # Pick a few columns to display
    if "reading_order" in page_df.columns:
        page_df = page_df.sort_values("reading_order")

    # Convert image to grayscale for crops
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Group by approximate x-position to find columns
    page_df = page_df.copy()
    page_df["cx"] = page_df["left"] + page_df["width"] // 2
    # Cluster into columns by binning cx
    if len(page_df) == 0:
        ax.text(0.5, 0.5, "(d) No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")
        ax.axis("off")
        return

    # Use the first n_show natural column groups
    col_x_centers = []
    df_sorted = page_df.sort_values("cx", ascending=False)  # right-to-left
    prev_cx = None
    cols = []
    current_col = []
    for _, row in df_sorted.iterrows():
        cx = row["cx"]
        if prev_cx is not None and abs(cx - prev_cx) > row["width"] * 1.5:
            if current_col:
                cols.append(current_col)
            current_col = []
        current_col.append(row)
        prev_cx = cx
    if current_col:
        cols.append(current_col)

    cols = cols[:n_show]
    if not cols:
        ax.axis("off")
        return

    # Build composite: for each column, show crop + OCR text
    col_images = []
    for col_rows in cols:
        col_df = pd.DataFrame(col_rows).sort_values("top")
        x0 = int(col_df["left"].min()) - 5
        y0 = int(col_df["top"].min()) - 5
        x1 = int((col_df["left"] + col_df["width"]).max()) + 5
        y1 = int((col_df["top"] + col_df["height"]).max()) + 5
        x0, y0 = max(x0, 0), max(y0, 0)
        x1 = min(x1, gray.shape[1])
        y1 = min(y1, gray.shape[0])

        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        # Build text column
        chars = col_df["char_chat"].tolist()
        col_images.append((crop, chars))

    if not col_images:
        ax.axis("off")
        return

    # Display crops + OCR text
    n = len(col_images)
    # Create sub-layout inside the axes using inset axes
    ax.axis("off")

    total_w = sum(c.shape[1] for c, _ in col_images) + 40 * n
    max_h = max(c.shape[0] for c, _ in col_images)

    composite = np.ones((max_h, total_w, 3), dtype=np.uint8) * 255
    x_offset = 0
    for crop, chars in col_images:
        h, w = crop.shape
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        composite[:h, x_offset:x_offset + w] = crop_rgb
        # Draw OCR text next to crop
        text_x = x_offset + w + 3
        text_y = 15
        for ch in chars:
            if ch and ch != "▯":
                cv2.putText(composite, ch, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 0), 1,
                            cv2.LINE_AA)
            text_y += 18
        x_offset += w + 40

    ax.imshow(composite)
    ax.set_title("(d) OCR line extraction", fontsize=9, fontweight="bold")


def panel_final_alignment(ax, image, page_df):
    """(e) Final result with OCR / transcription + colour-coded alignment."""
    if page_df is None or "char_chat" not in page_df.columns:
        ax.text(0.5, 0.5, "(e) Final alignment\n[run preprocessing first]",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="grey")
        ax.axis("off")
        return

    canvas = image.copy()
    has_transcription = "char_transcription" in page_df.columns

    for _, row in page_df.iterrows():
        left, top = int(row["left"]), int(row["top"])
        w, h = int(row["width"]), int(row["height"])

        ocr_ch = row.get("char_chat", "▯")
        trans_ch = row.get("char_transcription", None)

        if has_transcription and trans_ch is not None and trans_ch != "▯":
            if ocr_ch == trans_ch:
                color = _GREEN    # match
            else:
                color = _ORANGE   # replace
        elif ocr_ch and ocr_ch != "▯":
            color = _BLUE         # OCR only (no transcription)
        else:
            color = _GREY         # unknown

        cv2.rectangle(canvas, (left, top), (left + w, top + h), color, 2)

    ax.imshow(canvas)

    # Legend
    legend_items = [
        Patch(facecolor=np.array(_GREEN) / 255, label="OCR = Transcription"),
        Patch(facecolor=np.array(_ORANGE) / 255, label="OCR ≠ Transcription"),
        Patch(facecolor=np.array(_BLUE) / 255, label="OCR only"),
        Patch(facecolor=np.array(_GREY) / 255, label="Unknown"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=5,
              framealpha=0.85)
    ax.set_title("(e) Alignment result", fontsize=9, fontweight="bold")
    ax.axis("off")


# ====================================================================
#  Compose figure
# ====================================================================

def build_figure(image, components=None, page_df=None, output_path=None):
    """Compose the five-panel main pipeline figure.

    Layout:
        ┌───────┐  ┌───────┐  ┌───────┐
        │  (a)  │→│  (b)  │→│  (c)  │
        └───────┘  └───────┘  └───────┘
              ┌───────────┐  ┌───────────┐
              │    (d)    │→│    (e)    │
              └───────────┘  └───────────┘
    """
    fig = plt.figure(figsize=(14, 10))

    # Top row: 3 panels (input, extraction, columns)
    # Bottom row: 2 wider panels (OCR, alignment)
    gs = gridspec.GridSpec(
        2, 6,
        figure=fig,
        height_ratios=[1, 1],
        hspace=0.15,
        wspace=0.08,
    )

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])

    # (a) Input
    panel_input(ax_a, image)

    # (b) Extraction
    if components is not None:
        panel_extraction(ax_b, image, components)
    else:
        ax_b.text(0.5, 0.5, "(b) Extraction\n[load components]",
                  ha="center", va="center", transform=ax_b.transAxes,
                  fontsize=10, color="grey")
        ax_b.axis("off")
        ax_b.set_title("(b) Extracted characters", fontsize=9, fontweight="bold")

    # (c) Reading order + columns
    if components is not None:
        panel_reading_order(ax_c, image, components)
    else:
        ax_c.text(0.5, 0.5, "(c) Columns + reading order\n[load components]",
                  ha="center", va="center", transform=ax_c.transAxes,
                  fontsize=10, color="grey")
        ax_c.axis("off")
        ax_c.set_title("(c) Columns + reading order", fontsize=9, fontweight="bold")

    # (d) OCR matching
    panel_ocr_matching(ax_d, image, page_df)

    # (e) Final alignment
    panel_final_alignment(ax_e, image, page_df)

    # ── Arrows between panels (annotation arrows in figure coords) ──
    arrow_kw = dict(
        arrowstyle="->", color="black", lw=1.5,
        connectionstyle="arc3,rad=0",
    )
    for (src_ax, dst_ax) in [(ax_a, ax_b), (ax_b, ax_c), (ax_d, ax_e)]:
        fig.add_artist(FancyArrowPatch(
            (1.03, 0.5), (-0.03, 0.5),
            transform=src_ax.transAxes,
            **arrow_kw,
        ))

    # Downward arrow from top row to bottom row
    fig.add_artist(FancyArrowPatch(
        (0.5, -0.06), (0.5, 1.06),
        transform=ax_d.transAxes,
        arrowstyle="->", color="black", lw=1.5,
        connectionstyle="arc3,rad=0",
    ))

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"Figure saved to {output_path}")
    else:
        plt.show()

    return fig


# ====================================================================
#  CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate the main pipeline figure for the paper."
    )
    parser.add_argument(
        "--image", required=True, type=Path,
        help="Path to a page image (e.g. data/datasets/book1/wdl_13516_005.jpg)",
    )
    parser.add_argument(
        "--components", type=Path, default=None,
        help="Path to saved connectedComponent .npz file from extraction.",
    )
    parser.add_argument(
        "--dataframe", type=Path, default=None,
        help="Path to the preprocessed dataframe directory.",
    )
    parser.add_argument(
        "--page", type=int, default=None,
        help="Page index to filter from the dataframe.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("paper/figures/generated/main_pipeline.pdf"),
        help="Output path for the figure.",
    )
    args = parser.parse_args()

    # Load image
    image = np.array(Image.open(args.image))
    print(f"Loaded image: {args.image} ({image.shape})")

    # Load components
    components = None
    if args.components and args.components.exists():
        components = connectedComponent.load(args.components)
        n = sum(1 for r in components.regions
                if not components.is_deleted(r.label))
        print(f"Loaded components: {n} active characters")

    # Load dataframe
    page_df = None
    if args.dataframe and args.dataframe.exists():
        from notebook_utils.parquet_utils import load_columns
        cols = ["reading_order", "char_chat", "file",
                "left", "top", "width", "height", "label", "centroid"]
        try:
            cols.append("char_transcription")
            df = load_columns(args.dataframe, cols)
        except Exception:
            cols.remove("char_transcription")
            df = load_columns(args.dataframe, cols)

        if args.page is not None:
            # Filter by page
            if "file" in df.columns:
                # Match page number from filename
                df["_page"] = df["file"].apply(
                    lambda x: int(x.split("_")[-1].split(".")[0])
                )
                page_df = df[df["_page"] == args.page].copy()
                page_df.drop(columns=["_page"], inplace=True)
            else:
                page_df = df[df["page"] == args.page].copy()
        else:
            page_df = df

        print(f"Loaded dataframe: {len(page_df)} rows for page {args.page}")

    build_figure(image, components, page_df, args.output)


if __name__ == "__main__":
    main()
