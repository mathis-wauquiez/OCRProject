#!/usr/bin/env python3
"""
Generate the layout analysis figure for the paper.

Produces a two-panel figure showing column detection (horizontal
projection + boundaries) and sub-column splitting within each column.

Usage:
    python scripts/figure_generation/generate_layout_figure.py \
        --image       data/datasets/book1/wdl_13516_045.jpg \
        --components  results/extraction/book1/components/wdl_13516_045.jpg.npz \
        --output      paper/figures/generated/layout_analysis.pdf
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from font_config import configure_matplotlib_fonts
configure_matplotlib_fonts()

from src.utils import connectedComponent
from src.layout_analysis.parsing import find_columns, split_to_rectangles


def generate_layout_figure(image, components, output_path=None,
                           min_col_area=2000, dpi=300):
    """Build a two-panel layout analysis figure.

    Top:  horizontal ink-density projection with column boundaries.
    Bottom: page image with column overlays and sub-column dashed lines.
    """
    labels = components.labels
    bin_image = labels != 0

    # Column detection
    left_cols, right_cols = find_columns(
        bin_image, threshold=1, min_col_area=min_col_area,
    )

    # Sub-column detection via split_to_rectangles
    rectangles = split_to_rectangles(labels, min_col_area=min_col_area)

    # ── Build figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 14))
    gs = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[1, 3], hspace=0.06,
    )
    ax_proj = fig.add_subplot(gs[0])
    ax_img = fig.add_subplot(gs[1])

    # ── Top panel: horizontal projection ────────────────────────────
    proj = bin_image.sum(axis=0)
    ax_proj.plot(proj, linewidth=0.8, color='steelblue', label='Projection')

    ymax = proj.max() * 1.15
    col_colors = plt.cm.Set2(np.linspace(0, 1, max(len(left_cols), 1)))

    for i, (l, r) in enumerate(zip(left_cols, right_cols)):
        color = col_colors[i % len(col_colors)]
        ax_proj.axvspan(l, r, alpha=0.15, color=color)
        ax_proj.axvline(l, color=color, ls='--', lw=1.5)
        ax_proj.axvline(r, color=color, ls='--', lw=1.5)

    ax_proj.set_xlim(0, len(proj))
    ax_proj.set_ylim(0, ymax)
    ax_proj.set_ylabel('Ink density', fontsize=9)
    ax_proj.set_xlabel('Horizontal position (px)', fontsize=9)
    ax_proj.tick_params(labelsize=8)
    ax_proj.text(
        0.02, 0.92,
        f'{len(left_cols)} columns detected',
        transform=ax_proj.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    )

    # ── Bottom panel: image with overlays ───────────────────────────
    canvas = image.copy()
    h_img = canvas.shape[0]

    # Draw column overlays
    for i, (l, r) in enumerate(zip(left_cols, right_cols)):
        color_bgr = tuple(int(c * 255) for c in col_colors[i % len(col_colors)][:3])
        overlay = canvas.copy()
        cv2.rectangle(overlay, (int(l), 0), (int(r), h_img), color_bgr, -1)
        cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)
        cv2.rectangle(canvas, (int(l), 0), (int(r), h_img), color_bgr, 3)

    # Draw sub-column boundaries (dashed lines) within each column
    for _, rect in rectangles.iterrows():
        col_idx = rect['col_idx']
        left_c, top_r, right_c, bottom_r = rect['bbox']
        n_subcols = len(rect['labels'])
        if n_subcols <= 1:
            continue

        # Sub-columns are evenly spaced within the column boundaries
        col_width = right_c - left_c
        subcol_width = col_width / n_subcols

        for sc in range(1, n_subcols):
            x = int(left_c + sc * subcol_width)
            # Draw dashed line for sub-column boundary
            dash_len = 8
            gap_len = 6
            y = top_r
            while y < bottom_r:
                y_end = min(y + dash_len, bottom_r)
                cv2.line(canvas, (x, y), (x, y_end), (200, 60, 60), 2)
                y = y_end + gap_len

    ax_img.imshow(canvas)
    ax_img.axis('off')

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f"Layout figure saved to {output_path}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate the layout analysis figure for the paper.",
    )
    parser.add_argument(
        "--image", required=True, type=Path,
        help="Path to a page image.",
    )
    parser.add_argument(
        "--components", required=True, type=Path,
        help="Path to saved connectedComponent .npz file from extraction.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("paper/figures/generated/layout_analysis.pdf"),
        help="Output path for the figure.",
    )
    parser.add_argument("--min-col-area", type=int, default=2000)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    image = np.array(Image.open(args.image))
    print(f"Loaded image: {args.image} ({image.shape})")

    components = connectedComponent.load(args.components)
    n = sum(1 for r in components.regions if not components.is_deleted(r.label))
    print(f"Loaded components: {n} active characters")

    generate_layout_figure(
        image, components, args.output,
        min_col_area=args.min_col_area, dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
