#!/usr/bin/env python3
"""
Generate the character glossary figure for the paper.

Produces a grid of representative vectorised characters, sorted by
frequency, with their OCR label printed below each cell.
This is the main output of the reverse typography pipeline.

Usage:
    python scripts/figure_generation/generate_glossary_figure.py \
        --dataframe results/clustering/book1/clustered_patches \
        --output     paper/figures/generated/glossary.pdf

    Optional:
        --max-chars  200     Max characters to show
        --cols       20      Number of columns in the grid
        --cell-size  0.45    Size of each cell in inches
        --dpi        300     Output DPI
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from notebook_utils.parquet_utils import load_dataframe, load_columns
from notebook_utils.svg_utils import render_svg_grayscale
from src.clustering.post_clustering import build_glossary, compute_representative
from src.clustering.metrics import UNKNOWN_LABEL


def build_glossary_from_dataframe(dataframe):
    """Build a glossary dataframe from the clustering output.

    If the glossary was not saved, rebuild it from the membership and
    label columns.
    """
    # Need char_chat, membership, degree_centrality at minimum
    required = ['char_chat', 'membership', 'degree_centrality']
    for col in required:
        if col not in dataframe.columns:
            raise ValueError(f"Missing required column: {col}")

    return build_glossary(dataframe)


def generate_glossary_figure(
    dataframe,
    glossary_df,
    output_path=None,
    max_chars=200,
    n_cols=20,
    cell_size=0.45,
    dpi=300,
    render_size=64,
    show_frequency=True,
):
    """Generate a publication-quality glossary grid figure.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full clustered dataframe with 'svg' column.
    glossary_df : pd.DataFrame
        One row per character, sorted by frequency (descending).
        Must have 'character', 'n', 'representative_idx'.
    output_path : Path or None
        Where to save; if None, show interactively.
    max_chars : int
        Maximum number of characters to include.
    n_cols : int
        Number of columns in the grid.
    cell_size : float
        Size of each cell in inches.
    dpi : int
        Output resolution.
    render_size : int
        Pixel size for SVG rendering.
    show_frequency : bool
        If True, show occurrence count below the character label.
    """
    glossary_df = glossary_df.head(max_chars).reset_index(drop=True)
    n_chars = len(glossary_df)
    n_rows = int(np.ceil(n_chars / n_cols))

    fig_w = n_cols * cell_size + 0.4  # small margin
    fig_h = n_rows * cell_size * 1.35 + 0.6  # extra for labels

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={'hspace': 0.5, 'wspace': 0.08},
    )

    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(n_rows * n_cols):
        row_i = idx // n_cols
        col_i = idx % n_cols
        ax = axes[row_i, col_i]
        ax.axis('off')

        if idx >= n_chars:
            continue

        entry = glossary_df.iloc[idx]
        rep_idx = int(entry['representative_idx'])
        char_label = entry['character']
        freq = int(entry['n'])

        # Render the SVG
        svg_obj = dataframe.loc[rep_idx, 'svg']
        try:
            img = render_svg_grayscale(svg_obj, render_size, render_size)
        except Exception:
            img = np.ones((render_size, render_size), dtype=np.uint8) * 255

        ax.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='nearest')

        # Character label below
        label_text = char_label if char_label != UNKNOWN_LABEL else '?'
        if show_frequency:
            label_text += f'\n({freq})'

        ax.text(
            0.5, -0.08, label_text,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=5,
            fontfamily='serif',
        )

    fig.suptitle(
        'Character Glossary — Representative Vectorised Characters',
        fontsize=10, fontweight='bold', y=0.995,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Glossary figure saved to {output_path}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate the character glossary figure for the paper."
    )
    parser.add_argument(
        "--dataframe", required=True, type=Path,
        help="Path to the clustered_patches dataframe directory.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("paper/figures/generated/glossary.pdf"),
        help="Output path for the figure.",
    )
    parser.add_argument("--max-chars", type=int, default=200)
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--cell-size", type=float, default=0.45)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--render-size", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading dataframe from {args.dataframe}...")
    dataframe = load_dataframe(args.dataframe)
    print(f"Loaded {len(dataframe)} patches")

    print("Building glossary...")
    glossary_df = build_glossary_from_dataframe(dataframe)
    print(f"Glossary has {len(glossary_df)} character entries")

    generate_glossary_figure(
        dataframe,
        glossary_df,
        output_path=args.output,
        max_chars=args.max_chars,
        n_cols=args.cols,
        cell_size=args.cell_size,
        dpi=args.dpi,
        render_size=args.render_size,
    )


if __name__ == "__main__":
    main()
