#!/usr/bin/env python3
"""
Generate character glossary figures for the paper.

Produces three glossary pages:
  1. Most frequent characters (sorted by descending occurrence count)
  2. Least frequent characters (sorted by ascending occurrence count)
  3. Intermediate-size clusters (characters whose main cluster has exactly
     ``--intermediate-size`` members, default 2)

Each page is a grid of representative vectorised characters with their
label and count printed below.

Supports switching between OCR labels (``char_chat``) and ground-truth
labels (``char_consensus`` or ``char_transcription``) via ``--label-col``.

Usage:
    python scripts/figure_generation/generate_glossary_figure.py \
        --dataframe results/clustering/book1/clustered_patches \
        --output-dir paper/figures/generated

    Optional:
        --label-col  char_consensus   Use ground-truth labels
        --max-chars  300              Max characters per page
        --cols       20               Grid columns
        --cell-size  0.55             Cell size in inches
        --dpi        600              Output DPI
        --intermediate-size 2         Cluster size for intermediate page
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from font_config import configure_matplotlib_fonts, CJK_FONT_NAME
configure_matplotlib_fonts()

from notebook_utils.parquet_utils import load_dataframe, load_columns
from notebook_utils.svg_utils import render_svg_grayscale
from src.clustering.post_clustering import build_glossary, compute_representative
from src.clustering.metrics import UNKNOWN_LABEL


# ════════════════════════════════════════════════════════════════════
#  Glossary building helpers
# ════════════════════════════════════════════════════════════════════

def build_glossary_from_dataframe(dataframe, label_col='char_chat'):
    """Build a glossary dataframe from the clustering output.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Clustered dataframe with ``label_col``, ``membership``, and
        ``degree_centrality`` columns.
    label_col : str
        Which label column to use for character identity.
        ``'char_chat'`` = OCR predictions (default),
        ``'char_consensus'`` or ``'char_transcription'`` = ground truth.

    Returns
    -------
    pd.DataFrame
        Glossary with columns: character, n, n_chars_c, n_c, cluster_id,
        representative_idx.  Sorted by ``n`` descending.
    """
    required = [label_col, 'membership', 'degree_centrality']
    for col in required:
        if col not in dataframe.columns:
            raise ValueError(f"Missing required column: {col}")

    # Temporarily swap the label column to char_chat so build_glossary
    # picks it up (build_glossary hard-codes 'char_chat').
    if label_col != 'char_chat':
        tmp = dataframe['char_chat'].copy() if 'char_chat' in dataframe.columns else None
        dataframe['char_chat'] = dataframe[label_col]
        glossary = build_glossary(dataframe)
        if tmp is not None:
            dataframe['char_chat'] = tmp
        else:
            dataframe.drop(columns='char_chat', inplace=True)
    else:
        glossary = build_glossary(dataframe)

    return glossary


# ════════════════════════════════════════════════════════════════════
#  Single-page glossary figure
# ════════════════════════════════════════════════════════════════════

def generate_glossary_figure(
    dataframe,
    glossary_df,
    output_path=None,
    max_chars=300,
    n_cols=20,
    cell_size=0.55,
    dpi=600,
    render_size=256,
    show_frequency=True,
    title='Character Glossary \u2014 Representative Vectorised Characters',
):
    """Generate a publication-quality glossary grid figure.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full clustered dataframe with 'svg' column.
    glossary_df : pd.DataFrame
        One row per character, pre-sorted and pre-filtered.
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
        Pixel size for SVG rendering (higher = sharper).
    show_frequency : bool
        If True, show occurrence count below the character label.
    title : str
        Figure super-title.
    """
    glossary_df = glossary_df.head(max_chars).reset_index(drop=True)
    n_chars = len(glossary_df)
    if n_chars == 0:
        print(f"  [glossary] Skipping {output_path}: no entries")
        return None

    n_rows = int(np.ceil(n_chars / n_cols))

    fig_w = n_cols * cell_size + 0.4
    fig_h = n_rows * cell_size * 1.35 + 0.6

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

        svg_obj = dataframe.loc[rep_idx, 'svg']
        try:
            img = render_svg_grayscale(svg_obj, render_size, render_size)
        except Exception:
            img = np.ones((render_size, render_size), dtype=np.uint8) * 255

        ax.imshow(
            img, cmap='gray', vmin=0, vmax=255,
            interpolation='lanczos',
        )

        label_text = char_label if char_label != UNKNOWN_LABEL else '?'
        if show_frequency:
            label_text += f'\n({freq})'

        ax.text(
            0.5, -0.08, label_text,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=5,
        )

    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.995)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [glossary] Saved: {output_path}")
    else:
        plt.show()

    return fig


# ════════════════════════════════════════════════════════════════════
#  Multi-page generation
# ════════════════════════════════════════════════════════════════════

def generate_all_glossary_pages(
    dataframe,
    glossary_df,
    output_dir,
    label_col='char_chat',
    max_chars=300,
    n_cols=20,
    cell_size=0.55,
    dpi=600,
    render_size=256,
    intermediate_size=2,
):
    """Generate all three glossary pages.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full clustered dataframe.
    glossary_df : pd.DataFrame
        Full glossary (sorted descending by ``n``).
    output_dir : Path
        Directory for output PDFs.
    label_col : str
        Label column used (for titles).
    max_chars : int
        Max entries per page.
    intermediate_size : int
        Target main-cluster size for the intermediate page.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_tag = 'ground truth' if label_col != 'char_chat' else 'OCR'
    common_kw = dict(
        dataframe=dataframe, max_chars=max_chars, n_cols=n_cols,
        cell_size=cell_size, dpi=dpi, render_size=render_size,
    )

    # ── Page 1: most frequent characters ────────────────────────────
    most_freq = glossary_df.head(max_chars)
    generate_glossary_figure(
        glossary_df=most_freq,
        output_path=output_dir / 'glossary.pdf',
        title=f'Character Glossary \u2014 Most Frequent ({label_tag} labels)',
        **common_kw,
    )

    # ── Page 2: least frequent characters ───────────────────────────
    least_freq = (glossary_df
                  .sort_values('n', ascending=True)
                  .head(max_chars)
                  .reset_index(drop=True))
    generate_glossary_figure(
        glossary_df=least_freq,
        output_path=output_dir / 'glossary_least_frequent.pdf',
        title=f'Character Glossary \u2014 Least Frequent ({label_tag} labels)',
        **common_kw,
    )

    # ── Page 3: intermediate cluster size ───────────────────────────
    # Characters whose main cluster has exactly ``intermediate_size`` members.
    intermediate_df = (glossary_df[glossary_df['n_c'] == intermediate_size]
                       .sort_values('n', ascending=False)
                       .reset_index(drop=True))
    generate_glossary_figure(
        glossary_df=intermediate_df,
        output_path=output_dir / 'glossary_intermediate.pdf',
        title=(f'Character Glossary \u2014 '
               f'Main Cluster Size = {intermediate_size} ({label_tag} labels)'),
        **common_kw,
    )

    print(f"  [glossary] Generated 3 glossary pages in {output_dir}")
    return {
        'most_frequent': most_freq,
        'least_frequent': least_freq,
        'intermediate': intermediate_df,
    }


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate character glossary figures for the paper.",
    )
    parser.add_argument(
        "--dataframe", required=True, type=Path,
        help="Path to the clustered_patches dataframe directory.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Single-page output path (backward compatible). "
             "Mutually exclusive with --output-dir.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for multi-page glossary output.",
    )
    parser.add_argument(
        "--label-col", type=str, default="char_chat",
        help="Label column to use: char_chat (OCR), char_consensus, "
             "or char_transcription (ground truth).",
    )
    parser.add_argument("--max-chars", type=int, default=300)
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--cell-size", type=float, default=0.55)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument(
        "--intermediate-size", type=int, default=2,
        help="Main cluster size for the intermediate glossary page.",
    )
    args = parser.parse_args()

    print(f"Loading dataframe from {args.dataframe}...")
    dataframe = load_dataframe(args.dataframe)
    print(f"Loaded {len(dataframe)} patches")

    print(f"Building glossary (label_col={args.label_col})...")
    glossary_df = build_glossary_from_dataframe(dataframe, label_col=args.label_col)
    print(f"Glossary has {len(glossary_df)} character entries")

    if args.output_dir is not None:
        # Multi-page mode
        generate_all_glossary_pages(
            dataframe, glossary_df,
            output_dir=args.output_dir,
            label_col=args.label_col,
            max_chars=args.max_chars,
            n_cols=args.cols,
            cell_size=args.cell_size,
            dpi=args.dpi,
            render_size=args.render_size,
            intermediate_size=args.intermediate_size,
        )
    else:
        # Single-page mode (backward compatible)
        output = args.output or Path("paper/figures/generated/glossary.pdf")
        generate_glossary_figure(
            dataframe, glossary_df,
            output_path=output,
            max_chars=args.max_chars,
            n_cols=args.cols,
            cell_size=args.cell_size,
            dpi=args.dpi,
            render_size=args.render_size,
        )


if __name__ == "__main__":
    main()
