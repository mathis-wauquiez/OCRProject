#!/usr/bin/env python3
"""
Generate a character catalogue figure for the paper.

For each selected character, shows all patch occurrences grouped by cluster
(sorted by cluster size descending).  The representative patch is highlighted
with a coloured border (green = pure, orange = impure).

Usage:
    python scripts/figure_generation/generate_character_catalogue_figure.py \
        --dataframe results/clustering/book1/clustered_patches \
        --characters "之" "國" "不" "大" \
        --output paper/figures/generated/character_catalogue.pdf

    # To list available characters (sorted by frequency):
    python scripts/figure_generation/generate_character_catalogue_figure.py \
        --dataframe results/clustering/book1/clustered_patches \
        --list-characters
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from font_config import configure_matplotlib_fonts, CJK_FONT_NAME
configure_matplotlib_fonts()

from notebook_utils.parquet_utils import load_dataframe
from notebook_utils.svg_utils import render_svg_grayscale
from src.clustering.metrics import UNKNOWN_LABEL, compute_cluster_purity


def _render_patch(svg_obj, size=48):
    """Render an SVG object to a grayscale image."""
    try:
        return render_svg_grayscale(svg_obj, size, size)
    except Exception:
        return np.ones((size, size), dtype=np.uint8) * 255


def generate_character_catalogue(
    dataframe,
    characters,
    label_col='char_chat',
    membership_col='membership',
    max_patches_per_cluster=20,
    patch_size=48,
    output_path=None,
    dpi=300,
):
    """Generate a character catalogue figure.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Clustered dataframe with svg, membership, degree_centrality columns.
    characters : list of str
        Characters to include (each gets its own row block).
    label_col : str
        Column used for character labels.
    membership_col : str
        Column with cluster assignments.
    max_patches_per_cluster : int
        Maximum patches shown per cluster row.
    patch_size : int
        Pixel size for rendering SVG patches.
    output_path : Path or None
        Output PDF path.
    dpi : int
        Output DPI.
    """
    has_svg = 'svg' in dataframe.columns
    if not has_svg:
        raise ValueError("Dataframe must have 'svg' column")

    # Compute purity for all clusters
    purity_df, representatives = compute_cluster_purity(
        dataframe, membership_col, label_col,
    )
    cluster_sizes = dataframe[membership_col].value_counts()

    # Collect row data: each character -> list of (cluster_id, patches, rep_idx, purity)
    char_blocks = []
    for char in characters:
        char_df = dataframe[dataframe[label_col] == char]
        if char_df.empty:
            print(f"  Warning: character '{char}' not found, skipping")
            continue

        cluster_groups = char_df.groupby(membership_col)
        cluster_order = (
            cluster_groups.size()
            .sort_values(ascending=False)
            .index.tolist()
        )

        rows = []
        for cid in cluster_order:
            grp = cluster_groups.get_group(cid)
            # Sort by degree centrality (representative first)
            if 'degree_centrality' in grp.columns:
                grp = grp.sort_values('degree_centrality', ascending=False)
            sample = grp.head(max_patches_per_cluster)

            purity = purity_df.loc[cid, 'Purity'] if cid in purity_df.index else 0
            cl_total = int(cluster_sizes.get(cid, len(grp)))

            # Representative index
            rep_dict = representatives.get(cid, {})
            rep_idx = next(iter(rep_dict.values()), None) if rep_dict else None

            rows.append({
                'cluster_id': cid,
                'indices': sample.index.tolist(),
                'n_in_cluster': len(grp),
                'cluster_total': cl_total,
                'purity': purity,
                'rep_idx': rep_idx,
                'overflow': max(0, len(grp) - max_patches_per_cluster),
            })

        char_blocks.append({
            'character': char,
            'total': len(char_df),
            'n_clusters': len(cluster_order),
            'rows': rows,
        })

    if not char_blocks:
        print("No characters to display")
        return None

    # ── Calculate figure dimensions ─────────────────────────────────
    cell_w = 0.6  # inches per patch cell
    cell_h = 0.7
    label_w = 2.5  # inches for the left label column
    max_cols = max_patches_per_cluster

    total_rows = sum(len(b['rows']) for b in char_blocks)
    # Add separator rows between character blocks
    total_rows += len(char_blocks) - 1

    fig_w = label_w + max_cols * cell_w + 0.5
    fig_h = total_rows * cell_h + 1.0

    fig, axes_grid = plt.subplots(
        total_rows, max_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={'hspace': 0.3, 'wspace': 0.05,
                     'left': label_w / fig_w},
    )
    if total_rows == 1:
        axes_grid = axes_grid[np.newaxis, :]
    if max_cols == 1:
        axes_grid = axes_grid[:, np.newaxis]

    # ── Render patches ──────────────────────────────────────────────
    row_i = 0
    for block_idx, block in enumerate(char_blocks):
        char = block['character']

        for cluster_row_idx, crow in enumerate(block['rows']):
            cid = crow['cluster_id']
            purity = crow['purity']
            is_pure = purity >= 0.95

            for col_j in range(max_cols):
                ax = axes_grid[row_i, col_j]
                ax.axis('off')

                if col_j >= len(crow['indices']):
                    continue

                idx = crow['indices'][col_j]
                img = _render_patch(dataframe.loc[idx, 'svg'], patch_size)
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)

                # Highlight representative
                if idx == crow['rep_idx']:
                    border_color = 'green' if is_pure else 'orange'
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_color(border_color)
                        spine.set_linewidth(2.5)

            # Row label
            n_char = crow['n_in_cluster']
            cl_total = crow['cluster_total']
            overflow_text = f" +{crow['overflow']}" if crow['overflow'] > 0 else ""

            # Character header on first cluster row
            if cluster_row_idx == 0:
                bar_color = (
                    '#27ae60' if block['n_clusters'] == 1
                    else '#e67e22' if block['n_clusters'] <= 3
                    else '#e74c3c'
                )
                axes_grid[row_i, 0].text(
                    -0.15, 1.3,
                    f'"{char}" — {block["total"]} occurrences, '
                    f'{block["n_clusters"]} cluster'
                    f'{"s" if block["n_clusters"] != 1 else ""}',
                    transform=axes_grid[row_i, 0].transAxes,
                    fontsize=8, fontweight='bold', color=bar_color,
                    va='bottom', ha='left',
                )

            axes_grid[row_i, 0].set_ylabel(
                f'c{cid}  p={purity:.2f}  {n_char}/{cl_total}{overflow_text}',
                fontsize=6, rotation=0, labelpad=55, va='center',
                color='green' if is_pure else 'red',
                fontweight='bold',
            )

            row_i += 1

        # Separator row (blank) between character blocks
        if block_idx < len(char_blocks) - 1:
            for col_j in range(max_cols):
                axes_grid[row_i, col_j].axis('off')
            row_i += 1

    fig.suptitle(
        'Character Catalogue — Selected Characters',
        fontsize=11, fontweight='bold', y=0.995,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f"Character catalogue saved to {output_path}")
    else:
        plt.show()

    return fig


def list_characters(dataframe, label_col='char_chat', top_n=50):
    """Print the most frequent characters in the dataframe."""
    known = dataframe[
        dataframe[label_col].fillna(UNKNOWN_LABEL) != UNKNOWN_LABEL
    ]
    freq = known[label_col].value_counts()
    n_clusters = known.groupby(label_col)['membership'].nunique()

    print(f"\nTop {min(top_n, len(freq))} characters by frequency:\n")
    print(f"{'Char':>6}  {'Count':>6}  {'Clusters':>8}")
    print('-' * 26)
    for char, count in freq.head(top_n).items():
        nc = n_clusters.get(char, 0)
        print(f'{char:>6}  {count:>6}  {nc:>8}')
    print(f'\nTotal distinct characters: {len(freq)}')


def main():
    parser = argparse.ArgumentParser(
        description="Generate character catalogue figure for the paper.",
    )
    parser.add_argument(
        "--dataframe", required=True, type=Path,
        help="Path to clustered_patches dataframe directory.",
    )
    parser.add_argument(
        "--characters", nargs='+', default=None,
        help="Characters to include in the catalogue.",
    )
    parser.add_argument(
        "--label-col", type=str, default="char_chat",
        help="Label column to use.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("paper/figures/generated/character_catalogue.pdf"),
    )
    parser.add_argument("--max-patches", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=48)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--list-characters", action="store_true",
        help="List available characters and exit.",
    )
    args = parser.parse_args()

    print(f"Loading dataframe from {args.dataframe}...")
    dataframe = load_dataframe(args.dataframe)
    print(f"Loaded {len(dataframe)} patches")

    if args.list_characters:
        list_characters(dataframe, label_col=args.label_col)
        return

    if args.characters is None:
        parser.error(
            "--characters is required unless --list-characters is used. "
            "Example: --characters '之' '國' '不' '大'"
        )

    generate_character_catalogue(
        dataframe,
        characters=args.characters,
        label_col=args.label_col,
        max_patches_per_cluster=args.max_patches,
        patch_size=args.patch_size,
        output_path=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
