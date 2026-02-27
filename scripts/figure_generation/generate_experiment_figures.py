#!/usr/bin/env python3
"""
Generate all experiment figures and LaTeX tables for the paper.

Reads pipeline outputs and produces:
  F4:  Preprocessing before/after (img_patch -> bin_patch -> SVG render)
  F5:  HOG descriptor example (character, gradients, histogram)
  F6:  A contrario distribution (dissimilarity histogram + Gaussian fit)
  F8:  Cluster gallery (pure vs impure examples)
  F9:  Epsilon sensitivity (ARI vs epsilon per partition type)
  F10: Split threshold sensitivity (ARI vs tau_split)
  --   LaTeX macros file with all numeric values
  --   Ablation tables (A1: HOG config, A2: CPM vs Modularity, A4: reciprocal)

Usage:
    python scripts/figure_generation/generate_experiment_figures.py \
        --clustering-dir   results/clustering/book1 \
        --output-dir       paper/figures/generated

    Optional:
        --preprocessing-dir results/preprocessing/book1  (for F4)
        --dpi 300
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from font_config import configure_matplotlib_fonts
configure_matplotlib_fonts()

from notebook_utils.parquet_utils import load_dataframe, load_columns
from notebook_utils.svg_utils import render_svg_grayscale
from src.clustering.metrics import (
    UNKNOWN_LABEL, compute_metrics,
    compute_cluster_purity, compute_label_completeness,
)
from src.clustering.post_clustering import build_glossary


# ════════════════════════════════════════════════════════════════════
#  Utilities
# ════════════════════════════════════════════════════════════════════

def _filter_unknown(dataframe, label_col):
    labels = dataframe[label_col].fillna(UNKNOWN_LABEL)
    return labels != UNKNOWN_LABEL


def _evaluate(dataframe, membership_col, label_col):
    mask = _filter_unknown(dataframe, label_col)
    return compute_metrics(
        reference_labels=dataframe.loc[mask, label_col].values,
        predicted_labels=dataframe.loc[mask, membership_col].values,
    )


def _try_load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None


# ════════════════════════════════════════════════════════════════════
#  F4: Preprocessing before/after
# ════════════════════════════════════════════════════════════════════

def generate_preprocessing_figure(df, output_path, n_examples=6, dpi=300):
    """Show img_patch -> bin_patch -> SVG render for several characters."""
    required = ['img_patch', 'bin_patch', 'svg']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [F4] Skipping: missing columns {missing}")
        return None

    # Pick diverse examples (different characters)
    indices = []
    if 'char_chat' in df.columns:
        seen = set()
        for idx in df.index:
            char = df.loc[idx, 'char_chat']
            if char not in seen and len(indices) < n_examples:
                indices.append(idx)
                seen.add(char)
    if len(indices) < n_examples:
        for idx in df.index:
            if idx not in indices and len(indices) < n_examples:
                indices.append(idx)

    n = len(indices)
    fig, axes = plt.subplots(3, n, figsize=(n * 1.8, 3 * 1.8))
    if n == 1:
        axes = axes[:, np.newaxis]

    row_labels = ['Image patch', 'Binarized patch', 'Vectorised (SVG)']

    for col_i, idx in enumerate(indices):
        row = df.loc[idx]

        img = np.array(row['img_patch'])
        axes[0, col_i].imshow(img, cmap='gray')
        axes[0, col_i].axis('off')

        bimg = np.array(row['bin_patch'])
        axes[1, col_i].imshow(bimg, cmap='gray')
        axes[1, col_i].axis('off')

        svg_obj = row['svg']
        h = img.shape[0] if len(img.shape) >= 2 else 64
        scale = max(1.0, 256 / max(h, 1))
        try:
            svg_render = svg_obj.render(
                dpi=96, output_format='L', scale=scale,
                output_size=(256, 256), respect_aspect_ratio=True,
            )
        except Exception:
            svg_render = np.ones((64, 64), dtype=np.uint8) * 255
        axes[2, col_i].imshow(svg_render, cmap='gray', vmin=0, vmax=255)
        axes[2, col_i].axis('off')

        char = row.get('char_chat', '')
        if char:
            axes[0, col_i].set_title(char, fontsize=9, fontfamily='serif')

    for row_i, label in enumerate(row_labels):
        axes[row_i, 0].set_ylabel(label, fontsize=8, rotation=90, labelpad=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [F4] Saved: {output_path}")
    return fig


# ════════════════════════════════════════════════════════════════════
#  F5: HOG descriptor example
# ════════════════════════════════════════════════════════════════════

def _plot_hog_arrows(ax, img, histograms, Nh, Nw, ch, cw, Nbins,
                     threshold=0.05, arrow_scale=0.45):
    """Draw HOG orientation arrows overlaid on an image.

    Parameters
    ----------
    ax : matplotlib Axes
    img : 2-D array (H, W) — background image
    histograms : (Nh, Nw, Nbins) — cell histograms
    Nh, Nw : grid dimensions
    ch, cw : cell height/width in pixels
    Nbins : number of orientation bins
    """
    from matplotlib.colors import hsv_to_rgb

    ax.imshow(img, cmap='gray', alpha=0.3)

    bin_angles = np.linspace(0, np.pi, Nbins, endpoint=False)
    max_magnitude = histograms.max()
    min_threshold = threshold * max_magnitude

    for cell_y in range(Nh):
        for cell_x in range(Nw):
            center_x = (cell_x + 0.5) * cw
            center_y = (cell_y + 0.5) * ch
            cell_hist = histograms[cell_y, cell_x]

            for bin_idx, magnitude in enumerate(cell_hist):
                if magnitude > min_threshold:
                    angle = bin_angles[bin_idx]
                    length = magnitude * min(cw, ch) * arrow_scale / max_magnitude
                    dx = length * np.cos(angle)
                    dy = length * np.sin(angle)
                    color = plt.cm.hsv(angle / (2 * np.pi))

                    arrow_props = {
                        'head_width': 2.0, 'head_length': 2.0,
                        'fc': color, 'ec': color,
                        'alpha': 0.85, 'linewidth': 2.0,
                    }
                    ax.arrow(center_x, center_y, dx, dy, **arrow_props)
                    ax.arrow(center_x, center_y, -dx, -dy, **arrow_props)

    # Grid lines
    for gy in range(Nh + 1):
        ax.axhline(y=gy * ch - 0.5, color='cyan', linewidth=1.0, alpha=0.6)
    for gx in range(Nw + 1):
        ax.axvline(x=gx * cw - 0.5, color='cyan', linewidth=1.0, alpha=0.6)

    ax.set_xlim(-0.5, Nw * cw - 0.5)
    ax.set_ylim(Nh * ch - 0.5, -0.5)
    ax.axis('off')


def generate_hog_figure(df, output_path, dpi=300):
    """Render one character through the full HOG pipeline.

    Produces a 2x2 figure:
      top-left:     Gradient magnitude (hot colourmap)
      top-right:    Gradient orientation (HSV)
      bottom-left:  HSV composite (hue=orientation, value=magnitude)
      bottom-right: HOG histogram arrows overlaid on the magnitude map
    """
    if 'svg' not in df.columns or 'histogram' not in df.columns:
        print("  [F5] Skipping: missing svg or histogram columns")
        return None

    try:
        import torch
        from einops import rearrange
        from matplotlib.colors import hsv_to_rgb
        from src.patch_processing.configs import get_hog_cfg
        from src.patch_processing.hog import HOG
    except ImportError as e:
        print(f"  [F5] Skipping: {e}")
        return None

    cell_size = int(df['best_cell_size'].iloc[0]) if 'best_cell_size' in df.columns else 22
    grdt_sigma = float(df['best_grdt_sigma'].iloc[0]) if 'best_grdt_sigma' in df.columns else 5
    num_bins = int(df['best_num_bins'].iloc[0]) if 'best_num_bins' in df.columns else 16

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    renderer_factory, hog_params = get_hog_cfg(
        cell_size, grdt_sigma, num_bins, 'patch', device=device
    )

    # Pick a representative character
    if 'degree_centrality' in df.columns:
        idx = df['degree_centrality'].idxmax()
    else:
        idx = df.index[len(df) // 2]

    svg_obj = df.loc[idx, 'svg']
    renderer = renderer_factory([svg_obj])
    canvas_img = renderer[0]

    hog_params.partial_output = False
    hog = HOG(hog_params)

    with torch.no_grad():
        input_tensor = canvas_img.unsqueeze(0).unsqueeze(0).to(
            dtype=torch.float32, device=device
        )
        hog_output = hog(input_tensor)

    rendered = canvas_img.cpu().numpy()
    magnitude = hog_output.patches_grdt_magnitude[0, 0].cpu().numpy()
    orientation = hog_output.patches_grdt_orientation[0, 0].cpu().numpy()

    # Reshape histogram to spatial grid (Nh, Nw, Nbins)
    cw = hog_params.cell_width
    ch_cell = hog_params.cell_height
    Nw = rendered.shape[-1] // cw
    Nh = rendered.shape[-2] // ch_cell

    histograms_raw = hog_output.histograms[0, 0].cpu()  # (N_cells, N_bins)
    selected_histograms = rearrange(
        histograms_raw, '(Nh Nw) Nbins -> Nh Nw Nbins', Nh=Nh, Nw=Nw,
    ).numpy()

    # ── Build figure (2x2) ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # (a) Gradient magnitude
    ax1 = axes[0, 0]
    im1 = ax1.imshow(magnitude, cmap='hot')
    ax1.set_title('(a) Gradient magnitude', fontsize=10, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # (b) Gradient orientation (unsigned: 0-180)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(orientation, cmap='hsv', vmin=0, vmax=1)
    ax2.set_title('(b) Gradient orientation (0\u2013180\u00b0)', fontsize=10, fontweight='bold')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('[0,1] \u2192 [0\u00b0,180\u00b0]', rotation=270, labelpad=20)

    # (c) HSV composite  (hue = orientation, value = magnitude)
    ax_hsv = axes[1, 0]
    mag_norm = magnitude / magnitude.max() if magnitude.max() > 0 else magnitude
    hsv_image = np.zeros((*orientation.shape, 3))
    hsv_image[..., 0] = orientation          # H: [0,1] already
    hsv_image[..., 1] = 1.0                  # S: full
    hsv_image[..., 2] = mag_norm             # V: brightness = magnitude
    rgb_image = hsv_to_rgb(hsv_image)
    ax_hsv.imshow(rgb_image)
    ax_hsv.set_title('(c) HSV composite\n(hue=orientation, value=magnitude)',
                     fontsize=10, fontweight='bold')
    ax_hsv.axis('off')

    # (d) HOG histogram arrows
    ax3 = axes[1, 1]
    _plot_hog_arrows(
        ax3, magnitude, selected_histograms,
        Nh, Nw, ch_cell, cw, num_bins,
    )
    ax3.set_title(
        f'(d) HOG descriptor ({Nh}\u00d7{Nw} cells, {num_bins} bins)',
        fontsize=10, fontweight='bold',
    )

    char = df.loc[idx, 'char_chat'] if 'char_chat' in df.columns else ''
    fig.suptitle(
        f'HOG descriptor computation \u2014 character \u201c{char}\u201d',
        fontsize=12, fontweight='bold',
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [F5] Saved: {output_path}")
    return fig


# ════════════════════════════════════════════════════════════════════
#  F6: A contrario distribution
# ════════════════════════════════════════════════════════════════════

def generate_a_contrario_figure(df, output_path, n_sample=500, dpi=300):
    """Histogram of dissimilarities for a sample character + Gaussian fit."""
    if 'histogram' not in df.columns or 'mu_tot' not in df.columns:
        print("  [F6] Skipping: missing histogram or mu_tot columns")
        return None

    try:
        import torch
        from src.clustering.feature_matching import featureMatching
        from src.clustering.params import featureMatchingParameters
    except ImportError as e:
        print(f"  [F6] Skipping: {e}")
        return None

    # Pick a well-connected character
    if 'degree_centrality' in df.columns:
        idx = df['degree_centrality'].idxmax()
    else:
        idx = df.index[0]

    mu_i = float(df.loc[idx, 'mu_tot'])
    var_i = float(df.loc[idx, 'var_tot'])
    sigma_i = np.sqrt(max(var_i, 1e-12))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    query_hist = torch.tensor(
        np.array(df.loc[idx, 'histogram'])[np.newaxis],
        dtype=torch.float32, device=device,
    )

    other_indices = [i for i in df.index if i != idx]
    sample_indices = np.random.choice(
        other_indices, size=min(n_sample, len(other_indices)), replace=False,
    )
    key_hists = torch.tensor(
        np.stack([df.loc[i, 'histogram'] for i in sample_indices]),
        dtype=torch.float32, device=device,
    )

    params = featureMatchingParameters(
        metric='CEMD', epsilon=0.001, partial_output=False,
    )
    matcher = featureMatching(params)
    dissim = matcher.compute_dissimilarities(query_hist, key_hists)
    total_dissim = dissim.sum(-1)[0].cpu().numpy()

    N = len(df)
    nlfa_threshold = -(np.log(0.001) - 2 * np.log(N))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: histogram + Gaussian
    ax1.hist(
        total_dissim, bins=60, density=True, alpha=0.7, color='steelblue',
        edgecolor='white', linewidth=0.5, label='Observed',
    )
    x = np.linspace(total_dissim.min(), total_dissim.max(), 300)
    gaussian = np.exp(-0.5 * ((x - mu_i) / sigma_i) ** 2) / (
        sigma_i * np.sqrt(2 * np.pi)
    )
    ax1.plot(
        x, gaussian, 'r-', lw=2,
        label=f'$\\mathcal{{N}}(\\mu={mu_i:.2f},\\,\\sigma={sigma_i:.2f})$',
    )
    ax1.set_xlabel('Total dissimilarity $D(a^i, b^j)$', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.set_title('A contrario null-hypothesis test', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    char = df.loc[idx, 'char_chat'] if 'char_chat' in df.columns else '?'
    ax1.text(
        0.02, 0.95, f'Query: "{char}"  (N={len(sample_indices)} keys)',
        transform=ax1.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    # Right: scatter of per-character null-model params
    ax2.scatter(
        df['mu_tot'].values, np.sqrt(df['var_tot'].values.clip(0)),
        s=5, alpha=0.3, color='steelblue',
    )
    ax2.scatter(
        [mu_i], [sigma_i], s=80, color='red', zorder=5,
        edgecolors='black', label=f'Query "{char}"',
    )
    ax2.set_xlabel('$\\mu_i$ (null-model mean)', fontsize=10)
    ax2.set_ylabel('$\\sigma_i$ (null-model std)', fontsize=10)
    ax2.set_title('Per-character null-model parameters', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [F6] Saved: {output_path}")
    return fig


# ════════════════════════════════════════════════════════════════════
#  F8: Cluster gallery
# ════════════════════════════════════════════════════════════════════

def generate_cluster_gallery(
    dataframe, label_col='char_consensus',
    n_pure=4, n_impure=2, max_members=12,
    output_path=None, dpi=300,
):
    """Pure vs. impure cluster examples, representative highlighted."""
    if 'membership' not in dataframe.columns:
        print("  [F8] Skipping: no 'membership' column")
        return None

    purity_df, representatives = compute_cluster_purity(
        dataframe, 'membership', label_col,
    )
    purity_df = purity_df.sort_values(
        ['Purity', 'Size'], ascending=[False, False],
    )

    pure_mask = (purity_df['Purity'] >= 0.95) & (purity_df['Size'] >= 5)
    pure_cids = purity_df[pure_mask].head(n_pure).index.tolist()

    impure_mask = (purity_df['Purity'] < 0.85) & (purity_df['Size'] >= 5)
    impure_cids = purity_df[impure_mask].tail(n_impure).index.tolist()

    all_cids = pure_cids + impure_cids
    n_rows = len(all_cids)
    if n_rows == 0:
        print("  [F8] No suitable clusters found")
        return None

    fig, axes = plt.subplots(
        n_rows, max_members,
        figsize=(max_members * 0.6, n_rows * 0.8),
        gridspec_kw={'hspace': 0.4, 'wspace': 0.05},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_i, cid in enumerate(all_cids):
        members = dataframe[dataframe['membership'] == cid]
        members = members.sort_values('degree_centrality', ascending=False)
        members = members.head(max_members)
        is_pure = cid in pure_cids

        # Find representative (most-central known-label member)
        rep_dict = representatives.get(cid, {})
        rep_idx = next(iter(rep_dict.values()), None) if rep_dict else None

        purity_val = purity_df.loc[cid, 'Purity']
        size_val = purity_df.loc[cid, 'Size']

        for col_j in range(max_members):
            ax = axes[row_i, col_j]
            ax.axis('off')
            if col_j >= len(members):
                continue

            idx = members.index[col_j]
            img = render_svg_grayscale(dataframe.loc[idx, 'svg'], 48, 48)
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)

            if idx == rep_idx:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('green' if is_pure else 'orange')
                    spine.set_linewidth(2)

        label_char = members.iloc[0].get('char_chat', '?')
        color = 'green' if is_pure else 'red'
        axes[row_i, 0].set_ylabel(
            f'{label_char}  p={purity_val:.2f}  n={size_val}',
            fontsize=7, rotation=0, labelpad=60, va='center',
            color=color, fontweight='bold',
        )

    if pure_cids:
        axes[0, max_members // 2].set_title(
            'Pure clusters', fontsize=9, fontweight='bold', color='green',
        )
    if impure_cids and len(pure_cids) < n_rows:
        axes[len(pure_cids), max_members // 2].set_title(
            'Impure clusters', fontsize=9, fontweight='bold', color='red',
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [F8] Saved: {output_path}")
    return fig


# ════════════════════════════════════════════════════════════════════
#  F9: Epsilon sensitivity
# ════════════════════════════════════════════════════════════════════

def generate_epsilon_sensitivity(sweep_df, output_path, dpi=300):
    """ARI vs epsilon, one line per partition type."""
    if sweep_df is None or len(sweep_df) == 0:
        print("  [F9] Skipping: no sweep results")
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))

    has_pt = 'partition_type' in sweep_df.columns
    has_kr = 'keep_reciprocal' in sweep_df.columns

    if has_pt and has_kr and sweep_df['keep_reciprocal'].nunique() > 1:
        for (pt, kr), grp in sweep_df.groupby(['partition_type', 'keep_reciprocal']):
            avg = grp.groupby('epsilon')['adjusted_rand_index'].agg(['mean', 'std'])
            ls = '-' if kr else '--'
            label = f'{pt} ({"reciprocal" if kr else "non-reciprocal"})'
            ax.errorbar(
                avg.index, avg['mean'], yerr=avg['std'],
                marker='o', ls=ls, label=label, capsize=3, markersize=5,
            )
    elif has_pt:
        for pt, grp in sweep_df.groupby('partition_type'):
            avg = grp.groupby('epsilon')['adjusted_rand_index'].agg(['mean', 'std'])
            ax.errorbar(
                avg.index, avg['mean'], yerr=avg['std'],
                marker='o', label=pt, capsize=3, markersize=5,
            )
    else:
        avg = sweep_df.groupby('epsilon')['adjusted_rand_index'].agg(['mean', 'std'])
        ax.errorbar(
            avg.index, avg['mean'], yerr=avg['std'],
            marker='o', capsize=3, markersize=5,
        )

    ax.set_xlabel('NFA threshold $\\varepsilon$', fontsize=11)
    ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=11)
    ax.set_title('Sensitivity to NFA threshold', fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [F9] Saved: {output_path}")
    return fig


# ════════════════════════════════════════════════════════════════════
#  F10: Split threshold sensitivity
# ════════════════════════════════════════════════════════════════════

def generate_split_threshold_figure(split_sweep_df, output_path, dpi=300):
    """ARI and cluster count vs. Hausdorff split threshold."""
    if split_sweep_df is None or len(split_sweep_df) == 0:
        print("  [F10] Skipping: no split sweep results")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    metric_cols = [
        c for c in split_sweep_df.columns
        if c in {
            'adjusted_rand_index', 'normalized_mutual_info',
            'v_measure', 'purity',
        }
    ]

    for col in metric_cols:
        ax1.plot(
            split_sweep_df['split_threshold'], split_sweep_df[col],
            marker='o', label=col.replace('_', ' '), markersize=5,
        )

    if 'adjusted_rand_index' in split_sweep_df.columns:
        best_idx = split_sweep_df['adjusted_rand_index'].idxmax()
        best_thresh = split_sweep_df.loc[best_idx, 'split_threshold']
        ax1.axvline(
            best_thresh, color='red', ls='--', alpha=0.7,
            label=f'Best ($\\tau$={best_thresh:.1f})',
        )

    ax1.set_xlabel('Split threshold $\\tau_{\\mathrm{split}}$', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Metrics vs. Hausdorff split threshold', fontsize=11)
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(alpha=0.3)

    ax2.plot(
        split_sweep_df['split_threshold'],
        split_sweep_df['n_clusters_post_split'],
        marker='s', color='#667eea', label='Total clusters',
    )
    if 'n_clusters_actually_split' in split_sweep_df.columns:
        ax2.plot(
            split_sweep_df['split_threshold'],
            split_sweep_df['n_clusters_actually_split'],
            marker='^', color='#f5576c', label='Clusters split',
        )
    if 'adjusted_rand_index' in split_sweep_df.columns:
        ax2.axvline(best_thresh, color='red', ls='--', alpha=0.7)

    ax2.set_xlabel('Split threshold $\\tau_{\\mathrm{split}}$', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Fragmentation vs. split threshold', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [F10] Saved: {output_path}")
    return fig


# ════════════════════════════════════════════════════════════════════
#  OCR-only baseline
# ════════════════════════════════════════════════════════════════════

def compute_ocr_baseline(dataframe, label_col='char_consensus'):
    mask = _filter_unknown(dataframe, label_col)
    ocr_labels = dataframe.loc[mask, 'char_chat'].fillna(UNKNOWN_LABEL)
    ocr_mask = ocr_labels != UNKNOWN_LABEL
    return compute_metrics(
        reference_labels=dataframe.loc[mask, label_col].values[ocr_mask],
        predicted_labels=ocr_labels.values[ocr_mask],
    )


# ════════════════════════════════════════════════════════════════════
#  LaTeX macros
# ════════════════════════════════════════════════════════════════════

def generate_latex_macros(
    dataframe, graph, label_col='char_consensus', output_path=None,
):
    lines = [
        '% Auto-generated by generate_experiment_figures.py',
        '% Do not edit manually — re-run the script to update.',
        '',
    ]

    def cmd(name, value):
        lines.append(f'\\newcommand{{\\{name}}}{{{value}}}')

    cmd('nPatches', f'{len(dataframe):,}')
    cmd('nPages', f'{dataframe["file"].nunique() if "file" in dataframe.columns else "?"}')

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    density = nx.density(graph)
    degrees = [d for _, d in graph.degree()]
    avg_degree = np.mean(degrees) if degrees else 0
    n_components = nx.number_connected_components(graph)
    n_isolated = nx.number_of_isolates(graph)

    cmd('nNodes', f'{n_nodes:,}')
    cmd('nEdges', f'{n_edges:,}')
    cmd('graphDensity', f'{density:.4f}')
    cmd('avgDegree', f'{avg_degree:.1f}')
    cmd('nComponents', f'{n_components:,}')
    cmd('nIsolated', f'{n_isolated:,}')

    for stage, mem_col in [
        ('Baseline', 'membership_pre_split'),
        ('Final', 'membership'),
    ]:
        if mem_col not in dataframe.columns:
            continue
        metrics = _evaluate(dataframe, mem_col, label_col)
        n_clusters = dataframe[mem_col].nunique()
        prefix = stage
        cmd(f'nClusters{prefix}', f'{n_clusters:,}')
        cmd(f'ARI{prefix}', f'{metrics["adjusted_rand_index"]:.4f}')
        cmd(f'NMI{prefix}', f'{metrics["normalized_mutual_info"]:.4f}')
        cmd(f'Vmeasure{prefix}', f'{metrics["v_measure"]:.4f}')
        cmd(f'Purity{prefix}', f'{metrics["purity"]:.4f}')
        cmd(f'Accuracy{prefix}', f'{metrics["accuracy_optimal_match"]:.4f}')

    ocr_metrics = compute_ocr_baseline(dataframe, label_col)
    cmd('ARIocr', f'{ocr_metrics["adjusted_rand_index"]:.4f}')
    cmd('NMIocr', f'{ocr_metrics["normalized_mutual_info"]:.4f}')
    cmd('Vmeasureocr', f'{ocr_metrics["v_measure"]:.4f}')
    cmd('Purityocr', f'{ocr_metrics["purity"]:.4f}')

    glossary_df = build_glossary(dataframe)
    cmd('nGlossaryEntries', f'{len(glossary_df):,}')

    content = '\n'.join(lines) + '\n'

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"  [LaTeX] Saved: {output_path}")

    return content


# ════════════════════════════════════════════════════════════════════
#  Ablation tables (A1, A2, A4)
# ════════════════════════════════════════════════════════════════════

def export_ablation_tables(sweep_df, output_dir, dpi=300):
    """Export ablation results as LaTeX tables and figures."""
    if sweep_df is None or len(sweep_df) == 0:
        print("  [Ablations] Skipping: no sweep results")
        return

    output_dir = Path(output_dir)
    _keep_metrics = [
        'adjusted_rand_index', 'normalized_mutual_info',
        'v_measure', 'purity',
    ]

    def _best_per_group(df, group_col):
        return df.loc[df.groupby(group_col)['adjusted_rand_index'].idxmax()]

    # A1: HOG config comparison
    if 'hog_config' in sweep_df.columns and sweep_df['hog_config'].nunique() > 1:
        best = _best_per_group(sweep_df, 'hog_config')
        cols = ['hog_config', 'cell_size', 'grdt_sigma', 'num_bins'] + _keep_metrics
        cols = [c for c in cols if c in best.columns]
        table = best[cols].sort_values('adjusted_rand_index', ascending=False)
        latex = table.to_latex(index=False, float_format='%.4f')
        (output_dir / 'ablation_a1_hog_config.tex').write_text(latex)
        print(f"  [A1] HOG config ablation saved")

    # A2: CPM vs Modularity
    if 'partition_type' in sweep_df.columns and sweep_df['partition_type'].nunique() > 1:
        best = _best_per_group(sweep_df, 'partition_type')
        cols = ['partition_type', 'epsilon', 'gamma'] + _keep_metrics
        cols = [c for c in cols if c in best.columns]
        table = best[cols].sort_values('adjusted_rand_index', ascending=False)
        latex = table.to_latex(index=False, float_format='%.4f')
        (output_dir / 'ablation_a2_partition_type.tex').write_text(latex)
        print(f"  [A2] Partition type ablation saved")

        # Also generate a figure
        fig, ax = plt.subplots(figsize=(6, 4))
        for pt, grp in sweep_df.groupby('partition_type'):
            avg = grp.groupby('gamma')['adjusted_rand_index'].agg(['mean', 'std'])
            ax.errorbar(
                avg.index, avg['mean'], yerr=avg['std'],
                marker='o', label=pt, capsize=3, markersize=5,
            )
        ax.set_xlabel('Resolution $\\gamma$', fontsize=11)
        ax.set_ylabel('ARI', fontsize=11)
        ax.set_title('CPM vs. RBConfiguration', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(
            output_dir / 'ablation_a2_partition_type.pdf',
            dpi=dpi, bbox_inches='tight', facecolor='white',
        )
        plt.close(fig)
        print(f"  [A2] Partition type figure saved")

    # A4: Reciprocal vs non-reciprocal edges
    if 'keep_reciprocal' in sweep_df.columns and sweep_df['keep_reciprocal'].nunique() > 1:
        best = _best_per_group(sweep_df, 'keep_reciprocal')
        cols = ['keep_reciprocal', 'epsilon', 'gamma'] + _keep_metrics
        cols = [c for c in cols if c in best.columns]
        table = best[cols].sort_values('adjusted_rand_index', ascending=False)
        latex = table.to_latex(index=False, float_format='%.4f')
        (output_dir / 'ablation_a4_reciprocal.tex').write_text(latex)
        print(f"  [A4] Reciprocal edges ablation saved")

        fig, ax = plt.subplots(figsize=(6, 4))
        for kr, grp in sweep_df.groupby('keep_reciprocal'):
            avg = grp.groupby('epsilon')['adjusted_rand_index'].agg(['mean', 'std'])
            label = 'Reciprocal' if kr else 'Non-reciprocal'
            ax.errorbar(
                avg.index, avg['mean'], yerr=avg['std'],
                marker='o', label=label, capsize=3, markersize=5,
            )
        ax.set_xlabel('$\\varepsilon$', fontsize=11)
        ax.set_ylabel('ARI', fontsize=11)
        ax.set_xscale('log')
        ax.set_title('Reciprocal vs. non-reciprocal edges', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(
            output_dir / 'ablation_a4_reciprocal.pdf',
            dpi=dpi, bbox_inches='tight', facecolor='white',
        )
        plt.close(fig)
        print(f"  [A4] Reciprocal edges figure saved")


# ════════════════════════════════════════════════════════════════════
#  Discrepancy statistics
# ════════════════════════════════════════════════════════════════════

def generate_discrepancy_figure(
    dataframe, label_col='char_consensus', output_path=None, dpi=300,
):
    """Generate discrepancy statistics between OCR and ground-truth labels.

    Produces a two-panel figure:
      Left:  bar chart of per-category discrepancy counts
             (match / mismatch / OCR-only / GT-only).
      Right: top-N most confused character pairs
             (OCR predicted X but GT says Y).

    The ground-truth column is always ``char_transcription`` (the external
    transcription), *not* ``char_consensus``, which is derived from the
    agreement of OCR and transcription and would therefore never produce
    mismatches or GT-only entries when compared against ``char_chat``.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must have ``char_chat`` and ``char_transcription`` columns.
    label_col : str
        Ignored for this figure (kept for call-signature compatibility).
    output_path : Path or None
    dpi : int
    """
    gt_col = 'char_transcription'
    if 'char_chat' not in dataframe.columns or gt_col not in dataframe.columns:
        print(f"  [Discrepancy] Skipping: missing char_chat or {gt_col}")
        return None

    ocr = dataframe['char_chat'].fillna(UNKNOWN_LABEL)
    gt = dataframe[gt_col].fillna(UNKNOWN_LABEL)

    # Classify each patch
    both_known = (ocr != UNKNOWN_LABEL) & (gt != UNKNOWN_LABEL)
    ocr_only = (ocr != UNKNOWN_LABEL) & (gt == UNKNOWN_LABEL)
    gt_only = (ocr == UNKNOWN_LABEL) & (gt != UNKNOWN_LABEL)
    both_unknown = (ocr == UNKNOWN_LABEL) & (gt == UNKNOWN_LABEL)

    match = both_known & (ocr == gt)
    mismatch = both_known & (ocr != gt)

    counts = {
        'Match': int(match.sum()),
        'Mismatch': int(mismatch.sum()),
        'OCR only': int(ocr_only.sum()),
        'GT only': int(gt_only.sum()),
        'Both unknown': int(both_unknown.sum()),
    }
    total = len(dataframe)

    # Build confusion pairs for mismatches
    confusion = {}
    mismatch_idx = dataframe.index[mismatch]
    for idx in mismatch_idx:
        pair = (ocr.loc[idx], gt.loc[idx])
        confusion[pair] = confusion.get(pair, 0) + 1
    top_confused = sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:20]

    # ── Figure ─────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: category bar chart
    cats = list(counts.keys())
    vals = list(counts.values())
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#95a5a6']
    bars = ax1.bar(cats, vals, color=colors, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                 f'{v}\n({v / total * 100:.1f}%)',
                 ha='center', va='bottom', fontsize=8)
    ax1.set_ylabel('Number of patches', fontsize=10)
    ax1.set_title(f'OCR vs Ground Truth Discrepancy (n={total})', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # Right: top confused pairs
    if top_confused:
        pair_labels = [f'{ocr_c}\u2192{gt_c}' for (ocr_c, gt_c), _ in top_confused]
        pair_counts = [c for _, c in top_confused]
        y_pos = np.arange(len(pair_labels))
        ax2.barh(y_pos, pair_counts, color='#e74c3c', alpha=0.7, edgecolor='white')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(pair_labels, fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel('Count', fontsize=10)
        ax2.set_title('Top Confused Pairs (OCR\u2192GT)', fontsize=11)
        ax2.grid(axis='x', alpha=0.3)
        for i, c in enumerate(pair_counts):
            ax2.text(c + max(pair_counts) * 0.01, i, str(c), va='center', fontsize=7)
    else:
        ax2.text(0.5, 0.5, 'No mismatches found', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Top Confused Pairs', fontsize=11)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [Discrepancy] Saved: {output_path}")
    return fig


def generate_discrepancy_table(
    dataframe, label_col='char_consensus', output_path=None,
):
    """Generate a LaTeX table summarising discrepancy statistics.

    Uses ``char_transcription`` as ground truth (see
    :func:`generate_discrepancy_figure` for the rationale).

    Returns the table string and optionally writes it to a .tex file.
    """
    gt_col = 'char_transcription'
    if 'char_chat' not in dataframe.columns or gt_col not in dataframe.columns:
        print(f"  [Discrepancy table] Skipping: missing columns")
        return None

    ocr = dataframe['char_chat'].fillna(UNKNOWN_LABEL)
    gt = dataframe[gt_col].fillna(UNKNOWN_LABEL)

    both_known = (ocr != UNKNOWN_LABEL) & (gt != UNKNOWN_LABEL)
    match = both_known & (ocr == gt)
    mismatch = both_known & (ocr != gt)
    ocr_only = (ocr != UNKNOWN_LABEL) & (gt == UNKNOWN_LABEL)
    gt_only = (ocr == UNKNOWN_LABEL) & (gt != UNKNOWN_LABEL)

    n = len(dataframe)
    n_both = int(both_known.sum())
    n_match = int(match.sum())
    n_mismatch = int(mismatch.sum())
    n_ocr_only = int(ocr_only.sum())
    n_gt_only = int(gt_only.sum())

    accuracy = n_match / n_both * 100 if n_both > 0 else 0

    lines = [
        '% Auto-generated discrepancy statistics',
        r'\begin{table}[t]',
        r'    \centering\small',
        r'    \caption{Discrepancy between OCR predictions and ground-truth labels.}',
        r'    \label{tab:discrepancy}',
        r'    \begin{tabular}{@{}lr@{}}',
        r'        \toprule',
        r'        \textbf{Category} & \textbf{Count (\%)} \\',
        r'        \midrule',
        f'        Total patches & {n:,} \\\\',
        f'        Both labels known & {n_both:,} ({n_both/n*100:.1f}\\%) \\\\',
        f'        \\quad Match (OCR = GT) & {n_match:,} ({n_match/n*100:.1f}\\%) \\\\',
        f'        \\quad Mismatch (OCR $\\neq$ GT) & {n_mismatch:,} ({n_mismatch/n*100:.1f}\\%) \\\\',
        f'        OCR known, GT unknown & {n_ocr_only:,} ({n_ocr_only/n*100:.1f}\\%) \\\\',
        f'        GT known, OCR unknown & {n_gt_only:,} ({n_gt_only/n*100:.1f}\\%) \\\\',
        r'        \midrule',
        f'        OCR accuracy (where both known) & {accuracy:.1f}\\% \\\\',
        r'        \bottomrule',
        r'    \end{tabular}',
        r'\end{table}',
    ]
    content = '\n'.join(lines) + '\n'

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"  [Discrepancy table] Saved: {output_path}")
    return content


# ════════════════════════════════════════════════════════════════════
#  t-SNE cluster visualizations (selected clusters)
# ════════════════════════════════════════════════════════════════════

def generate_tsne_selection(
    dataframe, graph, label_col='char_consensus',
    n_pure=2, n_impure=2, n_large=1,
    output_dir=None, dpi=200,
):
    """Generate t-SNE visualizations for a selection of representative clusters.

    Selects:
      - ``n_pure`` highly pure clusters (purity >= 0.95, size >= 10)
      - ``n_impure`` impure clusters (purity < 0.8, size >= 10)
      - ``n_large`` largest clusters

    Parameters
    ----------
    dataframe : pd.DataFrame
    graph : networkx.Graph
    label_col : str
    n_pure, n_impure, n_large : int
    output_dir : Path or None
    dpi : int
    """
    if graph is None or 'membership' not in dataframe.columns:
        print("  [t-SNE] Skipping: missing graph or membership column")
        return None

    from src.clustering.tsne_plot import plot_community_tsne

    purity_df, _ = compute_cluster_purity(dataframe, 'membership', label_col)
    purity_df = purity_df.dropna(subset=['Purity'])

    selected = set()

    # Pure clusters (high purity, decent size)
    pure = (purity_df[(purity_df['Purity'] >= 0.95) & (purity_df['Size'] >= 10)]
            .sort_values('Size', ascending=False))
    for cid in pure.head(n_pure).index:
        selected.add(('pure', cid))

    # Impure clusters
    impure = (purity_df[(purity_df['Purity'] < 0.8) & (purity_df['Size'] >= 10)]
              .sort_values('Size', ascending=False))
    for cid in impure.head(n_impure).index:
        selected.add(('impure', cid))

    # Largest clusters
    largest = purity_df.sort_values('Size', ascending=False)
    for cid in largest.head(n_large).index:
        selected.add(('large', cid))

    if not selected:
        print("  [t-SNE] No suitable clusters found")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for tag, cid in sorted(selected, key=lambda x: x[1]):
        size = purity_df.loc[cid, 'Size']
        purity = purity_df.loc[cid, 'Purity']

        # Adapt figsize to cluster size
        if size < 30:
            figsize = (10, 8)
        elif size < 100:
            figsize = (16, 13)
        else:
            figsize = (22, 18)

        try:
            fig = plot_community_tsne(
                cluster_id=cid,
                dataframe=dataframe,
                graph=graph,
                target_lbl=label_col,
                figsize=figsize,
                dpi=dpi,
            )
            fname = f'tsne_cluster_{cid}_{tag}.pdf'
            fig.savefig(
                output_dir / fname, dpi=dpi,
                bbox_inches='tight', facecolor='white',
            )
            plt.close(fig)
            saved.append(fname)
            print(f"  [t-SNE] Saved: {fname}  (size={size}, purity={purity:.2f})")
        except Exception as e:
            print(f"  [t-SNE] Failed cluster {cid}: {e}")

    print(f"  [t-SNE] Generated {len(saved)} t-SNE figures in {output_dir}")
    return saved


# ════════════════════════════════════════════════════════════════════
#  Alignment visualization for paper
# ════════════════════════════════════════════════════════════════════

def copy_alignment_viz(
    alignment_viz_dir, output_dir, page_idx=None, dpi=150,
):
    """Copy/convert alignment visualization images for the paper.

    If *page_idx* is None, picks the first available page.

    Parameters
    ----------
    alignment_viz_dir : Path
        Directory containing ``page_NNN.png`` images.
    output_dir : Path
        Paper figures output directory.
    page_idx : int or None
        0-based page index.  If None, auto-selects the first available.
    dpi : int
    """
    alignment_viz_dir = Path(alignment_viz_dir)
    if not alignment_viz_dir.exists():
        print(f"  [Alignment] Skipping: {alignment_viz_dir} does not exist")
        return None

    pages = sorted(alignment_viz_dir.glob('page_*.png'))
    if not pages:
        print(f"  [Alignment] Skipping: no page images in {alignment_viz_dir}")
        return None

    # Select page
    if page_idx is not None:
        target = alignment_viz_dir / f'page_{page_idx:03d}.png'
        if not target.exists():
            print(f"  [Alignment] page_{page_idx:03d}.png not found, using first available")
            target = pages[0]
    else:
        target = pages[0]

    # Read and re-save as PDF for LaTeX inclusion
    from PIL import Image as PILImage
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = PILImage.open(target)
    img_w, img_h = img.size
    fig_w = 16
    fig_h = fig_w * img_h / img_w

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(np.array(img))
    ax.set_axis_off()
    ax.set_title(f'Alignment Visualization \u2014 {target.stem}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    out_path = output_dir / 'alignment_viz.pdf'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  [Alignment] Saved: {out_path}  (source: {target.name})")
    print(f"  [Alignment] Available pages: {[p.stem for p in pages]}")
    return out_path


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate all experiment figures and LaTeX macros.",
    )
    parser.add_argument(
        "--clustering-dir", required=True, type=Path,
        help="Root of clustering results (e.g. results/clustering/book1).",
    )
    parser.add_argument(
        "--preprocessing-dir", type=Path, default=None,
        help="Root of preprocessing results (for F4). If omitted, F4 is skipped.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("paper/figures/generated"),
    )
    parser.add_argument("--label-col", type=str, default="char_consensus")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--alignment-viz-dir", type=Path, default=None,
        help="Path to alignment_viz directory with page_NNN.png images.",
    )
    parser.add_argument(
        "--alignment-page", type=int, default=None,
        help="0-based page index for alignment visualization.",
    )
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    clust_dir = args.clustering_dir

    # ── Load data ─────────────────────────────────────────────────
    print("Loading clustering dataframe...")
    clust_df = load_dataframe(clust_dir / "clustered_patches")
    print(f"  {len(clust_df)} patches loaded")

    graph = None
    graph_path = clust_dir / "graph.gpickle"
    if graph_path.exists():
        print("Loading graph...")
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        print(f"  {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    sweep_df = _try_load_csv(clust_dir / "sweep_results.csv")
    split_sweep_df = _try_load_csv(clust_dir / "split_sweep.csv")

    preproc_df = None
    if args.preprocessing_dir is not None:
        print("Loading preprocessing dataframe...")
        preproc_df = load_dataframe(args.preprocessing_dir)
        print(f"  {len(preproc_df)} patches loaded")

    # ── Generate figures ──────────────────────────────────────────
    print("\nGenerating figures:")

    # F4: Preprocessing before/after
    if preproc_df is not None:
        generate_preprocessing_figure(preproc_df, out / "preprocessing.pdf", dpi=args.dpi)

    # F5: HOG descriptor example
    generate_hog_figure(clust_df, out / "hog_descriptor.pdf", dpi=args.dpi)

    # F6: A contrario distribution
    generate_a_contrario_figure(clust_df, out / "a_contrario.pdf", dpi=args.dpi)

    # F8: Cluster gallery
    generate_cluster_gallery(
        clust_df, label_col=args.label_col,
        output_path=out / "cluster_gallery.pdf", dpi=args.dpi,
    )

    # F9: Epsilon sensitivity
    generate_epsilon_sensitivity(sweep_df, out / "epsilon_sensitivity.pdf", dpi=args.dpi)

    # F10: Split threshold sensitivity
    generate_split_threshold_figure(
        split_sweep_df, out / "split_threshold.pdf", dpi=args.dpi,
    )

    # Discrepancy statistics
    generate_discrepancy_figure(
        clust_df, label_col=args.label_col,
        output_path=out / "discrepancy.pdf", dpi=args.dpi,
    )
    generate_discrepancy_table(
        clust_df, label_col=args.label_col,
        output_path=out / "discrepancy_table.tex",
    )

    # t-SNE cluster visualizations (selected)
    if graph is not None:
        generate_tsne_selection(
            clust_df, graph, label_col=args.label_col,
            output_dir=out / "tsne_clusters", dpi=args.dpi,
        )

    # Alignment visualization
    viz_dir = args.alignment_viz_dir
    if viz_dir is None:
        # Try default locations
        for candidate in [
            clust_dir.parent / 'preprocessing' / clust_dir.name / 'alignment_viz',
            args.preprocessing_dir / 'alignment_viz' if args.preprocessing_dir else None,
        ]:
            if candidate and candidate.exists():
                viz_dir = candidate
                break
    if viz_dir is not None:
        copy_alignment_viz(
            viz_dir, out,
            page_idx=args.alignment_page, dpi=args.dpi,
        )

    # LaTeX macros
    if graph is not None:
        generate_latex_macros(
            clust_df, graph, label_col=args.label_col,
            output_path=out / "experiment_macros.tex",
        )

    # Ablation tables (A1, A2, A4)
    export_ablation_tables(sweep_df, out, dpi=args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()
