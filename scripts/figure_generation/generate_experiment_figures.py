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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

def generate_hog_figure(df, output_path, dpi=300):
    """Render one character through the full HOG pipeline."""
    if 'svg' not in df.columns or 'histogram' not in df.columns:
        print("  [F5] Skipping: missing svg or histogram columns")
        return None

    try:
        import torch
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
    histogram = hog_output.histograms[0, 0].cpu().numpy()

    fig = plt.figure(figsize=(14, 4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5])

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(rendered, cmap='gray')
    ax1.set_title('Rendered character', fontsize=9)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(magnitude, cmap='hot')
    ax2.set_title('Gradient magnitude', fontsize=9)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(orientation, cmap='hsv')
    ax3.set_title('Gradient orientation', fontsize=9)
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[3])
    im = ax4.imshow(histogram, aspect='auto', cmap='viridis', interpolation='nearest')
    ax4.set_xlabel('Orientation bin', fontsize=8)
    ax4.set_ylabel('Cell index', fontsize=8)
    ax4.set_title('HOG descriptor', fontsize=9)
    plt.colorbar(im, ax=ax4, shrink=0.8)

    char = df.loc[idx, 'char_chat'] if 'char_chat' in df.columns else ''
    fig.suptitle(
        f'HOG descriptor computation — character "{char}"',
        fontsize=11, fontweight='bold',
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
