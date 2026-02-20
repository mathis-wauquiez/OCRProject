"""
ClusteringSweepReporter — All figure creation, HTML assembly, and section-based
report generation for the clustering sweep.

Holds a back-reference to the parent sweep object for config access
and AutoReport method delegation.
"""

import io
import base64
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from tqdm import tqdm

from .graph_visu import (
    matches_per_treshold, random_match_figure, size_distribution_figure,
    purity_figure, completeness_figure, report_community, plot_nearest_neighbors,
)
from .tsne_plot import plot_community_tsne

UNKNOWN_LABEL = '\u25af'  # ▯


def _get_b64(fig, dpi=75):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Glyph.*missing from font')
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_str


class ClusteringSweepReporter:
    """
    Encapsulates every reporting / visualisation method for the clustering sweep.

    Parameters
    ----------
    sweep : graphClusteringSweep
        The parent sweep object (inherits AutoReport).
        Used for config values and ``report_*()`` delegation.
    """

    def __init__(self, sweep):
        self.sweep = sweep

    # ── Convenience delegates to AutoReport ──────────────────────────

    def _report(self, *a, **kw):          return self.sweep.report(*a, **kw)
    def _report_figure(self, *a, **kw):   return self.sweep.report_figure(*a, **kw)
    def _report_table(self, *a, **kw):    return self.sweep.report_table(*a, **kw)
    def _report_raw_html(self, *a, **kw): return self.sweep.report_raw_html(*a, **kw)
    def _section(self, *a, **kw):         return self.sweep.section(*a, **kw)

    # ================================================================
    #  Figure saving (external file or embedded base64)
    # ================================================================

    def _save_figure(self, fig, prefix="fig"):
        """Save figure efficiently.  Returns an HTML ``<img>`` tag."""
        self.sweep.image_counter += 1

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Glyph.*missing from font')

            if self.sweep.embed_images:
                buf = io.BytesIO()
                fmt = 'jpeg' if self.sweep.use_jpeg else 'png'
                save_kwargs = {'format': fmt, 'bbox_inches': 'tight',
                               'dpi': self.sweep.image_dpi}
                if self.sweep.use_jpeg:
                    save_kwargs['pil_kwargs'] = {'quality': self.sweep.jpeg_quality}

                fig.savefig(buf, **save_kwargs)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode()
                plt.close(fig)

                mime = 'jpeg' if self.sweep.use_jpeg else 'png'
                return (f'<img src="data:image/{mime};base64,{img_str}" '
                        f'style="max-width: 100%; height: auto;" loading="lazy">')
            else:
                ext = 'jpg' if self.sweep.use_jpeg else 'png'
                filename = f"{prefix}_{self.sweep.image_counter:04d}.{ext}"
                filepath = self.sweep.assets_dir / filename

                save_kwargs = {'format': 'jpeg' if self.sweep.use_jpeg else 'png',
                               'bbox_inches': 'tight', 'dpi': self.sweep.image_dpi}
                if self.sweep.use_jpeg:
                    save_kwargs['pil_kwargs'] = {'quality': self.sweep.jpeg_quality}

                fig.savefig(filepath, **save_kwargs)
                plt.close(fig)

                return (f'<img src="assets_{self.sweep.report_name}/{filename}" '
                        f'style="max-width: 100%; height: auto;" loading="lazy">')

    # ================================================================
    #  HOG Configuration Sweep section
    # ================================================================

    def report_hog_sweep(self, results_df, hog_configs):
        """Best-per-config table, per-config pivot tables, comparison heatmaps."""
        best_per_config = results_df.loc[
            results_df.groupby('hog_config')['adjusted_rand_index'].idxmax()
        ].sort_values('adjusted_rand_index', ascending=False)

        with self._section("HOG Configuration Sweep"):
            self._report_table(best_per_config,
                               title="Best Results per HOG Configuration")

            for config_name in hog_configs:
                subset = results_df[results_df['hog_config'] == config_name]
                metric_names = [
                    c for c in subset.columns
                    if c not in ['hog_config', 'cell_size', 'grdt_sigma',
                                 'num_bins', 'epsilon', 'gamma']
                ]
                for metric in metric_names:
                    pivot = subset.pivot_table(
                        values=metric, index='gamma', columns='epsilon'
                    )
                    self._report_table(
                        pivot, title=f'{metric} (gamma × epsilon) — {config_name}'
                    )

                # Per-config parameter heatmaps (ε × γ grids for all metrics)
                heatmap_subset = subset[['epsilon', 'gamma'] + metric_names]
                fig = self.report_parameter_heatmaps(heatmap_subset)
                self._report_figure(fig, title=f"Parameter Heatmaps — {config_name}")

            fig = self._hog_comparison_heatmaps(results_df)
            self._report_figure(fig, title="HOG Configuration Comparison")

    def _hog_comparison_heatmaps(self, results_df):
        import seaborn as sns

        best_per_config = results_df.loc[
            results_df.groupby('hog_config')['adjusted_rand_index'].idxmax()
        ]
        ari_min = results_df['adjusted_rand_index'].min()
        ari_max = results_df['adjusted_rand_index'].max()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Bar chart
        ax = axes[0, 0]
        bpc = best_per_config.sort_values('adjusted_rand_index', ascending=False)
        ax.barh(range(len(bpc)), bpc['adjusted_rand_index'])
        ax.set_yticks(range(len(bpc)))
        ax.set_yticklabels(bpc['hog_config'])
        ax.set_xlabel('Adjusted Rand Index')
        ax.set_title('Best Performance by HOG Configuration')
        ax.grid(axis='x', alpha=0.3)
        best_bar_pos = list(bpc.index).index(bpc['adjusted_rand_index'].idxmax())
        ax.patches[best_bar_pos].set_color('green')

        # cell_size vs grdt_sigma
        pivot = best_per_config.pivot_table(
            values='adjusted_rand_index', index='grdt_sigma',
            columns='cell_size', aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                    ax=axes[0, 1], vmin=ari_min, vmax=ari_max)
        axes[0, 1].set_title('ARI: Cell Size vs Gradient Sigma')

        # num_bins vs cell_size
        pivot = best_per_config.pivot_table(
            values='adjusted_rand_index', index='num_bins',
            columns='cell_size', aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                    ax=axes[1, 0], vmin=ari_min, vmax=ari_max)
        axes[1, 0].set_title('ARI: Num Bins vs Cell Size')

        # Scatter
        ax = axes[1, 1]
        scatter = ax.scatter(
            best_per_config['epsilon'], best_per_config['gamma'],
            c=best_per_config['adjusted_rand_index'], s=200, cmap='RdYlGn',
            alpha=0.6, edgecolors='black', vmin=ari_min, vmax=ari_max
        )
        for _, row in best_per_config.iterrows():
            ax.annotate(row['hog_config'], (row['epsilon'], row['gamma']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        ax.set_xlabel('Epsilon'); ax.set_ylabel('Gamma')
        ax.set_title('Best ε/γ per HOG Config (colored by ARI)')
        plt.colorbar(scatter, ax=ax, label='ARI')

        plt.tight_layout()
        return fig

    # ================================================================
    #  Parameter heatmaps (single-config ε × γ grids)
    # ================================================================

    def report_parameter_heatmaps(self, results_df):
        import seaborn as sns

        metric_names = [c for c in results_df.columns if c not in ['epsilon', 'gamma']]
        n_metrics = len(metric_names)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), squeeze=False)
        axes = axes.flatten()

        for idx, metric in enumerate(metric_names):
            pivot = results_df.pivot_table(values=metric, index='gamma',
                                           columns='epsilon')
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                        ax=axes[idx], cbar_kws={'label': metric})
            axes[idx].set_title(f'{metric} vs Parameters', fontweight='bold')
            axes[idx].set_xlabel('ε'); axes[idx].set_ylabel('γ')

        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig

    # ================================================================
    #  Graph topology section
    # ================================================================

    def report_graph_topology(self, graph, best_epsilon, best_gamma):
        """Summary card + degree distribution for the similarity graph."""
        import networkx as nx

        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        density = nx.density(graph)
        n_components = nx.number_connected_components(graph)
        isolated = nx.number_of_isolates(graph)
        degrees = [d for _, d in graph.degree()]
        avg_degree = np.mean(degrees) if degrees else 0.0

        topo_html = f"""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                    color: white; padding: 30px; border-radius: 10px; margin: 20px 0;">
            <h2 style="margin: 0 0 15px 0;">Graph Topology</h2>
            <p>Best parameters: &epsilon;={best_epsilon:.4f}, &gamma;={best_gamma:.4f}</p>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
                <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 5px; text-align:center;">
                    <h4 style="margin:0;">Nodes</h4>
                    <p style="font-size:1.8em; margin:5px 0;">{n_nodes:,}</p></div>
                <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 5px; text-align:center;">
                    <h4 style="margin:0;">Edges</h4>
                    <p style="font-size:1.8em; margin:5px 0;">{n_edges:,}</p></div>
                <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 5px; text-align:center;">
                    <h4 style="margin:0;">Density</h4>
                    <p style="font-size:1.8em; margin:5px 0;">{density:.4f}</p></div>
                <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 5px; text-align:center;">
                    <h4 style="margin:0;">Components</h4>
                    <p style="font-size:1.8em; margin:5px 0;">{n_components:,}</p></div>
                <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 5px; text-align:center;">
                    <h4 style="margin:0;">Isolated</h4>
                    <p style="font-size:1.8em; margin:5px 0;">{isolated:,}</p></div>
                <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 5px; text-align:center;">
                    <h4 style="margin:0;">Avg Degree</h4>
                    <p style="font-size:1.8em; margin:5px 0;">{avg_degree:.1f}</p></div>
            </div>
        </div>
        """
        self._report_raw_html(topo_html, title="Graph Topology Summary")

        # Degree distribution histogram
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.hist(degrees, bins=min(50, max(degrees) + 1) if degrees else 1,
                color='#11998e', edgecolor='white', alpha=0.8)
        ax.set_xlabel('Degree')
        ax.set_ylabel('Count')
        ax.set_title('Degree Distribution')
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.hist(degrees, bins=min(50, max(degrees) + 1) if degrees else 1,
                color='#38ef7d', edgecolor='white', alpha=0.8)
        ax.set_xlabel('Degree')
        ax.set_ylabel('Count (log scale)')
        ax.set_title('Degree Distribution (Log Scale)')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        self._report_figure(fig, title="Degree Distribution")

    # ================================================================
    #  Executive summary card
    # ================================================================

    def report_executive_summary(self, dataframe, purity_dataframe,
                                 label_dataframe, best_epsilon, best_gamma,
                                 best_split_threshold=None):
        n_clusters = len(purity_dataframe)
        avg_purity = purity_dataframe['Purity'].mean()
        avg_size   = purity_dataframe['Size'].mean()
        n_hapax    = (purity_dataframe['Size'] == 1).sum()

        best_cluster    = purity_dataframe['Purity'].idxmax()
        worst_cluster   = purity_dataframe['Purity'].idxmin()
        largest_cluster = purity_dataframe['Size'].idxmax()

        summary_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 30px; border-radius: 10px; margin: 20px 0;">
            <h1 style="margin: 0 0 20px 0;">📊 Executive Summary</h1>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 20px;">
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 5px;">
                    <h3 style="margin: 0;">Clusters</h3>
                    <p style="font-size: 2em; margin: 10px 0;">{n_clusters}</p>
                    <small>{n_hapax} singletons ({n_hapax/n_clusters*100:.1f}%)</small>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 5px;">
                    <h3 style="margin: 0;">Avg Purity</h3>
                    <p style="font-size: 2em; margin: 10px 0;">{avg_purity:.2%}</p>
                    <small>Range: {purity_dataframe['Purity'].min():.1%} – {purity_dataframe['Purity'].max():.1%}</small>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 5px;">
                    <h3 style="margin: 0;">Avg Size</h3>
                    <p style="font-size: 2em; margin: 10px 0;">{avg_size:.1f}</p>
                    <small>Largest: {purity_dataframe['Size'].max()} patches</small>
                </div>
            </div>
            <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 5px;">
                <h3>Key Findings</h3>
                <ul style="margin: 10px 0;">
                    <li>Best Parameters: ε={best_epsilon:.4f}, γ={best_gamma:.4f}, split_threshold={best_split_threshold}</li>
                    <li>Purest Cluster: #{best_cluster} ({purity_dataframe.loc[best_cluster, 'Purity']:.1%} purity)</li>
                    <li>Most Scattered: #{worst_cluster} ({purity_dataframe.loc[worst_cluster, 'Purity']:.1%} purity)</li>
                    <li>Largest Cluster: #{largest_cluster} ({purity_dataframe.loc[largest_cluster, 'Size']} patches)</li>
                </ul>
            </div>
        </div>
        """
        self._report_raw_html(summary_html, title="Executive Summary")

    # ================================================================
    #  Split threshold sweep
    # ================================================================

    def report_split_sweep(self, sweep_results_df, best_threshold):
        self._report_table(
            sweep_results_df.set_index('split_threshold'),
            title="Split Threshold Sweep — All Metrics"
        )

        _keep = {
            'adjusted_rand_index', 'normalized_mutual_info', 'adjusted_mutual_info',
            'homogeneity', 'completeness', 'purity', 'v_measure',
        }
        metric_cols = [c for c in sweep_results_df.columns if c in _keep]

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        ax = axes[0]
        for col in metric_cols:
            ax.plot(sweep_results_df['split_threshold'], sweep_results_df[col],
                    marker='o', label=col, markersize=4)
        ax.axvline(best_threshold, color='red', ls='--', alpha=0.7,
                   label=f'best = {best_threshold}')
        ax.set_xlabel('Split Threshold'); ax.set_ylabel('Score')
        ax.set_title('Clustering Metrics vs Split Threshold')
        ax.legend(fontsize=7, loc='best'); ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(sweep_results_df['split_threshold'],
                sweep_results_df['n_clusters_post_split'],
                marker='s', color='#667eea', label='total clusters')
        ax.plot(sweep_results_df['split_threshold'],
                sweep_results_df['n_clusters_actually_split'],
                marker='^', color='#f5576c', label='clusters that were split')
        ax.axvline(best_threshold, color='red', ls='--', alpha=0.7)
        ax.set_xlabel('Split Threshold'); ax.set_ylabel('Count')
        ax.set_title('Number of Clusters vs Split Threshold')
        ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout()
        self._report_figure(fig, title="Split Threshold Sweep")

    # ================================================================
    #  Before / after split comparison
    # ================================================================

    def report_split_comparison(self, dataframe, pre_membership, post_membership,
                                split_log, best_threshold):
        split_log_df = pd.DataFrame(split_log)

        n_pre   = len(np.unique(pre_membership))
        n_post  = len(np.unique(post_membership))
        n_split = int((split_log_df['n_subclusters'] > 1).sum())
        max_sub = int(split_log_df['n_subclusters'].max())

        summary_html = f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white; padding: 30px; border-radius: 10px; margin: 20px 0;">
            <h2 style="margin: 0 0 15px 0;">🔪 Cluster Splitting Summary</h2>
            <p>Best threshold: <strong>{best_threshold}</strong> |
               Linkage: <strong>{self.sweep.split_linkage_method}</strong> |
               Swept {len(self.sweep.split_thresholds)} threshold(s)</p>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px;">
                <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 5px; text-align:center;">
                    <h4 style="margin:0;">Before</h4><p style="font-size:1.8em; margin:5px 0;">{n_pre}</p><small>clusters</small></div>
                <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 5px; text-align:center;">
                    <h4 style="margin:0;">After</h4><p style="font-size:1.8em; margin:5px 0;">{n_post}</p><small>clusters</small></div>
                <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 5px; text-align:center;">
                    <h4 style="margin:0;">Split</h4><p style="font-size:1.8em; margin:5px 0;">{n_split}</p><small>clusters split</small></div>
                <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 5px; text-align:center;">
                    <h4 style="margin:0;">Max Split</h4><p style="font-size:1.8em; margin:5px 0;">{max_sub}</p><small>sub-clusters</small></div>
            </div>
        </div>
        """
        self._report_raw_html(summary_html, title="Cluster Splitting Summary")

        actually_split = split_log_df[split_log_df['n_subclusters'] > 1].sort_values(
            'n_subclusters', ascending=False
        )
        if len(actually_split) > 0:
            self._report_table(actually_split, title="Clusters That Were Split")

        # Size distributions
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, sizes, color, title in [
            (axes[0], pd.Series(pre_membership).value_counts().values,
             '#667eea', f'Before splitting ({n_pre} clusters)'),
            (axes[1], pd.Series(post_membership).value_counts().values,
             '#f5576c', f'After splitting ({n_post} clusters)'),
        ]:
            ax.hist(sizes, bins=50, color=color, edgecolor='white', alpha=0.8)
            ax.set_title(title)
            ax.set_xlabel('Cluster size'); ax.set_ylabel('Count')
            ax.set_yscale('log')
        plt.tight_layout()
        self._report_figure(fig, title="Size Distribution: Before vs After Splitting")

        # Sub-cluster count distribution
        fig, ax = plt.subplots(figsize=(10, 4))
        sc = split_log_df['n_subclusters'].value_counts().sort_index()
        ax.bar(sc.index, sc.values, color='#667eea', edgecolor='white')
        ax.set_xlabel('Number of sub-clusters')
        ax.set_ylabel('Number of original clusters')
        ax.set_title('Distribution of Split Degree')
        ax.set_xticks(sc.index)
        plt.tight_layout()
        self._report_figure(fig, title="How Many Sub-Clusters Per Original Cluster")

        # Metrics comparison
        _to_list = lambda m: m if isinstance(m, list) else m.tolist()
        pre_metrics = self.sweep._evaluate_membership(
            target_labels=dataframe[self.sweep.target_lbl],
            membership=_to_list(pre_membership)
        )
        post_metrics = self.sweep._evaluate_membership(
            target_labels=dataframe[self.sweep.target_lbl],
            membership=_to_list(post_membership)
        )
        self._report_table(
            pd.DataFrame({'Before Split': pre_metrics, 'After Split': post_metrics}).T,
            title="Clustering Metrics: Before vs After Splitting"
        )

    # ================================================================
    #  Split visualization (thumbnails per sub-cluster)
    # ================================================================

    def report_split_visualization(self, dataframe, split_log, best_threshold):
        split_log_df = pd.DataFrame(split_log)
        actually_split = split_log_df[split_log_df['n_subclusters'] > 1].sort_values(
            'original_size', ascending=False
        )
        if len(actually_split) == 0:
            self._report_raw_html('<p>No clusters were split at this threshold.</p>',
                                  title="Split Visualization")
            return

        max_thumbs = 12
        target_lbl = self.sweep.target_lbl
        vis_html = '<div style="display: grid; gap: 40px;">'

        for _, row in tqdm(actually_split.iterrows(), total=len(actually_split),
                           desc="Split visualization", colour="cyan"):
            orig_cid, n_sub, orig_size = (
                int(row['original_cluster']),
                int(row['n_subclusters']),
                int(row['original_size']),
            )
            orig_df = dataframe[dataframe['membership_pre_split'] == orig_cid]

            vis_html += f'''
            <div style="border: 2px solid #667eea; border-radius: 10px; padding: 20px; background: #fafbff;">
                <h3 style="color: #667eea; margin-top: 0;">
                    Leiden Cluster {orig_cid} → {n_sub} sub-clusters
                    <span style="font-weight:normal; font-size:0.8em; color:#888;">(orig size: {orig_size})</span>
                </h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px;">
            '''

            for sub_id, sub_df in orig_df.groupby('membership'):
                sub_size = len(sub_df)
                known = sub_df[sub_df[target_lbl].fillna(UNKNOWN_LABEL) != UNKNOWN_LABEL]
                dom_label = known[target_lbl].value_counts().index[0] if len(known) else UNKNOWN_LABEL
                sub_purity = (known[target_lbl].value_counts().iloc[0] / len(known)
                              if len(known) else float('nan'))

                sample = sub_df.head(max_thumbs)
                thumbs = ''.join(
                    f'<div style="display:inline-block; text-align:center; margin:2px; '
                    f'border:1px solid #ddd; border-radius:4px; padding:3px; background:white;">'
                    f'<div style="width:40px; height:40px; display:flex; align-items:center; '
                    f'justify-content:center;">{srow["svg"].to_string()}</div>'
                    f'<div style="font-size:9px; color:#666;">{srow[target_lbl]}</div></div>'
                    for _, srow in sample.iterrows()
                )

                pur_str = f"{sub_purity:.0%}" if not np.isnan(sub_purity) else "?"
                color = ('#27ae60' if (not np.isnan(sub_purity) and sub_purity >= 0.9) else
                         '#f39c12' if (not np.isnan(sub_purity) and sub_purity >= 0.5) else '#e74c3c')
                overflow = (f'<div style="font-size:10px; color:#999; margin-top:4px;">… and '
                            f'{sub_size - len(sample)} more</div>' if sub_size > len(sample) else '')

                vis_html += f'''
                <div style="border:1px solid #ddd; border-radius:8px; padding:10px; background:white; border-left:4px solid {color};">
                    <div style="font-weight:bold; margin-bottom:5px;">Sub-cluster {sub_id}
                        <span style="font-weight:normal; color:#888; font-size:0.85em;">n={sub_size} | dom="{dom_label}" | pur={pur_str}</span></div>
                    <div style="display:flex; flex-wrap:wrap; gap:2px;">{thumbs}</div>{overflow}
                </div>'''

            vis_html += '</div></div>'
        vis_html += '</div>'
        self._report_raw_html(vis_html, title=f"Split Visualization (threshold={best_threshold})")

    # ================================================================
    #  Per-cluster section (representatives + t-SNE)
    # ================================================================

    def report_clusters_section(self, dataframe, membership_col, purity_dataframe,
                                representatives, graph, title_prefix):
        target_lbl = self.sweep.target_lbl

        if membership_col != 'membership':
            df_view = dataframe.copy(deep=False)
            df_view['membership'] = dataframe[membership_col]
        else:
            df_view = dataframe

        cluster_sizes = df_view.groupby('membership').size().sort_values(ascending=False)
        clusters_html = '<div style="display: grid; gap: 30px;">'
        min_cluster_size = 2

        for cluster in tqdm(cluster_sizes[cluster_sizes >= min_cluster_size].index,
                            desc=f"Reporting clusters ({title_prefix})", colour='magenta'):
            cluster_df    = df_view[df_view['membership'] == cluster]
            cluster_stats = purity_dataframe.loc[cluster]
            label_reps    = representatives[cluster]

            fig = report_community(cluster, cluster_stats, cluster_df, label_reps, target_lbl)
            img_tag = self._save_figure(fig, prefix="cluster")

            purity_display = (f"{cluster_stats['Purity']:.2%}"
                              if not np.isnan(cluster_stats['Purity']) else "N/A")

            clusters_html += f'''
            <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea; margin-top: 0;">Cluster {cluster} – Representatives</h3>
                <p><strong>Size:</strong> {cluster_stats["Size"]} | <strong>Purity:</strong> {purity_display}</p>
                {img_tag}
            '''

            if len(cluster_df) >= 10:
                fig = plot_community_tsne(cluster_id=cluster, dataframe=df_view,
                                          graph=graph, target_lbl=target_lbl)
                clusters_html += f'''
                    <h4 style="color: #667eea; margin-top: 20px;">t-SNE Visualization</h4>
                    {self._save_figure(fig, prefix="tsne")}
                '''
            clusters_html += '</div>'

        clusters_html += '</div>'
        self._report_raw_html(clusters_html,
                              title=f"{title_prefix} ({len(cluster_sizes)} clusters)")

    # ================================================================
    #  Summary & Metrics section
    # ================================================================

    def report_summary_and_metrics(self, dataframe, purity_dataframe, label_dataframe,
                                   best_metrics, nlfa, best_epsilon, best_gamma,
                                   best_threshold):
        """Returns hapax_df (singleton-cluster rows)."""
        target_lbl = self.sweep.target_lbl

        best_metrics_df = pd.DataFrame([{
            'epsilon': best_epsilon, 'gamma': best_gamma,
            'split_threshold': best_threshold, **best_metrics
        }])

        cluster_sizes  = dataframe['membership'].value_counts()
        hapax_clusters = cluster_sizes[cluster_sizes == 1].index
        hapax_df = dataframe[dataframe['membership'].isin(hapax_clusters)]

        with self._section("Summary & Metrics"):
            self._report_table(
                best_metrics_df.T,
                title=f'Best Parameters (ε={best_epsilon:.4f}, γ={best_gamma:.4f}, split_t={best_threshold})'
            )
            self.report_executive_summary(
                dataframe, purity_dataframe, label_dataframe,
                best_epsilon, best_gamma, best_split_threshold=best_threshold
            )
            self._report(matches_per_treshold(nlfa, best_epsilon),
                         title="Average number of matches = f(epsilon)")
            self._report(size_distribution_figure(dataframe['membership'],
                                                  dataframe[target_lbl]),
                         title="Cluster Size Distribution")
            self._report_figure(purity_figure(purity_dataframe),
                                title="Purity of the clusters")
            self._report_figure(completeness_figure(label_dataframe),
                                title="Completeness")

            self._report_table(pd.DataFrame({
                'Count': [len(hapax_df)],
                'Percentage': [100 * len(hapax_df) / len(dataframe)],
                'Unique_Labels': [hapax_df[target_lbl].nunique()]
            }), title="Hapax (Singleton Clusters)")

            unknown_df = dataframe[dataframe[target_lbl] == UNKNOWN_LABEL]
            self._report_table(pd.DataFrame({
                'Total Unknown': [len(unknown_df)],
                'Percentage': [100 * len(unknown_df) / len(dataframe)],
                'Clusters Containing Unknown': [unknown_df['membership'].nunique()],
                'Pure Unknown Clusters': [
                    (dataframe.groupby('membership')[target_lbl]
                     .apply(lambda x: (x == UNKNOWN_LABEL).all())).sum()
                ]
            }), title="Unknown Character Statistics (▯)")

        return hapax_df

    # ================================================================
    #  Examples section
    # ================================================================

    def report_examples(self, dataframe, dissimilarities, graph, best_epsilon):
        rng = np.random.RandomState(42)
        with self._section("Examples"):
            # Nearest-neighbor examples
            nn_html = '<div style="display: grid; gap: 30px;">'
            for i in tqdm(range(30), desc="Random NN examples", colour='cyan'):
                idx = rng.randint(len(dataframe))
                fig = plot_nearest_neighbors(query_idx=idx, dataframe=dataframe,
                                             dissimilarities=dissimilarities,
                                             graph=graph, n_to_show=23)
                nn_html += f'''
                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #667eea; margin-top: 0;">Example {i+1}: NN for Sample {idx}</h3>
                    {self._save_figure(fig, prefix="nn_example")}
                </div>'''
            nn_html += '</div>'
            self._report_raw_html(nn_html, title="Random Nearest Neighbor Examples (30 samples)")

            # Random match examples
            match_html = '<div style="display: grid; gap: 30px;">'
            for i in tqdm(range(30), desc="Random match figures", colour='yellow'):
                fig1, fig2, idx = random_match_figure(
                    self.sweep.featureMatcher, dataframe['histogram'],
                    best_epsilon, svgs=dataframe['svg']
                )
                match_html += f'''
                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #667eea; margin-top: 0;">Example {i+1}: Sample {idx}</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div><h4>Distribution</h4>
                            <img src="data:image/png;base64,{_get_b64(fig1)}" style="max-width: 100%; height: auto;"></div>
                        <div><h4>Matches</h4>
                            <img src="data:image/png;base64,{_get_b64(fig2)}" style="max-width: 100%; height: auto;"></div>
                    </div>
                </div>'''
            match_html += '</div>'
            self._report_raw_html(match_html, title="Random Match Examples (30 samples)")

    # ================================================================
    #  Hapax analysis section
    # ================================================================

    def report_hapax(self, dataframe, hapax_df, dissimilarities, graph):
        target_lbl = self.sweep.target_lbl
        rng = np.random.RandomState(42)

        with self._section("Hapax Analysis"):
            idxs = list(rng.permutation(hapax_df.index))[:10]
            hapax_nn_html = '<div style="display: grid; gap: 30px;">'
            for idx in tqdm(idxs, desc="Hapax NN examples", colour='red'):
                fig = plot_nearest_neighbors(query_idx=idx, dataframe=dataframe,
                                             dissimilarities=dissimilarities,
                                             graph=graph, n_to_show=23)
                hapax_nn_html += f'''
                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #667eea; margin-top: 0;">Hapax Example – Sample {idx}</h3>
                    <img src="data:image/png;base64,{_get_b64(fig)}" style="max-width: 100%; height: auto;">
                </div>'''
            hapax_nn_html += '</div>'
            self._report_raw_html(hapax_nn_html,
                                  title="Hapax Nearest Neighbor Examples (10 samples)")

            # All hapax thumbnails
            parts = ['<div style="display: grid; grid-template-columns: '
                     'repeat(auto-fill, minmax(120px, 1fr)); gap: 15px;">']
            for idx, row in hapax_df.iterrows():
                parts.append(f'''
                <div style="border: 1px solid #ddd; padding: 10px; text-align: center; border-radius: 5px;">
                    <div style="width: 100px; height: 100px; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                        {row["svg"].to_string()}
                    </div>
                    <div style="margin-top: 5px; font-size: 11px; font-weight: bold;">{row[target_lbl]}</div>
                    <div style="font-size: 9px; color: #666;">idx: {idx}</div>
                </div>''')
            parts.append('</div>')
            self._report_raw_html(''.join(parts), title=f"All Hapax ({len(hapax_df)} items)")

    # ================================================================
    #  Master orchestrator
    # ================================================================

    def report_graph_results(self, *, dataframe, graph, partition, nlfa,
                             dissimilarities, best_epsilon, best_gamma,
                             pre_split_membership, post_split_membership,
                             best_threshold, best_split_log,
                             sweep_results_df, did_split,
                             purity_dataframe, representatives,
                             pre_purity_df, pre_representatives,
                             best_metrics, label_dataframe):
        """
        Single entry-point producing every report section for a given
        graph + partition.  Called once ALL computation is done.
        """
        # Section 0: Graph topology + best-config summary
        with self._section("Graph Topology"):
            self.report_graph_topology(graph, best_epsilon, best_gamma)

            # Best-config parameter summary for quick reference
            config_html = f"""
            <div style="background: #f8f9fa; border-left: 4px solid #667eea;
                        padding: 15px; border-radius: 4px; margin: 15px 0;">
                <h3 style="margin: 0 0 10px 0; color: #667eea;">Best Configuration</h3>
                <table style="border-collapse: collapse; width: auto;">
                    <tr><td style="padding: 4px 12px; font-weight:bold;">Epsilon (&epsilon;)</td>
                        <td style="padding: 4px 12px;">{best_epsilon:.4f}</td></tr>
                    <tr><td style="padding: 4px 12px; font-weight:bold;">Gamma (&gamma;)</td>
                        <td style="padding: 4px 12px;">{best_gamma:.4f}</td></tr>
                    <tr><td style="padding: 4px 12px; font-weight:bold;">Split Threshold</td>
                        <td style="padding: 4px 12px;">{best_threshold}</td></tr>
                    <tr><td style="padding: 4px 12px; font-weight:bold;">Best ARI</td>
                        <td style="padding: 4px 12px;">{best_metrics.get('adjusted_rand_index', 'N/A')}</td></tr>
                </table>
            </div>
            """
            self._report_raw_html(config_html, title="Best Configuration Parameters")

        # Section 1: Pre-split clusters
        with self._section("Clusters (Pre-Split)"):
            self.report_clusters_section(
                dataframe, 'membership_pre_split', pre_purity_df,
                pre_representatives, graph,
                title_prefix="All Clusters Analysis (Before Splitting)"
            )

        # Section 2: Cluster splitting
        if did_split:
            with self._section("Cluster Splitting"):
                if len(self.sweep.split_thresholds) > 1:
                    self.report_split_sweep(sweep_results_df, best_threshold)
                self.report_split_comparison(
                    dataframe, pre_split_membership, post_split_membership,
                    best_split_log, best_threshold
                )
                self.report_split_visualization(dataframe, best_split_log, best_threshold)
                self.report_clusters_section(
                    dataframe, 'membership', purity_dataframe,
                    representatives, graph,
                    title_prefix="All Clusters Analysis After Splitting"
                )

        # Section 3: Summary & metrics
        hapax_df = self.report_summary_and_metrics(
            dataframe, purity_dataframe, label_dataframe,
            best_metrics, nlfa, best_epsilon, best_gamma, best_threshold
        )

        # Section 4: Examples
        self.report_examples(dataframe, dissimilarities, graph, best_epsilon)

        # Section 5: Hapax
        self.report_hapax(dataframe, hapax_df, dissimilarities, graph)