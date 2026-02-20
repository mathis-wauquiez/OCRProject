
import torch
import numpy as np

import matplotlib.pyplot as plt
UNKNOWN_LABEL = '▯'  # U+25AF - represents unrecognized characters

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'sans-serif']

def matches_per_threshold(nlfa, best_eps):
    N = nlfa.shape[0]
    nlfa_threshold = -np.log(best_eps) + 2 * np.log(N)

    hist, bin_edges = torch.histogram(nlfa.reshape(-1).cpu(), bins=300)
    cumhist = torch.flip(hist, dims=[0]).cumsum(0).flip(0)
    avrg_neighbors = cumhist / N

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Plot cumulative on left axis
    color1 = 'steelblue'
    ax1.bar(bin_centers, avrg_neighbors, width=bin_widths, 
            color=color1, alpha=0.6, label='Avg edges/node')
    ax1.set_xlabel('NLFA Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Edges per Node', color=color1, 
                fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 150)
    
    # Plot histogram on right axis
    ax2 = ax1.twinx()
    color2 = 'darkgreen'
    ax2.plot(bin_centers, torch.log(hist + 1), color=color2, 
            linewidth=2, alpha=0.8, label='Log density')
    ax2.set_ylabel('Log Count', color=color2, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Threshold line
    ax1.axvline(nlfa_threshold, color='orangered', linewidth=2.5,
                linestyle='--', label=f'Threshold = {nlfa_threshold:.2f}', zorder=10)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.grid(True, alpha=0.3)
    plt.title('NLFA Threshold Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def random_match_figure(featureMatcher, features, best_epsilon, svgs):

    features = torch.tensor(features, device='cuda')

    NBINS = 50
    N = len(features)
    idx = np.random.randint(N)
    queries = features[idx :idx +1]
    dissim = featureMatcher.compute_dissimilarities(queries, features)

    # show the distribution
    total_dissim = dissim.sum(-1).reshape(-1).cpu()
    mu_tot   = dissim.mean(dim=1).sum(dim=1).item()
    var_tot  = dissim.var(dim=1).sum(dim=1).item()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # -- Top-left plot: distribution of dissimarities & gaussian approx. --
    ax = axes[0]
    ax.hist(total_dissim.numpy(), bins=NBINS, density=True, alpha=0.7, label='Empirical')

    # add Gaussian overlay
    x = np.linspace(total_dissim.min(), total_dissim.max(), 200)
    gaussian = (1 / np.sqrt(2 * np.pi * var_tot)) * np.exp(-0.5 * ((x - mu_tot) / np.sqrt(var_tot))**2)
    ax.plot(x, gaussian, 'r-', linewidth=2, label=f'Gaussian fit')

    # add the threshold
    nlfa_threshold = -np.log(best_epsilon) + 2 * np.log(N)

    # NLFA threshold --> dissimilarity threshold
    from scipy.stats import norm
    dissim_threshold = mu_tot + np.sqrt(var_tot) * norm.ppf(np.exp(-nlfa_threshold))

    ax.axvline(dissim_threshold, color='g', linestyle='--', linewidth=2, 
        label=f'Threshold (ε={best_epsilon:.0e})\nD={dissim_threshold:.3f}')

    ax.legend()
    ax.set_xlabel('Total dissimilarity')
    ax.set_ylabel('Density')
    ax.set_title('Dissimilarity distribution')

    # Cumulative histogram (log scale) for tail visualization
    ax = axes[1]
    hist, bin_edges = torch.histogram(total_dissim, bins=NBINS)
    cumhist = torch.log(1+hist.cumsum(0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    n_under_threshold = (total_dissim <= dissim_threshold).sum().item()

    ax.bar(bin_centers.numpy(), cumhist.numpy(), width=bin_widths.numpy(), 
        edgecolor='black', alpha=0.7, align='center', label='Cumulative count')
    ax.axvline(dissim_threshold, color='g', linestyle='--', linewidth=2, 
        label=f'Threshold\nD={dissim_threshold:.3f}\nN≤threshold={n_under_threshold}')

    ax.legend()
    ax.set_xlabel('Total dissimilarity')
    ax.set_ylabel('Cumulative count (log scale)')
    ax.set_title('Cumulative histogram')

    ax = axes[2]
    ax.imshow(svgs[idx].render(scale=2), cmap='gray')


    plt.tight_layout()

    ################# ============ ##############

    # Show elements under the threshold (limit to 20), sorted by distance
    under_threshold_mask = total_dissim <= dissim_threshold
    under_threshold_indices = torch.where(under_threshold_mask)[0]
    under_threshold_dissim = total_dissim[under_threshold_mask]

    # Sort by dissimilarity
    sorted_indices = torch.argsort(under_threshold_dissim)
    sorted_under_threshold_indices = under_threshold_indices[sorted_indices]
    sorted_dissim_values = under_threshold_dissim[sorted_indices]

    n_to_show = min(20, len(sorted_under_threshold_indices))

    fig2, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(n_to_show):
        idx = sorted_under_threshold_indices[i].item()
        dissim_value = sorted_dissim_values[i].item()
        
        patch_img = svgs[idx].render()
        axes[i].imshow(patch_img, cmap='gray')
        axes[i].set_title(f'Idx {idx}\nD={dissim_value:.3f}')
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(n_to_show, 20):
        axes[i].axis('off')

    plt.suptitle(f'Query patch (idx={idx}) vs matches under threshold | Showing {n_to_show} out of {len(sorted_under_threshold_indices)}', fontsize=14)
    plt.tight_layout()

    return fig, fig2, idx
    

def size_distribution_figure(membership, label):
    cluster_sizes = membership.value_counts()
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # 1. Cluster size distribution
    axes[0].hist(cluster_sizes.values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(cluster_sizes.mean(), color='red', linestyle='--', 
                    label=f'Mean: {cluster_sizes.mean():.1f}')
    axes[0].set_xlabel('Cluster Size')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Cluster Sizes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Label frequency distribution
    label_sizes = label.value_counts()
    axes[1].hist(label_sizes.values, bins=50, alpha=0.7, color='salmon', edgecolor='black')
    axes[1].axvline(label_sizes.mean(), color='red', linestyle='--', 
                    label=f'Mean: {label_sizes.mean():.1f}')
    axes[1].set_xlabel('Label Frequency')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Label Frequencies')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    return fig


def purity_figure(purity_dataframe):
    fig, axes = plt.subplots(4, 1, figsize=(16, 18))

    # 1. Purity distribution
    axes[0].hist(purity_dataframe['Purity'], bins=20, alpha=0.7, color='mediumseagreen', edgecolor='black')
    axes[0].axvline(purity_dataframe['Purity'].mean(), color='red', linestyle='--',
                    label=f'Mean: {purity_dataframe["Purity"].mean():.3f}')
    axes[0].axvline(1.0, color='green', linestyle=':', alpha=0.5, label='Perfect purity')
    axes[0].set_xlabel('Label dominance')
    axes[0].set_ylabel('Number of Clusters')
    axes[0].set_title('Distribution of label dominance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Purity vs Cluster Size (scatter plot)
    scatter = axes[1].scatter(purity_dataframe['Size'], purity_dataframe['Purity'], 
                                alpha=0.6, c=purity_dataframe['Unique_Labels'], 
                                cmap='viridis', edgecolors='black', linewidth=0.5, s=30)
    axes[1].set_xlabel('Cluster Size (number of samples)')
    axes[1].set_ylabel('% of dominance')
    axes[1].set_title('% of dominance vs Cluster Size')
    axes[1].grid(True, alpha=0.3)

    # Add colorbar for number of unique labels
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('Number of Unique Labels in Cluster')

    # 3. Entropy vs Cluster Size (scatter plot)
    scatter = axes[2].scatter(purity_dataframe['Size'], purity_dataframe['Entropy'], 
                                alpha=0.6, c=purity_dataframe['Unique_Labels'], 
                                cmap='viridis', edgecolors='black', linewidth=0.5, s=30)
    axes[2].set_xlabel('Cluster Size (number of samples)')
    axes[2].set_ylabel('Entropy')
    axes[2].set_title('Entropy vs Cluster Size')
    axes[2].grid(True, alpha=0.3)

    # Add colorbar for number of unique labels
    cbar = plt.colorbar(scatter, ax=axes[2])
    cbar.set_label('Number of Unique Labels in Cluster')


    # 4. Normalized Entropy vs Cluster Size (scatter plot)
    scatter = axes[3].scatter(purity_dataframe['Size'], purity_dataframe['Normalized entropy'], 
                                alpha=0.6, c=purity_dataframe['Unique_Labels'], 
                                cmap='viridis', edgecolors='black', linewidth=0.5, s=30)
    axes[3].set_xlabel('Cluster Size (number of samples)')
    axes[3].set_ylabel('Normalized entropy')
    axes[3].set_title('Normalized entropy vs Cluster Size')
    axes[3].grid(True, alpha=0.3)

    # Add colorbar for number of unique labels
    cbar = plt.colorbar(scatter, ax=axes[3])
    cbar.set_label('Number of Unique Labels in Cluster')

    return fig

def completeness_figure(label_dataframe):
    fig, axes = plt.subplots(4, 1, figsize=(16, 18))

    # 1. Best share distribution (analogous to purity)
    axes[0].hist(label_dataframe['Best share'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(label_dataframe['Best share'].mean(), color='red', linestyle='--',
                    label=f'Mean: {label_dataframe["Best share"].mean():.3f}')
    axes[0].axvline(1.0, color='green', linestyle=':', alpha=0.5, label='Perfect completeness')
    axes[0].set_xlabel('Cluster dominance')
    axes[0].set_ylabel('Number of Labels')
    axes[0].set_title('Distribution of cluster dominance per label')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Best share vs Label Size (scatter plot)
    scatter = axes[1].scatter(label_dataframe['Size'], label_dataframe['Best share'], 
                              alpha=0.6, c=label_dataframe['Unique_Clusters'], 
                              cmap='plasma', edgecolors='black', linewidth=0.5, s=30)
    axes[1].set_xlabel('Label Size (number of samples)')
    axes[1].set_ylabel('% of cluster dominance')
    axes[1].set_title('% of cluster dominance vs Label Size')
    axes[1].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('Number of Unique Clusters for Label')

    # 3. Entropy vs Label Size (scatter plot)
    scatter = axes[2].scatter(label_dataframe['Size'], label_dataframe['Entropy'], 
                              alpha=0.6, c=label_dataframe['Unique_Clusters'], 
                              cmap='plasma', edgecolors='black', linewidth=0.5, s=30)
    axes[2].set_xlabel('Label Size (number of samples)')
    axes[2].set_ylabel('Entropy')
    axes[2].set_title('Entropy vs Label Size')
    axes[2].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[2])
    cbar.set_label('Number of Unique Clusters for Label')

    # 4. Normalized Entropy vs Label Size (scatter plot)
    scatter = axes[3].scatter(label_dataframe['Size'], label_dataframe['Normalized entropy'], 
                              alpha=0.6, c=label_dataframe['Unique_Clusters'], 
                              cmap='plasma', edgecolors='black', linewidth=0.5, s=30)
    axes[3].set_xlabel('Label Size (number of samples)')
    axes[3].set_ylabel('Normalized entropy')
    axes[3].set_title('Normalized entropy vs Label Size')
    axes[3].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[3])
    cbar.set_label('Number of Unique Clusters for Label')
    
    plt.tight_layout()
    return fig


def report_community(cluster, cluster_stats, cluster_df, label_representatives, target_lbl):
    
    known_df = cluster_df[cluster_df[target_lbl] != UNKNOWN_LABEL]
    unknown_count = len(cluster_df) - len(known_df)
    
    label_counts = known_df[target_lbl].value_counts()
    n_labels = len(label_counts)
    
    # Handle edge case: cluster with only unknowns
    if n_labels == 0:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.text(0.5, 0.5, f"Cluster {cluster}\n{unknown_count} unknown characters\nNo known labels", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    # Determine grid layout
    max_cols = 5
    n_cols = min(n_labels, max_cols)
    n_rows = (n_labels + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = np.array(axes).flatten() if n_labels > 1 else [axes]
    
    # Build title with unknown count if present
    title_suffix = f" | Unknown: {unknown_count}" if unknown_count > 0 else ""
    purity_str = f"{cluster_stats['Purity']:.2f}" if not np.isnan(cluster_stats['Purity']) else "N/A"
    entropy_str = f"{cluster_stats['Entropy']:.2f}" if not np.isnan(cluster_stats['Entropy']) else "N/A"
    
    fig.suptitle(
        f"Cluster {cluster} | Size: {cluster_stats['Size']}{title_suffix} | "
        f"Purity: {purity_str} | Entropy: {entropy_str}",
        fontsize=28, fontweight='bold'
    )
    
    # Compute percentage based on known characters only
    known_size = len(known_df)
    
    for idx, label in enumerate(label_counts.index):
        representative_idx = label_representatives.get(label)
        
        if representative_idx is not None and representative_idx in cluster_df.index:
            representative_svg = cluster_df.loc[representative_idx, 'svg'].render(scale=2)
            axes[idx].imshow(representative_svg)
        else:
            axes[idx].text(0.5, 0.5, "No repr.", ha='center', va='center', transform=axes[idx].transAxes)
        
        label_count = label_counts[label]
        label_pct = (label_count / known_size) * 100  # Percentage of known labels
        
        axes[idx].axis('off')
        axes[idx].set_title(f"{label}\n{label_count} ({label_pct:.1f}%)", fontsize=28)
    
    # Hide unused subplots
    for idx in range(n_labels, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_nearest_neighbors(
    query_idx, 
    dataframe,
    dissimilarities,
    graph,
    n_to_show=23,
    figsize_scale=1.0
):
    """
    Plot query patch and its nearest neighbors with color coding based on
    community membership and edge connectivity.
    
    Color scheme:
    - Black: Same community AND edge exists
    - Magenta: Same community BUT no edge
    - Orange: Different community BUT edge exists
    - Red: Different community AND no edge
    
    Parameters
    ----------
    query_idx : int
        Index of the query patch
    dataframe : pd.DataFrame
        DataFrame with 'svg' and 'membership' columns
    dissimilarities : torch.Tensor
        Distance/dissimilarity matrix
    graph : networkx.Graph
        Graph containing edges between nodes
    n_to_show : int, default=23
        Number of nearest neighbors to display
    figsize_scale : float, default=1.0
        Scale factor for figure size
    
    Returns
    -------
    fig : matplotlib.Figure
        The generated figure
    """
    
    n_to_show = min(n_to_show, len(dataframe) - 1)
    
    query_community = dataframe['membership'].iloc[query_idx]
    query_neighbors = set(graph.neighbors(query_idx)) if graph.has_node(query_idx) else set()
    
    distances = dissimilarities[query_idx].cpu().numpy()
    distances[query_idx] = np.inf
    sorted_indices = np.argsort(distances)[:n_to_show]
    
    n_total = n_to_show + 1
    n_cols = min(8, n_total)
    n_rows = int(np.ceil(n_total / n_cols))
    
    fig_width = min(20, n_cols * 2.2) * figsize_scale
    fig_height = n_rows * 2.2 * figsize_scale
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axs = np.atleast_2d(axs).ravel()
    
    # Plot query patch
    ax = axs[0]
    ax.imshow(dataframe['svg'].iloc[query_idx].render(), cmap='gray_r')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#2E86AB')
        spine.set_linewidth(4)
        spine.set_visible(True)
    
    # Plot nearest neighbors
    for j, neighbor_idx in enumerate(sorted_indices):
        ax = axs[j + 1]
        ax.imshow(dataframe['svg'].iloc[neighbor_idx].render(), cmap='gray_r')
        
        same_community = dataframe['membership'].iloc[neighbor_idx] == query_community
        has_edge = neighbor_idx in query_neighbors
        
        if same_community and has_edge:
            color = '#000000'
        elif same_community and not has_edge:
            color = '#FF00FF'
        elif not same_community and has_edge:
            color = '#FF8C00'
        else:
            color = '#DC143C'
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
            spine.set_visible(True)
    
    # Hide unused subplots
    for j in range(n_total, len(axs)):
        axs[j].axis('off')
        axs[j].set_visible(False)
    
    # Legend
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#000000', linewidth=2.5, label='Same comm + Edge'),
        Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#FF00FF', linewidth=2.5, label='Same comm, No edge'),
        Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#FF8C00', linewidth=2.5, label='Diff comm + Edge'),
        Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#DC143C', linewidth=2.5, label='Diff comm, No edge'),
    ]
    
    plt.suptitle(
        f'Query {query_idx} (Community {query_community})',
        fontsize=12, fontweight='bold', y=0.99
    )
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
               bbox_to_anchor=(0.5, -0.01), fontsize=9, frameon=True, framealpha=0.95)
    
    plt.tight_layout()
    
    return fig
