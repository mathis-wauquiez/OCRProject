import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def analyze_community_patch(
    query_idx,
    patches_df,
    dataset,
    communities,
    match_indices,
    nlfa,
    nlfa_threshold,
    eps,
    torch_to_pil,
    n_to_show=23,
    min_patches_for_pca=3,
    show_pca=True,
    show_histogram=True,
    show_scatter=True,
    show_patches=True,
    figsize_scale=1.0,
):
    """
    Analyze a query patch's community membership, PCA structure, and nearest neighbors.
    
    Parameters
    ----------
    query_idx : int
        Index of the query patch to analyze.
    patches_df : pd.DataFrame
        DataFrame containing patch data with 'svg' and 'bin_patch' columns.
    dataset : torch.Tensor or indexable
        The dataset of patches (used for PCA).
    communities : list of lists/sets
        List of communities, where each community is a collection of patch indices.
    match_indices : torch.Tensor
        Tensor of shape (N, 2) containing matched pairs from statistical test.
    nlfa : torch.Tensor
        NLFA (Negative Log False Alarm) matrix of shape (N, N) or (N,) for distances.
    nlfa_threshold : float
        Threshold value for accepting matches.
    eps : float
        Epsilon value used in the statistical test.
    torch_to_pil : callable
        Function to convert torch tensor to PIL image.
    n_to_show : int, default=23
        Number of nearest neighbors to display.
    min_patches_for_pca : int, default=3
        Minimum number of patches required to perform PCA.
    show_pca : bool, default=True
        Whether to show PCA visualizations.
    show_histogram : bool, default=True
        Whether to show distance distribution histogram.
    show_scatter : bool, default=True
        Whether to show nearest neighbor scatter plot.
    show_patches : bool, default=True
        Whether to show patch visualizations.
    figsize_scale : float, default=1.0
        Scale factor for figure sizes.
    
    Returns
    -------
    results : dict
        Dictionary containing analysis results
    figs : dict
        Dictionary of matplotlib figures (None if not generated)
    """
    i = query_idx
    n_to_show = min(n_to_show, len(patches_df['svg']))
    
    # Find which community the query belongs to
    query_community_idx = None
    for comm_idx, community in enumerate(communities):
        if i in community:
            query_community_idx = comm_idx
            break
    
    if query_community_idx is None:
        raise ValueError(f"Query index {i} not found in any community")
    
    # Get all matches for query i from the statistical test
    query_matches = match_indices[match_indices[:, 0] == i][:, 1]
    
    # Compute distances from query i to all candidates
    D_i = nlfa[i].detach().cpu().numpy()
    
    # Get sorted candidate indices by distance (nearest first)
    sorted_indices = np.argsort(-D_i)
    sorted_distances = D_i[sorted_indices]
    
    # Determine which are accepted (ε-meaningful matches)
    accepted_mask = np.isin(sorted_indices[:n_to_show], query_matches.cpu().numpy())
    
    # Determine which are in the same community
    same_community_mask = np.array([
        idx in communities[query_community_idx] 
        for idx in sorted_indices[:n_to_show]
    ])
    
    # Initialize results
    results = {
        'query_idx': i,
        'community_idx': query_community_idx,
        'community_size': len(communities[query_community_idx]),
        'n_matches': len(query_matches),
        'same_comm_matches': int(sum(accepted_mask & same_community_mask)),
        'diff_comm_matches': int(sum(accepted_mask & ~same_community_mask)),
        'pca': None,
        'explained_variance': None,
    }

    figs = {}
    
    # Get community info
    community_indices = list(communities[query_community_idx])
    community_size = len(community_indices)
    
    print(f"\n{'='*60}")
    print(f"Community {query_community_idx} Analysis")
    print(f"{'='*60}")
    print(f"Community size: {community_size} patches")
    
    # PCA Analysis
    if show_pca:
        pca_figs = _plot_pca_analysis(
            i, community_indices, community_size, query_community_idx,
            dataset, patches_df, torch_to_pil, min_patches_for_pca,
            figsize_scale, results
        )
        if pca_figs is not None:
            figs['pca_components'] = pca_figs[0]
            figs['pca_variance'] = pca_figs[1]
            figs['pca_distributions'] = pca_figs[2]
        else:
            figs['pca_components'] = None
            figs['pca_variance'] = None
            figs['pca_distributions'] = None
    
    # Histogram
    if show_histogram:
        figs['histogram'] = _plot_histogram(
            nlfa, i, query_community_idx, nlfa_threshold, eps,
            query_matches, figsize_scale
        )
    
    # Scatter plot
    if show_scatter:
        figs['scatter'] = _plot_scatter(
            sorted_distances, accepted_mask, same_community_mask,
            nlfa_threshold, i, query_community_idx, eps,
            query_matches, n_to_show, figsize_scale
        )
    
    # Patch visualization
    if show_patches:
        figs['patches'] = _plot_patches(
            patches_df, i, query_community_idx, sorted_indices,
            sorted_distances, accepted_mask, same_community_mask,
            nlfa_threshold, eps, query_matches, n_to_show, figsize_scale
        )
    
    # Print summary
    _print_summary(
        i, query_community_idx, communities, query_matches,
        same_community_mask, accepted_mask, n_to_show
    )
    
    return results, figs


def _plot_pca_analysis(
    i, community_indices, community_size, query_community_idx,
    dataset, patches_df, torch_to_pil, min_patches_for_pca,
    figsize_scale, results
):
    """Plot PCA analysis of community patches. Returns None if PCA can't be performed."""
    
    if community_size < min_patches_for_pca:
        print(f"⚠️  Community too small for PCA analysis (size={community_size}, min={min_patches_for_pca})")
        return None
    
    community_patches = [dataset[idx].unsqueeze(0) for idx in community_indices]
    community_patches_flat = torch.stack([
        patch.flatten() for patch in community_patches
    ]).cpu().numpy()
    
    max_components = min(community_size - 1, community_patches_flat.shape[1])
    
    if community_size < 2 or max_components < 1:
        print(f"⚠️  Community too small for PCA (size={community_size})")
        return None
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    community_patches_scaled = scaler.fit_transform(community_patches_flat)
    
    n_components = min(10, max_components)
    pca = PCA(n_components=n_components)
    community_pca = pca.fit_transform(community_patches_scaled)
    
    results['pca'] = pca
    results['explained_variance'] = np.cumsum(pca.explained_variance_ratio_ * 100)
    
    print(f"PCA computed with {n_components} components")
    cumsum_var = results['explained_variance']
    print(f"  - Total variance explained: {cumsum_var[-1]:.1f}%")
    if cumsum_var[-1] >= 95:
        print(f"  - Components for 95% variance: {np.argmax(cumsum_var >= 95) + 1}")
    
    # Compute mean patch
    mean_patch_flat = community_patches_flat.mean(axis=0)
    mean_patch = mean_patch_flat.reshape(community_patches[0].shape)
    
    # Reconstruct principal component patches
    n_pc_to_show = min(5, n_components)
    pc_patches = []
    for pc_idx in range(n_pc_to_show):
        pc = pca.components_[pc_idx]
        pc_patch = pc.reshape(community_patches[0].shape)
        pc_patches.append(pc_patch)
    
    # Create visualizations
    fig1 = _plot_mean_and_pcs(
        mean_patch, pc_patches, pca, query_community_idx,
        community_size, n_pc_to_show, torch_to_pil, figsize_scale
    )
    
    fig2 = _plot_variance_explained(
        pca, n_components, query_community_idx, community_size, figsize_scale
    )
    
    fig3 = _plot_violin(
        community_pca, pca, i, community_indices, query_community_idx,
        community_size, n_components, figsize_scale
    )
    
    return fig1, fig2, fig3


def _plot_mean_and_pcs(
    mean_patch, pc_patches, pca, query_community_idx,
    community_size, n_pc_to_show, torch_to_pil, figsize_scale
):
    """Plot mean patch and principal components."""
    fig, axs = plt.subplots(
        1, n_pc_to_show + 1,
        figsize=(2.5 * (n_pc_to_show + 1) * figsize_scale, 2.8 * figsize_scale)
    )
    if n_pc_to_show == 0:
        axs = [axs]
    
    # Mean patch
    axs[0].imshow(
        torch_to_pil(torch.from_numpy(mean_patch)).resize((256, 256)),
        cmap="gray"
    )
    axs[0].set_title(
        f'Mean Patch\n(n={community_size})',
        fontsize=10, pad=8
    )
    axs[0].axis('off')
    
    # PC patches
    for pc_idx in range(n_pc_to_show):
        ax = axs[pc_idx + 1]
        pc_patch_normalized = (pc_patches[pc_idx] - pc_patches[pc_idx].min()) / \
                             (pc_patches[pc_idx].max() - pc_patches[pc_idx].min() + 1e-8)
        ax.imshow(
            torch_to_pil(torch.from_numpy(pc_patch_normalized)).resize((256, 256)),
            cmap="RdBu_r"
        )
        variance_explained = pca.explained_variance_ratio_[pc_idx] * 100
        ax.set_title(f'PC{pc_idx + 1}\n({variance_explained:.1f}%)', fontsize=10, pad=8)
        ax.axis('off')
    
    plt.suptitle(
        f'Community {query_community_idx}: Mean and Principal Components',
        fontsize=12, y=0.98
    )
    plt.tight_layout()
    
    return fig


def _plot_variance_explained(pca, n_components, query_community_idx, community_size, figsize_scale):
    """Plot variance explained bar chart and cumulative plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * figsize_scale, 3.5 * figsize_scale))
    
    # Bar plot
    ax1.bar(
        range(1, n_components + 1),
        pca.explained_variance_ratio_ * 100,
        color='#4682b4', edgecolor='black', linewidth=0.5
    )
    ax1.set_xlabel('Principal Component', fontsize=10)
    ax1.set_ylabel('Variance Explained (%)', fontsize=10)
    ax1.set_title(f'Variance per Component', fontsize=11)
    ax1.set_xticks(range(1, n_components + 1))
    ax1.grid(alpha=0.2, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Cumulative plot
    cumsum_var = np.cumsum(pca.explained_variance_ratio_ * 100)
    ax2.plot(
        range(1, n_components + 1), cumsum_var,
        marker='o', linewidth=2, markersize=6, color='#4682b4'
    )
    ax2.set_xlabel('Number of Components', fontsize=10)
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=10)
    ax2.set_title(f'Cumulative Variance', fontsize=11)
    ax2.set_xticks(range(1, n_components + 1))
    ax2.grid(alpha=0.2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add reference line
    if cumsum_var[-1] >= 95:
        ax2.axhline(y=95, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.text(1, 95, ' 95%', fontsize=8, va='bottom', color='gray')
    
    fig.suptitle(f'Community {query_community_idx} - PCA Variance Analysis', fontsize=12, y=0.98)
    plt.tight_layout()
    
    return fig


def _plot_violin(
    community_pca, pca, i, community_indices, query_community_idx,
    community_size, n_components, figsize_scale
):
    """Plot violin/box plots of PCA component distributions."""
    n_pc_violin = min(6, n_components)
    n_cols = 3
    n_rows = int(np.ceil(n_pc_violin / n_cols))
    
    fig, axs = plt.subplots(
        n_rows, n_cols, 
        figsize=(12 * figsize_scale, 3.5 * n_rows * figsize_scale)
    )
    axs = np.atleast_1d(axs).ravel()
    
    query_idx_in_community = community_indices.index(i)
    
    for pc_idx in range(n_pc_violin):
        ax = axs[pc_idx]
        component_values = community_pca[:, pc_idx]
        
        # Choose visualization based on community size
        if community_size <= 5:
            bp = ax.boxplot(
                [component_values], positions=[0], widths=0.4,
                patch_artist=True, showmeans=True
            )
            bp['boxes'][0].set_facecolor('#a8c5e0')
            bp['boxes'][0].set_alpha(0.6)
        else:
            parts = ax.violinplot(
                [component_values], positions=[0], widths=0.6,
                showmeans=False, showextrema=True, showmedians=True
            )
            for pc in parts['bodies']:
                pc.set_facecolor('#a8c5e0')
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
        
        # Scatter individual points
        jitter_amount = 0.03 if community_size > 10 else 0.02
        jitter = np.random.normal(0, jitter_amount, size=len(component_values))
        ax.scatter(
            jitter, component_values, alpha=0.4, s=25,
            color='#4682b4', edgecolors='none'
        )
        
        # Highlight query patch
        query_value = component_values[query_idx_in_community]
        ax.scatter(
            0, query_value, s=150, color='#d62728', marker='*',
            edgecolors='black', linewidth=1, zorder=10
        )
        
        variance_explained = pca.explained_variance_ratio_[pc_idx] * 100
        ax.set_title(f'PC{pc_idx + 1} ({variance_explained:.1f}%)', fontsize=10)
        ax.set_ylabel('Value', fontsize=9)
        ax.set_xticks([])
        ax.set_xlim([-0.25, 0.25])
        ax.grid(alpha=0.2, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide unused subplots
    for pc_idx in range(n_pc_violin, len(axs)):
        axs[pc_idx].axis('off')
    
    plt.suptitle(
        f'Community {query_community_idx} - PCA Component Distributions (★ = query)',
        fontsize=12, y=0.98
    )
    plt.tight_layout()
    
    return fig


def _plot_histogram(nlfa, i, query_community_idx, nlfa_threshold, eps, query_matches, figsize_scale):
    """Plot histogram of distance distribution."""
    fig, ax = plt.subplots(figsize=(10 * figsize_scale, 3.5 * figsize_scale))
    
    hist_values, bin_edges, patches_hist = ax.hist(
        nlfa[i], bins=50, color='#a8c5e0', edgecolor='black', 
        alpha=0.7, linewidth=0.5
    )
    
    # Threshold line
    ax.axvline(
        nlfa_threshold, color='#d62728', linestyle='--',
        linewidth=2, label=f'Threshold = {nlfa_threshold:.3f}'
    )
    
    # Accepted region
    ax.axvspan(
        nlfa[i].max(), nlfa_threshold, alpha=0.15, color='#2ca02c',
        label=f'Accepted ({len(query_matches)} matches)'
    )
    
    ax.set_xlabel('NLFA', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(
        f'Query {i} (Community {query_community_idx}) - Distance Distribution (ε={eps})',
        fontsize=11, pad=10
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    return fig


def _plot_scatter(
    sorted_distances, accepted_mask, same_community_mask,
    nlfa_threshold, i, query_community_idx, eps,
    query_matches, n_to_show, figsize_scale
):
    """Plot scatter plot of nearest neighbors."""
    fig, ax = plt.subplots(figsize=(10 * figsize_scale, 3.5 * figsize_scale))
    
    # Define colors and markers
    colors = []
    markers = []
    sizes = []
    for accepted, same_comm in zip(accepted_mask, same_community_mask):
        if accepted:
            colors.append('#2ca02c')  # Green for accepted
            sizes.append(80)
        else:
            colors.append('#d62728' if not same_comm else '#ff7f0e')  # Red/orange for rejected
            sizes.append(60)
        markers.append('o' if same_comm else 's')
    
    # Plot points
    for rank, (dist, color, marker, size) in enumerate(
        zip(sorted_distances[:n_to_show], colors, markers, sizes), 1
    ):
        ax.scatter(
            rank, dist, c=color, s=size, marker=marker, 
            alpha=0.7, edgecolors='black', linewidth=0.5, zorder=3
        )
    
    # Threshold line
    ax.axhline(
        y=nlfa_threshold, color='gray', linestyle='--', linewidth=1.5,
        label=f'Threshold = {nlfa_threshold:.3f}', alpha=0.7
    )
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', 
               markersize=8, label='Accepted (same comm)', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ca02c', 
               markersize=8, label='Accepted (diff comm)', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', 
               markersize=8, label='Rejected (same comm)', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728', 
               markersize=8, label='Rejected (diff comm)', markeredgecolor='black', markeredgewidth=0.5),
    ]
    
    ax.set_xlabel('Nearest Neighbor Rank', fontsize=10)
    ax.set_ylabel('NLFA Distance', fontsize=10)
    ax.set_title(
        f'Query {i} (Community {query_community_idx}) - Nearest Neighbors (ε={eps}, {len(query_matches)} matches)',
        fontsize=11, pad=10
    )
    ax.set_xticks(range(1, min(n_to_show + 1, 25), max(1, n_to_show // 10)))
    ax.grid(alpha=0.2)
    ax.legend(handles=legend_elements, fontsize=8, loc='best', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    return fig


def _plot_patches(
    patches_df, i, query_community_idx, sorted_indices,
    sorted_distances, accepted_mask, same_community_mask,
    nlfa_threshold, eps, query_matches, n_to_show, figsize_scale
):
    """Plot query patch and nearest neighbors."""
    n_total = n_to_show + 1
    n_cols = min(8, n_total)
    n_rows = int(np.ceil(n_total / n_cols))
    
    fig_width = min(20, n_cols * 2.2) * figsize_scale
    fig_height = n_rows * 2.2 * figsize_scale
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axs = np.atleast_2d(axs).ravel()
    
    # Query patch
    ax = axs[0]
    ax.imshow(patches_df['svg'][i].render(), cmap="gray")
    ax.set_title(f'Query\n(Comm {query_community_idx})', fontsize=9, pad=5)
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor('#4682b4')
        spine.set_linewidth(2.5)
        spine.set_visible(True)
    
    # Nearest neighbors
    for j in range(n_to_show):
        candidate_idx = sorted_indices[j]
        distance = sorted_distances[j]
        
        ax = axs[j + 1]
        ax.imshow(patches_df['svg'][candidate_idx].render(), cmap="gray")
        
        is_accepted = accepted_mask[j]
        is_same_community = same_community_mask[j]
        
        # Determine styling
        if is_accepted:
            status = '✓'
            color = '#2ca02c'
        else:
            status = '✗'
            color = '#d62728' if not is_same_community else '#ff7f0e'
        
        comm_marker = '●' if is_same_community else ''
        
        ax.set_title(
            f'{status} {comm_marker} {j+1}\n{distance:.3f}',
            color=color, fontsize=8, pad=5
        )
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2 if is_accepted else 1.5)
            spine.set_visible(True)
    
    # Hide unused subplots
    for j in range(n_total, len(axs)):
        axs[j].axis('off')
        axs[j].set_visible(False)
    
    same_comm_accepted = sum(accepted_mask & same_community_mask)
    diff_comm_accepted = sum(accepted_mask & ~same_community_mask)
    
    plt.suptitle(
        f'Query {i} Matches (ε={eps}) | {len(query_matches)} accepted '
        f'({same_comm_accepted} same, {diff_comm_accepted} diff) | ● = same community',
        fontsize=11, y=0.98
    )
    plt.tight_layout()
    
    return fig


def _print_summary(i, query_community_idx, communities, query_matches, same_community_mask, accepted_mask, n_to_show):
    """Print summary statistics."""
    same_comm_accepted = sum(accepted_mask & same_community_mask)
    diff_comm_accepted = sum(accepted_mask & ~same_community_mask)
    
    print(f"\n{'='*60}")
    print(f"Query Patch {i} - Community {query_community_idx}")
    print(f"{'='*60}")
    print(f"Total community size: {len(communities[query_community_idx])}")
    print(f"Total accepted matches: {len(query_matches)}")
    print(f"  - Same community: {same_comm_accepted}")
    print(f"  - Different community: {diff_comm_accepted}")
    print(f"Among top {n_to_show} nearest neighbors:")
    print(f"  - Same community: {sum(same_community_mask)}")
    print(f"  - Different community: {sum(~same_community_mask)}")