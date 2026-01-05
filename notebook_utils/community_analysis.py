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
        Index of the query patch to analyze. If None, a random index is chosen.
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
    dict
        Dictionary containing analysis results:
        - 'query_idx': The query index used
        - 'community_idx': Index of the query's community
        - 'community_size': Size of the community
        - 'n_matches': Number of accepted matches
        - 'same_comm_matches': Matches within same community
        - 'diff_comm_matches': Matches in different communities
        - 'pca': PCA object (if computed)
        - 'explained_variance': Cumulative variance explained (if PCA computed)
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
    
    # Get community info
    community_indices = list(communities[query_community_idx])
    community_size = len(community_indices)
    
    print(f"\n{'='*60}")
    print(f"Community {query_community_idx} Analysis")
    print(f"{'='*60}")
    print(f"Community size: {community_size} patches")
    
    # PCA Analysis
    if show_pca:
        _plot_pca_analysis(
            i, community_indices, community_size, query_community_idx,
            dataset, patches_df, torch_to_pil, min_patches_for_pca,
            figsize_scale, results
        )
    
    # Histogram
    if show_histogram:
        _plot_histogram(
            nlfa, i, query_community_idx, nlfa_threshold, eps,
            query_matches, figsize_scale
        )
    
    # Scatter plot
    if show_scatter:
        _plot_scatter(
            sorted_distances, accepted_mask, same_community_mask,
            nlfa_threshold, i, query_community_idx, eps,
            query_matches, n_to_show, figsize_scale
        )
    
    # Patch visualization
    if show_patches:
        _plot_patches(
            patches_df, i, query_community_idx, sorted_indices,
            sorted_distances, accepted_mask, same_community_mask,
            nlfa_threshold, eps, query_matches, n_to_show, figsize_scale
        )
    
    # Print summary
    _print_summary(
        i, query_community_idx, communities, query_matches,
        same_community_mask, accepted_mask, n_to_show
    )
    
    return results


def _plot_pca_analysis(
    i, community_indices, community_size, query_community_idx,
    dataset, patches_df, torch_to_pil, min_patches_for_pca,
    figsize_scale, results
):
    """Plot PCA analysis of community patches."""
    
    if community_size >= min_patches_for_pca:
        community_patches = [dataset[idx].unsqueeze(0) for idx in community_indices]
        community_patches_flat = torch.stack([
            patch.flatten() for patch in community_patches
        ]).cpu().numpy()
        
        # Compute mean patch
        mean_patch_flat = community_patches_flat.mean(axis=0)
        mean_patch = mean_patch_flat.reshape(community_patches[0].shape)
        
        max_components = min(community_size - 1, community_patches_flat.shape[1])
        
        if community_size >= 2 and max_components >= 1:
            # Standardize and apply PCA
            scaler = StandardScaler()
            community_patches_scaled = scaler.fit_transform(community_patches_flat)
            
            n_components = min(10, max_components)
            pca = PCA(n_components=n_components)
            community_pca = pca.fit_transform(community_patches_scaled)
            
            results['pca'] = pca
            results['explained_variance'] = np.cumsum(pca.explained_variance_ratio_ * 100)
            
            # Reconstruct principal component patches
            n_pc_to_show = min(5, n_components)
            pc_patches = []
            for pc_idx in range(n_pc_to_show):
                pc = pca.components_[pc_idx]
                pc_patch = pc.reshape(community_patches[0].shape)
                pc_patches.append(pc_patch)
            
            print(f"PCA computed with {n_components} components")
            
            # Visualization 1: Mean and PC patches
            _plot_mean_and_pcs(
                mean_patch, pc_patches, pca, query_community_idx,
                community_size, n_pc_to_show, torch_to_pil, figsize_scale
            )
            
            # Visualization 2: Variance explained
            _plot_variance_explained(
                pca, n_components, query_community_idx, community_size, figsize_scale
            )
            
            # Visualization 3: Violin plots
            _plot_violin(
                community_pca, pca, i, community_indices, query_community_idx,
                community_size, n_components, figsize_scale
            )
            
            cumsum_var = results['explained_variance']
            print(f"PCA Analysis:")
            print(f"  - Total variance explained by {n_components} components: {cumsum_var[-1]:.1f}%")
            if cumsum_var[-1] >= 95:
                print(f"  - Components needed for 95% variance: {np.argmax(cumsum_var >= 95) + 1}")
            else:
                print(f"  - Note: Only {cumsum_var[-1]:.1f}% variance achievable with {n_components} components")
        
        else:
            print(f"⚠️  Community too small for PCA (size={community_size})")
            _plot_mean_only(mean_patch, query_community_idx, community_size, torch_to_pil, figsize_scale)
    
    else:
        print(f"⚠️  Community too small for analysis (size={community_size}, min={min_patches_for_pca})")
        _plot_all_community_patches(
            community_indices, i, query_community_idx, community_size,
            patches_df, torch_to_pil, figsize_scale
        )


def _plot_mean_and_pcs(
    mean_patch, pc_patches, pca, query_community_idx,
    community_size, n_pc_to_show, torch_to_pil, figsize_scale
):
    """Plot mean patch and principal components."""
    fig, axs = plt.subplots(
        1, n_pc_to_show + 1,
        figsize=(3 * (n_pc_to_show + 1) * figsize_scale, 3 * figsize_scale)
    )
    if n_pc_to_show == 0:
        axs = [axs]
    
    # Mean patch
    axs[0].imshow(
        torch_to_pil(torch.from_numpy(mean_patch)).resize((256, 256)),
        cmap="gray"
    )
    axs[0].set_title(
        f'Mean Patch\n(Community {query_community_idx}, n={community_size})',
        fontweight='bold', fontsize=11
    )
    axs[0].axis('off')
    for spine in axs[0].spines.values():
        spine.set_edgecolor('blue')
        spine.set_linewidth(3)
        spine.set_visible(True)
    
    # PC patches
    for pc_idx in range(n_pc_to_show):
        ax = axs[pc_idx + 1]
        pc_patch_normalized = (pc_patches[pc_idx] - pc_patches[pc_idx].min()) / \
                             (pc_patches[pc_idx].max() - pc_patches[pc_idx].min() + 1e-8)
        ax.imshow(
            torch_to_pil(torch.from_numpy(pc_patch_normalized)).resize((256, 256)),
            cmap="seismic"
        )
        variance_explained = pca.explained_variance_ratio_[pc_idx] * 100
        ax.set_title(f'PC{pc_idx + 1}\n({variance_explained:.1f}% var)', fontweight='bold', fontsize=11)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('orange')
            spine.set_linewidth(2)
            spine.set_visible(True)
    
    plt.suptitle(
        f'Community {query_community_idx}: Mean and Principal Components ({community_size} patches)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()


def _plot_variance_explained(pca, n_components, query_community_idx, community_size, figsize_scale):
    """Plot variance explained bar chart and cumulative plot."""
    fig = plt.figure(figsize=(10 * figsize_scale, 4 * figsize_scale))
    
    plt.subplot(1, 2, 1)
    plt.bar(
        range(1, n_components + 1),
        pca.explained_variance_ratio_ * 100,
        color='steelblue', edgecolor='black'
    )
    plt.xlabel('Principal Component', fontsize=11)
    plt.ylabel('Variance Explained (%)', fontsize=11)
    plt.title(
        f'Variance Explained by Each PC\n(Community {query_community_idx}, n={community_size})',
        fontweight='bold'
    )
    plt.xticks(range(1, n_components + 1))
    plt.grid(alpha=0.3, axis='y')
    
    plt.subplot(1, 2, 2)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_ * 100)
    plt.plot(
        range(1, n_components + 1), cumsum_var,
        marker='o', linewidth=2, markersize=8, color='steelblue'
    )
    plt.xlabel('Number of Components', fontsize=11)
    plt.ylabel('Cumulative Variance Explained (%)', fontsize=11)
    plt.title(
        f'Cumulative Variance Explained\n(Community {query_community_idx}, n={community_size})',
        fontweight='bold'
    )
    plt.xticks(range(1, n_components + 1))
    plt.grid(alpha=0.3)
    
    if cumsum_var[-1] < 95:
        plt.axhline(y=cumsum_var[-1], color='orange', linestyle='--', linewidth=1,
                   label=f'Max: {cumsum_var[-1]:.1f}%')
    else:
        plt.axhline(y=95, color='red', linestyle='--', linewidth=1, label='95% threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def _plot_violin(
    community_pca, pca, i, community_indices, query_community_idx,
    community_size, n_components, figsize_scale
):
    """Plot violin/box plots of PCA component distributions."""
    n_pc_violin = min(6, n_components)
    n_cols = 3
    n_rows = int(np.ceil(n_pc_violin / n_cols))
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15 * figsize_scale, 5 * n_rows * figsize_scale))
    axs = np.atleast_1d(axs).ravel()
    
    query_idx_in_community = community_indices.index(i)
    
    for pc_idx in range(n_pc_violin):
        ax = axs[pc_idx]
        component_values = community_pca[:, pc_idx]
        
        if community_size <= 5:
            bp = ax.boxplot([component_values], positions=[0], widths=0.4,
                           patch_artist=True, showmeans=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
        else:
            parts = ax.violinplot([component_values], positions=[0], widths=0.7,
                                 showmeans=True, showextrema=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor('lightblue')
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
        
        jitter_amount = 0.04 if community_size > 10 else 0.02
        jitter = np.random.normal(0, jitter_amount, size=len(component_values))
        ax.scatter(jitter, component_values, alpha=0.5, s=30,
                  color='steelblue', edgecolors='black', linewidth=0.5)
        
        query_value = component_values[query_idx_in_community]
        ax.scatter(0, query_value, s=200, color='red', marker='*',
                  edgecolors='black', linewidth=2, label='Query patch', zorder=10)
        
        variance_explained = pca.explained_variance_ratio_[pc_idx] * 100
        ax.set_title(f'PC{pc_idx + 1} ({variance_explained:.1f}% var)',
                    fontweight='bold', fontsize=12)
        ax.set_ylabel('Component Value', fontsize=10)
        ax.set_xticks([])
        ax.set_xlim([-0.3, 0.3])
        ax.grid(alpha=0.3, axis='y')
        ax.legend(fontsize=9)
        
        mean_val = component_values.mean()
        std_val = component_values.std()
        ax.text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    for pc_idx in range(n_pc_violin, len(axs)):
        axs[pc_idx].axis('off')
    
    plt.suptitle(
        f'PCA Component Distributions: Community {query_community_idx} ({community_size} patches)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()


def _plot_mean_only(mean_patch, query_community_idx, community_size, torch_to_pil, figsize_scale):
    """Plot only the mean patch when PCA isn't possible."""
    print("   Showing mean patch only...")
    
    fig, ax = plt.subplots(1, 1, figsize=(4 * figsize_scale, 4 * figsize_scale))
    ax.imshow(torch_to_pil(torch.from_numpy(mean_patch)).resize((256, 256)), cmap="gray")
    ax.set_title(
        f'Mean Patch\n(Community {query_community_idx}, n={community_size})',
        fontweight='bold', fontsize=12
    )
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor('blue')
        spine.set_linewidth(3)
        spine.set_visible(True)
    plt.tight_layout()
    plt.show()


def _plot_all_community_patches(
    community_indices, i, query_community_idx, community_size,
    patches_df, torch_to_pil, figsize_scale
):
    """Plot all patches in a small community."""
    print("   Showing all patches in community...")
    
    fig, axs = plt.subplots(
        1, community_size,
        figsize=(3 * community_size * figsize_scale, 3 * figsize_scale)
    )
    if community_size == 1:
        axs = [axs]
    
    for idx, patch_idx in enumerate(community_indices):
        axs[idx].imshow(
            torch_to_pil(patches_df.iloc[patch_idx]['bin_patch'][None, ...]).resize((256, 256)),
            cmap="gray"
        )
        is_query = (patch_idx == i)
        title = f'{"Query" if is_query else f"Patch {idx+1}"}\n(idx={patch_idx})'
        color = 'red' if is_query else 'blue'
        axs[idx].set_title(title, fontweight='bold', fontsize=10, color=color)
        axs[idx].axis('off')
        for spine in axs[idx].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3 if is_query else 2)
            spine.set_visible(True)
    
    plt.suptitle(
        f'All Patches in Community {query_community_idx} ({community_size} patches)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()


def _plot_histogram(nlfa, i, query_community_idx, nlfa_threshold, eps, query_matches, figsize_scale):
    """Plot histogram of distance distribution."""
    plt.figure(figsize=(12 * figsize_scale, 4 * figsize_scale))
    hist_values, bin_edges, patches_hist = plt.hist(
        nlfa[i], bins=50, color='skyblue', edgecolor='black', alpha=0.7
    )
    
    plt.vlines(nlfa_threshold, 0, hist_values.max(), color='orange', linestyle='--',
               linewidth=3, label=f'Threshold nlfa = {nlfa_threshold:.3f}')
    
    plt.axvspan(nlfa[i].max(), nlfa_threshold, alpha=0.2, color='green',
                label=f'Accepted region ({len(query_matches)} matches)')
    
    plt.xlabel('NLFA', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(
        f'Distance Distribution: Query {i} (Community {query_community_idx}, ε={eps})',
        fontsize=13, fontweight='bold'
    )
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def _plot_scatter(
    sorted_distances, accepted_mask, same_community_mask,
    nlfa_threshold, i, query_community_idx, eps,
    query_matches, n_to_show, figsize_scale
):
    """Plot scatter plot of nearest neighbors."""
    plt.figure(figsize=(12 * figsize_scale, 4 * figsize_scale))
    
    colors = []
    markers = []
    for accepted, same_comm in zip(accepted_mask, same_community_mask):
        if same_comm:
            colors.append('green' if accepted else 'orange')
            markers.append('o')
        else:
            colors.append('green' if accepted else 'red')
            markers.append('s')
    
    for rank, (dist, color, marker) in enumerate(zip(sorted_distances[:n_to_show], colors, markers), 1):
        plt.scatter(rank, dist, c=color, s=100, marker=marker, zorder=3,
                   edgecolors='black', linewidth=0.5)
    
    plt.axhline(y=nlfa_threshold, color='orange', linestyle='--', linewidth=2,
                label=f'Threshold NLFA = {nlfa_threshold:.3f}')
    
    plt.axhspan(nlfa_threshold, 0, alpha=0.2, color='red', label='Rejected region')
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10,
               label='Accepted + Same Community', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10,
               label='Accepted + Different Community', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10,
               label='Rejected + Same Community', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10,
               label='Rejected + Different Community', markeredgecolor='black'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=2,
               label=f'Threshold δ_i(ε) = {nlfa_threshold:.3f}')
    ]
    
    plt.xlabel('Nearest Neighbor Rank', fontsize=12)
    plt.ylabel('Distance D(a^i, b^j)', fontsize=12)
    plt.title(
        f'Statistical Test: Query {i} (Community {query_community_idx}, ε={eps}, {len(query_matches)} matches)',
        fontsize=13, fontweight='bold'
    )
    plt.xticks(range(1, n_to_show + 1))
    plt.grid(alpha=0.3)
    plt.legend(handles=legend_elements, fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.show()


def _plot_patches(
    patches_df, i, query_community_idx, sorted_indices,
    sorted_distances, accepted_mask, same_community_mask,
    nlfa_threshold, eps, query_matches, n_to_show, figsize_scale
):
    """Plot query patch and nearest neighbors."""
    n_total = n_to_show + 1
    n_cols = min(8, n_total)
    n_rows = int(np.ceil(n_total / n_cols))
    
    fig_width = min(23, n_cols * 2.5) * figsize_scale
    fig_height = n_rows * 2.5 * figsize_scale
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axs = np.atleast_2d(axs).ravel()
    
    ax = axs[0]
    ax.imshow(patches_df['svg'][i].render(), cmap="gray")
    ax.set_title(f'Query patch\n(Community {query_community_idx})', fontweight='bold', fontsize=10, pad=8)
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor('blue')
        spine.set_linewidth(4)
        spine.set_visible(True)
    
    for j in range(n_to_show):
        candidate_idx = sorted_indices[j]
        distance = sorted_distances[j]
        
        ax = axs[j + 1]
        ax.imshow(patches_df['svg'][candidate_idx].render(), cmap="gray")
        
        is_accepted = accepted_mask[j]
        is_same_community = same_community_mask[j]
        
        if is_same_community:
            status = '✓●' if is_accepted else '❌●'
            color = 'green' if is_accepted else 'orange'
            border_style = '-'
        else:
            status = '✓' if is_accepted else '❌'
            color = 'green' if is_accepted else 'red'
            border_style = '--'
        
        ax.set_title(f'{status} NN{j+1}\nd={distance*100:.3f}',
                    fontweight='bold', color=color, fontsize=9, pad=8)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_linestyle(border_style)
            spine.set_visible(True)
    
    for j in range(n_total, len(axs)):
        axs[j].axis('off')
        axs[j].set_visible(False)
    
    same_comm_accepted = sum(accepted_mask & same_community_mask)
    diff_comm_accepted = sum(accepted_mask & ~same_community_mask)
    
    plt.suptitle(
        f'ε-Meaningful Matches: Query {i} (Community {query_community_idx}) | ε={eps} | δ={nlfa_threshold:.3f}\n'
        f'{len(query_matches)} accepted ({same_comm_accepted} same comm, {diff_comm_accepted} diff comm) | '
        f'● = same community',
        fontsize=13, fontweight='bold', y=0.99
    )
    plt.tight_layout()
    plt.show()


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