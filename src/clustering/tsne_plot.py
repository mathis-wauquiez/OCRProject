import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from PIL import Image
import colorsys
import warnings
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties


def _find_chinese_font():
    """Find an available font that supports Chinese characters."""
    possible_fonts = [
        'Noto Sans CJK TC', 'Noto Sans CJK SC', 'Noto Sans CJK JP',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'AR PL UMing TW', 'AR PL UKai TW', 'AR PL UMing CN',
        'Microsoft YaHei', 'SimHei', 'SimSun',
        'PingFang SC', 'PingFang TC', 'Heiti TC', 'Heiti SC', 'STHeiti',
        'Arial Unicode MS',
    ]
    
    available_fonts = {f.name: f.fname for f in fm.fontManager.ttflist}
    
    for font_name in possible_fonts:
        if font_name in available_fonts:
            return FontProperties(family=font_name)
    
    # Search for any font with CJK support
    for font_name in available_fonts:
        if any(keyword in font_name.lower() for keyword in ['cjk', 'chinese', 'han', 'hei', 'sung', 'ming', 'kai']):
            return FontProperties(family=font_name)
    
    return FontProperties()


UNKNOWN_LABEL = '▯'  # U+25AF

def plot_community_tsne(
    cluster_id,
    dataframe,
    graph,
    target_lbl='char',
    figsize=(24, 20),
    dpi=100,
    disable_svg=False,
    disable_color=False,
    disable_char=False,
    zoom=None,
    color_by_membership=None
):
    """
    Visualize a cluster/community using t-SNE layout with SVG patches.
    Unknown labels (▯) are displayed in gray and excluded from entropy calculations.
    
    Parameters:
    -----------
    color_by_membership : pd.Series, dict, np.ndarray, or None
        If provided, nodes will be colored by their membership value instead of target_lbl.
        - pd.Series: should have same index as dataframe
        - dict: maps node IDs to membership values
        - np.ndarray: should be aligned with cluster_df (same length and order)
    """
    # Get nodes in this cluster
    cluster_df = dataframe[dataframe['membership'] == cluster_id]
    cluster_nodes = cluster_df.index.tolist()
    cluster_size = len(cluster_nodes)
    
    if cluster_size == 0:
        raise ValueError(f"Cluster {cluster_id} is empty")
    
    # Create subgraph
    G_sub = graph.subgraph(cluster_nodes).copy()
    
    # Determine what to use for coloring
    use_membership_coloring = color_by_membership is not None
    
    if use_membership_coloring:
        # Extract membership values for nodes in this cluster
        if hasattr(color_by_membership, 'loc'):
            # pandas Series
            color_values = [color_by_membership.loc[node] for node in cluster_nodes]
        elif isinstance(color_by_membership, np.ndarray):
            # numpy array - assume it's aligned with cluster_df by position
            if len(color_by_membership) != len(cluster_df):
                raise ValueError(f"color_by_membership array length ({len(color_by_membership)}) "
                               f"doesn't match cluster size ({len(cluster_df)})")
            color_values = color_by_membership.tolist()
        else:
            # dict
            color_values = [color_by_membership[node] for node in cluster_nodes]
        
        unique_values = sorted(set(color_values))
        num_unique = len(unique_values)
        
        # Compute entropy for membership distribution
        value_counts = {}
        for val in color_values:
            value_counts[val] = value_counts.get(val, 0) + 1
        
        value_probs = {val: count / len(color_values) for val, count in value_counts.items()}
        entropy_val = -sum(p * np.log2(p) for p in value_probs.values() if p > 0)
        max_entropy = np.log2(len(value_counts)) if len(value_counts) > 1 else 0.0
        normalized_entropy = entropy_val / max_entropy if max_entropy > 0 else 0.0
        
    else:
        # Original label-based coloring
        labels_in_cluster = cluster_df[target_lbl].fillna(UNKNOWN_LABEL).tolist()
        color_values = labels_in_cluster
        known_mask = [lbl != UNKNOWN_LABEL for lbl in labels_in_cluster]
        known_labels = [lbl for lbl in labels_in_cluster if lbl != UNKNOWN_LABEL]
        unknown_count = labels_in_cluster.count(UNKNOWN_LABEL)
        
        unique_labels = sorted(set(labels_in_cluster))
        unique_known_labels = sorted(set(known_labels))
        num_unique_known = len(unique_known_labels)
        unique_values = unique_known_labels
        num_unique = num_unique_known
        
        # Compute entropy only on known labels
        if len(known_labels) > 0:
            label_counts = {}
            for lbl in known_labels:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            
            total_known = len(known_labels)
            label_probs = {lbl: count / total_known for lbl, count in label_counts.items()}
            entropy_val = -sum(p * np.log2(p) for p in label_probs.values() if p > 0)
            max_entropy = np.log2(len(label_counts)) if len(label_counts) > 1 else 0.0
            normalized_entropy = entropy_val / max_entropy if max_entropy > 0 else 0.0
            value_counts = label_counts
            value_probs = label_probs
        else:
            label_counts = {}
            label_probs = {}
            entropy_val = normalized_entropy = max_entropy = 0.0
            total_known = 0
            value_counts = {}
            value_probs = {}
    
    # Assign colors
    color_map = {}
    for i, val in enumerate(unique_values):
        hue = i / max(num_unique, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.8)
        color_map[val] = (r, g, b, 1.0)
    
    if not use_membership_coloring:
        # Unknown gets gray, semi-transparent
        color_map[UNKNOWN_LABEL] = (0.5, 0.5, 0.5, 0.5)
    
    # t-SNE layout
    X = np.stack(cluster_df['histogram'].values)
    X = X.reshape(X.shape[0], -1)
    perplexity = min(30, max(5, len(cluster_df) // 4))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(X)
    coords = coords * 20 / coords.std()
    
    # Determine zoom based on cluster size
    if zoom is not None:
        pass
    elif cluster_size < 50:
        zoom = 0.3
    elif cluster_size < 200:
        zoom = 0.225
    elif cluster_size < 500:
        zoom = 0.175
    else:
        zoom = 0.125
    
    # Get Chinese font
    font_prop = _find_chinese_font()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Draw edges
        node_to_idx = {node: i for i, node in enumerate(cluster_nodes)}
        for i, j in G_sub.edges():
            i_idx = node_to_idx[i]
            j_idx = node_to_idx[j]
            
            color_i = color_values[i_idx]
            color_j = color_values[j_idx]
            
            # Edge coloring
            if disable_color:
                edge_color = (0.5, 0.5, 0.5, 1.0)
                alpha = 0.2
                linewidth = 1.0
            elif not use_membership_coloring and (color_i == UNKNOWN_LABEL or color_j == UNKNOWN_LABEL):
                edge_color = (0.5, 0.5, 0.5, 1.0)
                alpha = 0.15
                linewidth = 0.8
            elif color_i == color_j:
                edge_color = color_map[color_i]
                alpha = 0.4
                linewidth = 1.5
            else:
                edge_color = (0.5, 0.5, 0.5, 1.0)
                alpha = 0.2
                linewidth = 1.0
            
            ax.plot([coords[i_idx, 0], coords[j_idx, 0]], 
                    [coords[i_idx, 1], coords[j_idx, 1]], 
                    color=edge_color, alpha=alpha, linewidth=linewidth, zorder=1)
        
        # Draw nodes
        degrees = np.array([G_sub.degree(node) for node in cluster_nodes])
        max_degree = degrees.max() if degrees.max() > 0 else 1
        
        for idx, node in enumerate(cluster_nodes):
            color_value = color_values[idx]
            node_color = color_map[color_value] if not disable_color else (0., 0.5, 0.5, 1.0)
            
            # For label display (still uses target_lbl)
            label = dataframe[target_lbl].loc[node] if not use_membership_coloring else None

            if not disable_svg:
                svg_img = dataframe['svg'].loc[node].render(scale=2)
                pil_img = Image.fromarray(svg_img)
                imagebox = OffsetImage(pil_img, zoom=zoom, cmap='gray')
                
                border_width = 1.5 + 2.5 * (degrees[idx] / max_degree)
                
                # Make unknown nodes more transparent (only when coloring by label)
                if not use_membership_coloring:
                    node_alpha = 0.5 if label == UNKNOWN_LABEL else 1.0
                else:
                    node_alpha = 1.0
                
                ab = AnnotationBbox(
                    imagebox, 
                    (coords[idx, 0], coords[idx, 1]),
                    frameon=True,
                    pad=0.0,
                    bboxprops=dict(
                        edgecolor=node_color,
                        linewidth=border_width,
                        facecolor='white',
                        alpha=node_alpha
                    ),
                    zorder=10
                )
                ax.add_artist(ab)
            else:
                # Scatter plot as alternative to SVG images
                size = 50 + 200 * (degrees[idx] / max_degree)
                if not use_membership_coloring:
                    node_alpha = 0.5 if label == UNKNOWN_LABEL else 0.8
                else:
                    node_alpha = 0.8

                ax.scatter(coords[idx, 0], coords[idx, 1], 
                        s=size, 
                        c=[node_color], 
                        alpha=node_alpha,
                        edgecolors='black',
                        linewidths=1.5,
                        zorder=10)
            
            # Add character label below the node (skip for unknown, or if coloring by membership)
            if not use_membership_coloring and label != UNKNOWN_LABEL and not disable_char:
                if disable_svg:
                    size = 50 + 200 * (degrees[idx] / max_degree)
                    text_offset = np.sqrt(size) * 0.15
                else:
                    text_offset = 0.8 * zoom * 10
                
                fontsize = max(8, min(16, 14 * zoom * 4)) if not disable_svg else 10
                
                ax.text(coords[idx, 0], coords[idx, 1] - text_offset, 
                        label,
                        fontsize=fontsize,
                        ha='center', 
                        va='top',
                        fontproperties=font_prop,
                        color=node_color,
                        fontweight='bold',
                        zorder=11,
                        bbox=dict(boxstyle='round,pad=0.2', 
                                facecolor='white', 
                                edgecolor=node_color,
                                alpha=0.8,
                                linewidth=1))        
        # Set limits
        margin = 3
        ax.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + margin)
        ax.set_ylim(coords[:, 1].min() - margin, coords[:, 1].max() + margin)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Title
        if use_membership_coloring:
            title_text = (f'Cluster {cluster_id} (t-SNE layout, colored by membership)\n'
                         f'{cluster_size} patches, {G_sub.number_of_edges()} edges, '
                         f'{len(value_counts)} unique memberships\n'
                         f'Normalized Entropy: {normalized_entropy:.3f}')
        else:
            title_text = (f'Cluster {cluster_id} (t-SNE layout)\n'
                         f'{cluster_size} patches, {G_sub.number_of_edges()} edges, '
                         f'{len(value_counts)} unique known labels')
            if unknown_count > 0:
                title_text += f' ({unknown_count} unknown ▯)'
            title_text += f'\nNormalized Entropy (known only): {normalized_entropy:.3f}'
        
        ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20, fontproperties=font_prop)
        
        # Distribution box
        if use_membership_coloring:
            distribution_text = "Membership Distribution:\n"
            sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
            for val, count in sorted_values[:10]:
                prob = value_probs[val]
                distribution_text += f"Membership {val}: {count} ({prob*100:.1f}%)\n"
            
            if len(sorted_values) > 10:
                distribution_text += f"... and {len(sorted_values) - 10} more\n"
        else:
            distribution_text = "Label Distribution (known):\n"
            sorted_labels = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
            for lbl, count in sorted_labels[:10]:
                prob = value_probs[lbl]
                distribution_text += f"{lbl}: {count} ({prob*100:.1f}%)\n"
            
            if len(sorted_labels) > 10:
                distribution_text += f"... and {len(sorted_labels) - 10} more\n"
            
            if unknown_count > 0:
                distribution_text += f"\n▯ (unknown): {unknown_count} ({unknown_count/cluster_size*100:.1f}% of cluster)"
        
        ax.text(0.02, 0.98, distribution_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                fontproperties=font_prop,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8),
                zorder=100)
        
        plt.tight_layout()
    
    return fig