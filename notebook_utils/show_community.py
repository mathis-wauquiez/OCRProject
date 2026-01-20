import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import gc
import warnings

from PIL import Image
import numpy as np

def show_community(
        selected_comm_idx,
        communities,
        patches_df,
        histograms,
        G,
        show_plot=True,
        save_path=None
):
    size = len(communities[selected_comm_idx])

    selected_nodes = list(communities[selected_comm_idx])

    print(f"\nShowing community {selected_comm_idx}: {len(selected_nodes)} nodes")

    # Create subgraph
    G_sub = G.subgraph(selected_nodes).copy()

    # ============================================
    # t-SNE LAYOUT (instead of spring layout)
    # ============================================

    print("Computing t-SNE layout...")

    # Get the feature vectors for selected nodes
    X_selected = histograms.reshape(histograms.shape[0], -1)[selected_nodes].cpu().numpy()

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(selected_nodes)-1))
    tsne_coords = tsne.fit_transform(X_selected)

    # Scale to similar range as before (optional, for similar visualization size)
    tsne_coords = tsne_coords * 20 / tsne_coords.std()

    # Create position dictionary
    pos = {node: tsne_coords[i] for i, node in enumerate(selected_nodes)}
    coords = tsne_coords

    # ============================================
    # Character-based coloring
    # ============================================

    # Get unique characters in this community
    chars_in_community = [patches_df['char'][node] for node in selected_nodes]
    unique_chars = list(set(chars_in_community))
    unique_chars.sort()  # Sort for consistency

    print(f"Found {len(unique_chars)} unique characters: {unique_chars}")

    # ============================================
    # Empirical Distribution and Normalized Entropy
    # (IGNORING EMPTY STRINGS)
    # ============================================
    
    # Filter out empty strings for entropy calculation
    chars_filtered = [char for char in chars_in_community if char != '']
    
    # Count occurrences of each character (excluding empty strings)
    char_counts = {}
    for char in chars_filtered:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Count empty strings separately for reporting
    empty_count = chars_in_community.count('')
    
    if empty_count > 0:
        print(f"Note: {empty_count} empty string(s) ignored in entropy calculation")
    
    # Build empirical distribution (probabilities)
    total_nodes = len(chars_filtered)  # Only count non-empty characters
    
    if total_nodes > 0:
        char_probs = {char: count / total_nodes for char, count in char_counts.items()}
        
        # Compute Shannon entropy
        entropy = -sum(p * np.log2(p) for p in char_probs.values() if p > 0)
        
        # Compute normalized entropy (divide by max possible entropy)
        # Max entropy occurs when all characters are equally likely
        num_unique_chars = len(char_counts)  # Number of unique non-empty characters
        max_entropy = np.log2(num_unique_chars) if num_unique_chars > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        print(f"Character distribution (excluding ''): {char_counts}")
        print(f"Entropy: {entropy:.3f} bits")
        print(f"Normalized entropy: {normalized_entropy:.3f} (max: {max_entropy:.3f} bits)")
    else:
        # All characters are empty strings
        entropy = 0.0
        max_entropy = 0.0
        normalized_entropy = 0.0
        char_probs = {}
        print("Warning: All characters are empty strings!")

    # Assign a color to each character
    import colorsys
    char_colors = {}
    for i, char in enumerate(unique_chars):
        hue = i / len(unique_chars)  # Evenly space hues
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.8)
        char_colors[char] = [r, g, b, 1.0]

    # Create a mapping from node to color
    node_colors = {node: char_colors[patches_df['char'][node]] for node in selected_nodes}

    # ============================================
    # Find Chinese Font
    # ============================================

    def find_chinese_font():
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
                print(f"✓ Using font: {font_name}")
                return font_name, FontProperties(family=font_name)
        
        # Search for any font with CJK support
        for font_name in available_fonts:
            if any(keyword in font_name.lower() for keyword in ['cjk', 'chinese', 'han', 'hei', 'sung', 'ming', 'kai']):
                print(f"✓ Found CJK font: {font_name}")
                return font_name, FontProperties(family=font_name)
        
        print("⚠ Warning: No Chinese font found. Chinese characters may not display correctly.")
        print("   Consider installing: sudo apt-get install fonts-noto-cjk (Linux)")
        print("   Or: brew install --cask font-noto-sans-cjk (macOS)")
        return None, FontProperties()

    chinese_font, font_prop = find_chinese_font()

    # ============================================
    # Visualization
    # ============================================

    # Adjust display size
    N_sub = len(selected_nodes)
    if N_sub < 50:
        display_size, zoom = 64, 1.2
    elif N_sub < 200:
        display_size, zoom = 48, 0.9
    elif N_sub < 500:
        display_size, zoom = 32, 0.7
    else:
        display_size, zoom = 24, 0.5

    zoom /= 4

    # Create figure
    fig, ax = plt.subplots(figsize=(26, 22), dpi=100)

    # Suppress font warnings during plotting
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')
        
        # Draw edges with Option 2: gray for cross-character edges
        for i, j in G_sub.edges():
            i_idx = selected_nodes.index(i)
            j_idx = selected_nodes.index(j)
            
            # Use gray for edges between different characters
            if patches_df['char'][i] == patches_df['char'][j]:
                edge_color = node_colors[i]  # Same character: use that color
                alpha = 0.4
                linewidth = 1.5
            else:
                edge_color = [0.5, 0.5, 0.5, 1.0]  # Different characters: gray
                alpha = 0.2
                linewidth = 1.0
            
            ax.plot([coords[i_idx, 0], coords[j_idx, 0]], 
                [coords[i_idx, 1], coords[j_idx, 1]], 
                color=edge_color, alpha=alpha, linewidth=linewidth, zorder=1)
        
        # Draw nodes with patches
        degrees_sub = np.array([G_sub.degree(node) for node in selected_nodes])
        
        for idx, node in enumerate(selected_nodes):
            patch = patches_df['svg'][node].render(scale=2)
            pil_img = Image.fromarray(patch)
            
            imagebox = OffsetImage(pil_img, zoom=zoom, cmap='gray')
            
            # Border width by degree
            if degrees_sub.max() > 0:
                border_width = 1.5 + 2.5 * (degrees_sub[idx] / degrees_sub.max())
            else:
                border_width = 1.5
            
            # Use character-specific color
            node_color = node_colors[node]
            
            ab = AnnotationBbox(
                imagebox, 
                (coords[idx, 0], coords[idx, 1]),
                frameon=True,
                pad=0.0,
                bboxprops=dict(
                    edgecolor=node_color,
                    linewidth=border_width,
                    facecolor='white',
                    alpha=1.0
                ),
                zorder=10
            )
            ax.add_artist(ab)
            
            # Add character label below the annotation box
            char_label = patches_df['char'][node]
            # Adjust font size based on zoom level
            fontsize = max(30, min(20, 14 * zoom * 4))
            
            # Position text closer to the image
            text_offset = 0.8 * zoom * 10
            
            ax.text(coords[idx, 0], coords[idx, 1] - text_offset, 
                    char_label,
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
        
        # Finalize plot
        margin = 3
        ax.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + margin)
        ax.set_ylim(coords[:, 1].min() - margin, coords[:, 1].max() + margin)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Enhanced title with entropy information
        num_valid_chars = len(char_counts)
        title_text = (f'Community {selected_comm_idx} (t-SNE layout)\n'
                     f'{N_sub} patches, {G_sub.number_of_edges()} connections, '
                     f'{num_valid_chars} unique characters')
        
        if empty_count > 0:
            title_text += f' ({empty_count} empty ignored)'
        
        title_text += (f'\nNormalized Entropy: {normalized_entropy:.3f} '
                      f'(H={entropy:.2f} bits, max={max_entropy:.2f} bits)')
        
        ax.set_title(title_text, 
                    fontsize=18, fontweight='bold', pad=20, fontproperties=font_prop)
        
        # Add text box with character distribution in the corner
        distribution_text = "Character Distribution:\n"
        # Sort by count (descending) - only non-empty characters
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        for char, count in sorted_chars:
            prob = char_probs[char]
            distribution_text += f"{char}: {count} ({prob*100:.1f}%)\n"
        
        if empty_count > 0:
            distribution_text += f"\n(ignored): {empty_count}"
        
        # Add text box to plot - use Chinese font
        ax.text(0.02, 0.98, distribution_text,
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment='top',
                fontproperties=font_prop,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8),
                zorder=100)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"✓ Saved to: {save_path}")
        
        # Show or close
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    print(f"✓ Visualization complete")
    
    return fig