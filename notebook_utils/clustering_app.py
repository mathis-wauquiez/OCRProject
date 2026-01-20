"""
Clustering Comparison Dashboard - High-Dimensional Version
Optimized for comparisons with ~2,000 clusters with enhanced features
Now with SVG rendering support for cluster labels!
"""

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from sklearn.metrics import (
    adjusted_rand_score, 
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure
)
from collections import Counter
import networkx as nx
import base64
from io import BytesIO

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def svg_to_base64_thumbnail(svg_obj, size=(60, 60)):
    """
    Convert SVG object to base64 encoded thumbnail image
    
    Parameters
    ----------
    svg_obj : SVG object
        SVG object with render() method
    size : tuple
        Target size (width, height)
    
    Returns
    -------
    str : base64 encoded PNG image with data URI prefix
    """
    try:
        # Render SVG to numpy array
        img_array = svg_obj.render(
            output_size=size,
            background_color=(255, 255, 255, 255),
            scale=1.0,
            output_format='RGBA'
        )
        
        # Convert to PIL Image
        from PIL import Image
        img = Image.fromarray(img_array)
        
        # Save to bytes
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Warning: Could not render SVG: {e}")
        return None


def find_majority_character_svg(c1_label, c1_array, c2_array, svg_series):
    """
    For a C1 cluster, find the majority C2 character and return a representative SVG
    
    Parameters
    ----------
    c1_label : cluster label for C1
    c1_array : array of C1 labels
    c2_array : array of C2 labels  
    svg_series : pandas Series of SVG objects
    
    Returns
    -------
    tuple : (majority_c2_label, svg_object) or (None, None) if not found
    """
    # Find all samples in this C1 cluster
    mask = c1_array == c1_label
    
    if not np.any(mask):
        return None, None
    
    # Get C2 labels for this cluster
    c2_in_cluster = c2_array[mask]
    
    # Find most common C2 label
    c2_counter = Counter(c2_in_cluster)
    if not c2_counter:
        return None, None
    
    majority_c2, count = c2_counter.most_common(1)[0]
    
    # Find a sample that has this C1 label AND the majority C2 label
    match_mask = mask & (c2_array == majority_c2)
    match_indices = np.where(match_mask)[0]
    
    if len(match_indices) == 0:
        return majority_c2, None
    
    # Get the first matching SVG
    idx = match_indices[0]
    svg_obj = svg_series.iloc[idx]
    
    return majority_c2, svg_obj


def create_svg_thumbnail_map(c1_array, c2_array, svg_series, top_clusters, size=(60, 60)):
    """
    Create a mapping from C1 cluster labels to SVG thumbnails based on majority C2 character
    
    Parameters
    ----------
    c1_array : array of C1 labels
    c2_array : array of C2 labels
    svg_series : pandas.Series containing SVG objects
    top_clusters : list of cluster labels to process
    size : tuple, thumbnail size
    
    Returns
    -------
    dict : Mapping from C1 cluster label to (majority_c2_label, base64_image, count, percentage)
    """
    thumbnail_map = {}
    
    print(f"Generating SVG thumbnails for {len(top_clusters)} clusters...")
    
    for i, c1_label in enumerate(top_clusters):
        if i % 20 == 0:
            print(f"  Processing cluster {i+1}/{len(top_clusters)}...")
        
        majority_c2, svg_obj = find_majority_character_svg(c1_label, c1_array, c2_array, svg_series)
        
        if svg_obj is None:
            continue
        
        # Render thumbnail
        thumbnail_b64 = svg_to_base64_thumbnail(svg_obj, size=size)
        
        if thumbnail_b64:
            # Calculate percentage
            mask = c1_array == c1_label
            c2_in_cluster = c2_array[mask]
            majority_count = np.sum(c2_in_cluster == majority_c2)
            total_count = len(c2_in_cluster)
            percentage = (majority_count / total_count * 100) if total_count > 0 else 0
            
            thumbnail_map[c1_label] = {
                'majority_c2': majority_c2,
                'image': thumbnail_b64,
                'count': majority_count,
                'percentage': percentage
            }
    
    print(f"Generated {len(thumbnail_map)} SVG thumbnails")
    return thumbnail_map


def compute_similarity_matrix(clustering1, clustering2):
    """
    Compute similarity matrix between two clusterings.
    Returns sparse representation for efficiency with many clusters.
    """
    unique1 = np.unique(clustering1)
    unique2 = np.unique(clustering2)
    
    # Use pandas crosstab for efficiency
    df = pd.DataFrame({'c1': clustering1, 'c2': clustering2})
    matrix = pd.crosstab(df['c1'], df['c2'])
    
    # Ensure all clusters are represented
    for label in unique1:
        if label not in matrix.index:
            matrix.loc[label] = 0
    for label in unique2:
        if label not in matrix.columns:
            matrix[label] = 0
    
    matrix = matrix.sort_index().sort_index(axis=1)
    
    return matrix


def compute_metrics(clustering1, clustering2):
    """Compute comparison metrics"""
    return {
        'ARI': adjusted_rand_score(clustering1, clustering2),
        'NMI': normalized_mutual_info_score(clustering1, clustering2)
    }


def compute_homogeneity_metrics(clustering, ground_truth):
    """Compute homogeneity metrics"""
    h, c, v = homogeneity_completeness_v_measure(ground_truth, clustering)
    return {'homogeneity': h, 'completeness': c, 'v_measure': v}


def get_cluster_statistics(clustering):
    """Get detailed statistics about cluster sizes"""
    counter = Counter(clustering)
    sizes = np.array(list(counter.values()))
    
    return {
        'n_clusters': len(counter),
        'mean_size': np.mean(sizes),
        'median_size': np.median(sizes),
        'min_size': np.min(sizes),
        'max_size': np.max(sizes),
        'std_size': np.std(sizes),
        'sizes': sizes,
        'labels': list(counter.keys()),
        'counts': list(counter.values())
    }


def suggest_network_parameters(n_clusters_c1, n_clusters_c2):
    """Suggest good default parameters based on cluster counts"""
    total_clusters = n_clusters_c1 + n_clusters_c2
    
    if total_clusters < 100:
        return {'top_n': total_clusters // 2, 'min_pct': 0.5}
    elif total_clusters < 500:
        return {'top_n': 100, 'min_pct': 1}
    elif total_clusters < 2000:
        return {'top_n': 150, 'min_pct': 2}
    else:
        return {'top_n': 200, 'min_pct': 3}


def create_cluster_size_distribution(clustering, name, color):
    """Create histogram of cluster sizes"""
    stats = get_cluster_statistics(clustering)
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=stats['sizes'],
        nbinsx=50,
        name='Cluster sizes',
        marker_color=color
    ))
    
    # Add vertical lines for statistics
    fig.add_vline(x=stats['mean_size'], line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {stats['mean_size']:.1f}")
    fig.add_vline(x=stats['median_size'], line_dash="dash", line_color="green",
                  annotation_text=f"Median: {stats['median_size']:.1f}")
    
    fig.update_layout(
        title=f"{name} - Cluster Size Distribution ({stats['n_clusters']} clusters)",
        xaxis_title="Cluster Size",
        yaxis_title="Frequency",
        height=400,
        showlegend=False
    )
    
    return fig


def create_top_clusters_comparison(similarity_matrix, top_n=50):
    """
    Create heatmap showing only top N clusters by total overlap.
    """
    # Calculate total overlap for each cluster
    c1_totals = similarity_matrix.sum(axis=1).sort_values(ascending=False)
    c2_totals = similarity_matrix.sum(axis=0).sort_values(ascending=False)
    
    # Select top clusters
    top_c1 = c1_totals.head(top_n).index
    top_c2 = c2_totals.head(top_n).index
    
    # Filter matrix
    filtered = similarity_matrix.loc[top_c1, top_c2]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=filtered.values,
        x=[f'C2-{i}' for i in filtered.columns],
        y=[f'C1-{i}' for i in filtered.index],
        colorscale='Viridis',
        hovertemplate='C1: %{y}<br>C2: %{x}<br>Overlap: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Clusters by Size (Overlap Counts)",
        xaxis_title="C2 Clusters",
        yaxis_title="C1 Clusters",
        height=700,
        xaxis={'tickangle': 45}
    )
    
    return fig


def create_overlap_distribution(similarity_matrix):
    """
    Analyze the distribution of overlaps between clusters.
    """
    # Get all non-zero overlaps
    overlaps = similarity_matrix.values.flatten()
    overlaps = overlaps[overlaps > 0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=overlaps,
        nbinsx=50,
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title=f"Distribution of Cluster Overlaps ({len(overlaps)} non-zero overlaps)",
        xaxis_title="Overlap Size",
        yaxis_title="Frequency",
        height=400,
        xaxis_type='log'
    )
    
    return fig


def create_matching_quality_analysis(similarity_matrix):
    """
    Analyze how well clusters match between the two clusterings.
    """
    # For each C1 cluster, find best match percentage
    c1_best_match_pct = []
    for i in range(len(similarity_matrix)):
        total = similarity_matrix.iloc[i].sum()
        if total > 0:
            best = similarity_matrix.iloc[i].max()
            c1_best_match_pct.append(best / total * 100)
    
    # For each C2 cluster, find best match percentage
    c2_best_match_pct = []
    for j in range(len(similarity_matrix.columns)):
        total = similarity_matrix.iloc[:, j].sum()
        if total > 0:
            best = similarity_matrix.iloc[:, j].max()
            c2_best_match_pct.append(best / total * 100)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "C1 → C2 Best Match %",
        "C2 → C1 Best Match %"
    ))
    
    fig.add_trace(go.Histogram(
        x=c1_best_match_pct,
        nbinsx=20,
        name='C1 → C2',
        marker_color='#636EFA'
    ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        x=c2_best_match_pct,
        nbinsx=20,
        name='C2 → C1',
        marker_color='#EF553B'
    ), row=1, col=2)
    
    fig.update_layout(
        title="How well do clusters match?",
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Best Match %", row=1, col=1)
    fig.update_xaxes(title_text="Best Match %", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    return fig


def create_best_matches_table(similarity_matrix, top_n=100):
    """
    Create a table of best matches for each clustering direction.
    """
    # C1 to C2 best matches
    c1_matches = []
    for i, idx in enumerate(similarity_matrix.index):
        if similarity_matrix.iloc[i].sum() > 0:
            best_c2_idx = similarity_matrix.iloc[i].idxmax()
            overlap = similarity_matrix.iloc[i].max()
            total = similarity_matrix.iloc[i].sum()
            pct = overlap / total * 100
            
            c1_matches.append({
                'C1': str(idx),
                'C1_size': int(total),
                'Best_C2': str(best_c2_idx),
                'Overlap': int(overlap),
                'Match_%': f"{pct:.1f}"
            })
    
    # Sort by C1 size and take top N
    c1_df = pd.DataFrame(c1_matches).sort_values('C1_size', ascending=False).head(top_n)
    
    # C2 to C1 best matches
    c2_matches = []
    for j, col in enumerate(similarity_matrix.columns):
        if similarity_matrix.iloc[:, j].sum() > 0:
            best_c1_idx = similarity_matrix.iloc[:, j].idxmax()
            overlap = similarity_matrix.iloc[:, j].max()
            total = similarity_matrix.iloc[:, j].sum()
            pct = overlap / total * 100
            
            c2_matches.append({
                'C2': str(col),
                'C2_size': int(total),
                'Best_C1': str(best_c1_idx),
                'Overlap': int(overlap),
                'Match_%': f"{pct:.1f}"
            })
    
    # Sort by C2 size and take top N
    c2_df = pd.DataFrame(c2_matches).sort_values('C2_size', ascending=False).head(top_n)
    
    return c1_df, c2_df


def find_orphan_clusters(similarity_matrix, threshold=5):
    """Find clusters with very few connections (orphans)"""
    c1_orphans = []
    c2_orphans = []
    
    for i, idx in enumerate(similarity_matrix.index):
        n_connections = np.sum(similarity_matrix.iloc[i].values > 0)
        if n_connections <= threshold:
            c1_orphans.append({
                'Cluster': f'C1-{idx}',
                'Size': int(similarity_matrix.iloc[i].sum()),
                'Connections': n_connections
            })
    
    for j, col in enumerate(similarity_matrix.columns):
        n_connections = np.sum(similarity_matrix.iloc[:, j].values > 0)
        if n_connections <= threshold:
            c2_orphans.append({
                'Cluster': f'C2-{col}',
                'Size': int(similarity_matrix.iloc[:, j].sum()),
                'Connections': n_connections
            })
    
    return pd.DataFrame(c1_orphans), pd.DataFrame(c2_orphans)


def create_cluster_network_graph(similarity_matrix, top_n=100, min_overlap_pct=1, 
                                 filter_empty=True, c1_svg_map=None):
    """
    Create a network graph showing connections between clusters from two clusterings.
    Uses smart filtering to keep it fast.
    
    Parameters
    ----------
    similarity_matrix : pd.DataFrame
        Similarity matrix where rows are C1 clusters and columns are C2 clusters
    top_n : int
        Show only top N clusters from each clustering (default: 100)
    min_overlap_pct : float
        Minimum overlap percentage to show an edge (default: 1%)
    filter_empty : bool
        If True, filter out empty string "" clusters (default: True)
    c1_svg_map : dict, optional
        Mapping from C1 cluster labels to SVG info dict with 'image', 'majority_c2', etc.
    
    Returns
    -------
    fig : plotly.graph_objs.Figure
        Network graph visualization
    network_stats : dict
        Statistics about the network
    """
    # Filter out empty string clusters if requested
    filtered_matrix = similarity_matrix.copy()
    n_filtered_c1 = 0
    n_filtered_c2 = 0
    
    if filter_empty:
        # Remove "" from C1 (rows)
        if "" in filtered_matrix.index:
            n_filtered_c1 = 1
            filtered_matrix = filtered_matrix.drop("", axis=0)
        
        # Remove "" from C2 (columns)
        if "" in filtered_matrix.columns:
            n_filtered_c2 = 1
            filtered_matrix = filtered_matrix.drop("", axis=1)
        
        if n_filtered_c1 > 0 or n_filtered_c2 > 0:
            print(f"Filtered out {n_filtered_c1} empty C1 cluster(s) and {n_filtered_c2} empty C2 cluster(s)")
    
    # Select top clusters by size
    c1_totals = filtered_matrix.sum(axis=1).nlargest(top_n)
    c2_totals = filtered_matrix.sum(axis=0).nlargest(top_n)
    
    # Filter matrix to top clusters
    filtered = filtered_matrix.loc[c1_totals.index, c2_totals.index]
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes for C1 clusters
    c1_nodes = [f'C1-{idx}' for idx in filtered.index]
    G.add_nodes_from(c1_nodes)
    
    # Add nodes for C2 clusters
    c2_nodes = [f'C2-{col}' for col in filtered.columns]
    G.add_nodes_from(c2_nodes)
    
    # Add edges with percentage filtering
    edge_count = 0
    total_overlap = 0
    for i, c1_idx in enumerate(filtered.index):
        c1_total = filtered.iloc[i].sum()
        for j, c2_idx in enumerate(filtered.columns):
            overlap = filtered.iloc[i, j]
            if overlap > 0:
                # Calculate percentage from both perspectives
                pct_c1 = (overlap / c1_total * 100) if c1_total > 0 else 0
                c2_total = filtered.iloc[:, j].sum()
                pct_c2 = (overlap / c2_total * 100) if c2_total > 0 else 0
                
                # Include edge if significant from either perspective
                if pct_c1 >= min_overlap_pct or pct_c2 >= min_overlap_pct:
                    G.add_edge(f'C1-{c1_idx}', f'C2-{c2_idx}', weight=overlap)
                    edge_count += 1
                    total_overlap += overlap
    
    print(f"Network: {len(G.nodes())} nodes, {edge_count} edges")
    
    # Use spring layout with edge weights - IMPROVED PARAMETERS
    pos = nx.spring_layout(G, weight='weight', k=None, iterations=500, seed=42)
    
    # Normalize positions to [0, 1] range for easier image placement
    x_coords = [pos[node][0] for node in pos]
    y_coords = [pos[node][1] for node in pos]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.1
    
    for node in pos:
        pos[node] = (
            (pos[node][0] - x_min) / x_range * (1 - 2*padding) + padding,
            (pos[node][1] - y_min) / y_range * (1 - 2*padding) + padding
        )
    
    # Calculate network statistics
    if edge_count > 0:
        avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
        density = nx.density(G)
    else:
        avg_degree = 0
        density = 0
    
    network_stats = {
        'n_nodes': len(G.nodes()),
        'n_edges': edge_count,
        'avg_degree': avg_degree,
        'density': density,
        'total_overlap': int(total_overlap),
        'filtered_c1': n_filtered_c1,
        'filtered_c2': n_filtered_c2
    }
    
    # Normalize edge weights for visualization
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    if edge_weights:
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
    else:
        max_weight = min_weight = 1
    
    def normalize_weight(w, new_min=0.5, new_max=8):
        if max_weight == min_weight:
            return (new_min + new_max) / 2
        return new_min + (w - min_weight) * (new_max - new_min) / (max_weight - min_weight)
    
    def normalize_opacity(w, new_min=0.2, new_max=0.8):
        if max_weight == min_weight:
            return (new_min + new_max) / 2
        return new_min + (w - min_weight) * (new_max - new_min) / (max_weight - min_weight)
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=normalize_weight(weight),
                color=f'rgba(150, 150, 150, {normalize_opacity(weight)})'
            ),
            hoverinfo='text',
            text=f'{edge[0]} ↔ {edge[1]}<br>Overlap: {int(weight):,} samples',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Calculate node sizes based on cluster sizes
    c1_sizes = filtered.sum(axis=1)
    c2_sizes = filtered.sum(axis=0)
    
    # Combine all sizes for normalization
    all_sizes = list(c1_sizes.values) + list(c2_sizes.values)
    
    def scale_node_size(size, min_size=15, max_size=60):
        if len(all_sizes) == 0 or max(all_sizes) == min(all_sizes):
            return min_size
        return min_size + (size - min(all_sizes)) * (max_size - min_size) / (max(all_sizes) - min(all_sizes))
    
    # Create node traces for C1
    c1_x = [pos[node][0] for node in c1_nodes if node in pos]
    c1_y = [pos[node][1] for node in c1_nodes if node in pos]
    c1_labels = [str(node).replace('C1-', '') for node in c1_nodes if node in pos]
    c1_indices = [idx for idx in filtered.index if f'C1-{idx}' in pos]
    c1_node_sizes = [scale_node_size(c1_sizes.loc[idx]) for idx in c1_indices]
    
    # Calculate node strength (total connections)
    c1_strength = {}
    for node in c1_nodes:
        if node in G:
            c1_strength[node] = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
        else:
            c1_strength[node] = 0
    
    # Build hover text
    c1_hovertext = []
    for idx in c1_indices:
        node_name = f'C1-{idx}'
        size = int(c1_sizes.loc[idx])
        strength = int(c1_strength.get(node_name, 0))
        n_connections = len(list(G.neighbors(node_name))) if node_name in G else 0
        
        hover_parts = [
            f'<b>{node_name}</b>',
            f'Cluster size: {size:,}',
            f'Total overlap: {strength:,}',
            f'Connections: {n_connections}'
        ]
        
        # Add majority character info if available
        if c1_svg_map and idx in c1_svg_map:
            svg_info = c1_svg_map[idx]
            hover_parts.insert(1, f'Majority char: "{svg_info["majority_c2"]}" ({svg_info["percentage"]:.1f}%)')
        
        c1_hovertext.append('<br>'.join(hover_parts))
    
    # For C1 nodes with SVGs, make them invisible (SVG will show instead)
    c1_node_trace = go.Scatter(
        x=c1_x, y=c1_y,
        mode='markers',
        marker=dict(
            size=c1_node_sizes,
            color='#636EFA',
            opacity=0.3 if c1_svg_map else 1.0,  # Make semi-transparent if showing SVGs
            line=dict(width=2, color='white')
        ),
        hovertext=c1_hovertext,
        hoverinfo='text',
        name='C1 Clusters',
        showlegend=True
    )
    
    # Create node traces for C2
    c2_x = [pos[node][0] for node in c2_nodes if node in pos]
    c2_y = [pos[node][1] for node in c2_nodes if node in pos]
    c2_labels = [str(node).replace('C2-', '') for node in c2_nodes if node in pos]
    c2_indices = [col for col in filtered.columns if f'C2-{col}' in pos]
    c2_node_sizes = [scale_node_size(c2_sizes.loc[col]) for col in c2_indices]
    
    c2_strength = {}
    for node in c2_nodes:
        if node in G:
            c2_strength[node] = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
        else:
            c2_strength[node] = 0
    
    c2_hovertext = []
    for col in c2_indices:
        node_name = f'C2-{col}'
        size = int(c2_sizes.loc[col])
        strength = int(c2_strength.get(node_name, 0))
        n_connections = len(list(G.neighbors(node_name))) if node_name in G else 0
        c2_hovertext.append(
            f'<b>{node_name}</b><br>'
            f'Character: "{col}"<br>'
            f'Cluster size: {size:,}<br>'
            f'Total overlap: {strength:,}<br>'
            f'Connections: {n_connections}'
        )
    
    c2_node_trace = go.Scatter(
        x=c2_x, y=c2_y,
        mode='markers+text',
        marker=dict(
            size=c2_node_sizes,
            color='#EF553B',
            line=dict(width=2, color='white')
        ),
        text=c2_labels,
        textposition='middle center',
        textfont=dict(size=9, color='white', family='Arial Black'),
        hovertext=c2_hovertext,
        hoverinfo='text',
        name='C2 Clusters',
        showlegend=True
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [c1_node_trace, c2_node_trace])
    
    # Add SVG images as overlays
    if c1_svg_map:
        images = []
        image_size = 0.04  # Size in plot coordinates
        
        for idx in c1_indices:
            if idx in c1_svg_map:
                node_name = f'C1-{idx}'
                if node_name in pos:
                    x, y = pos[node_name]
                    svg_info = c1_svg_map[idx]
                    
                    images.append(dict(
                        source=svg_info['image'],
                        xref="x",
                        yref="y",
                        x=x,
                        y=y,
                        sizex=image_size,
                        sizey=image_size,
                        xanchor="center",
                        yanchor="middle",
                        layer="above"
                    ))
        
        fig.update_layout(images=images)
    
    filter_note = f" (excluding empty strings)" if filter_empty and (n_filtered_c1 > 0 or n_filtered_c2 > 0) else ""
    svg_note = f" with {len(c1_svg_map)} SVG glyphs" if c1_svg_map else ""
    
    fig.update_layout(
        title=dict(
            text=f'<b>Cluster Network{filter_note}{svg_note}</b><br>'
                 f'<sub>Top {top_n} clusters, {min_overlap_pct}%+ overlap | C1 shows majority character glyph | C2 shows character label</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=80),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-0.05, 1.05]
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-0.05, 1.05],
            scaleanchor="x",
            scaleratio=1
        ),
        height=700,
        plot_bgcolor='rgba(240,240,240,1)',
        paper_bgcolor='rgba(240,240,240,1)',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig, network_stats


def analyze_cluster_fragmentation(similarity_matrix):
    """
    Analyze how fragmented clusters are across the two clusterings.
    """
    # For each C1 cluster, count how many C2 clusters it maps to
    c1_fragmentation = []
    for i in range(len(similarity_matrix)):
        n_mapped = np.sum(similarity_matrix.iloc[i].values > 0)
        total = similarity_matrix.iloc[i].sum()
        if total > 0:
            c1_fragmentation.append(n_mapped)
    
    # For each C2 cluster, count how many C1 clusters it maps to
    c2_fragmentation = []
    for j in range(len(similarity_matrix.columns)):
        n_mapped = np.sum(similarity_matrix.iloc[:, j].values > 0)
        total = similarity_matrix.iloc[:, j].sum()
        if total > 0:
            c2_fragmentation.append(n_mapped)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "C1 clusters map to how many C2 clusters?",
        "C2 clusters map to how many C1 clusters?"
    ))
    
    fig.add_trace(go.Histogram(
        x=c1_fragmentation,
        nbinsx=30,
        marker_color='#636EFA'
    ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        x=c2_fragmentation,
        nbinsx=30,
        marker_color='#EF553B'
    ), row=1, col=2)
    
    fig.update_layout(
        title="Cluster Fragmentation Analysis",
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Number of mapped clusters", row=1, col=1)
    fig.update_xaxes(title_text="Number of mapped clusters", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    return fig


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def compare_clusterings(c1, c2, ground_truth=None, c1_svgs=None, c2_labels=None, mode='inline', port=8050, height=900):
    """
    Launch an interactive dashboard to compare two high-dimensional clusterings.
    
    Parameters
    ----------
    c1 : array-like, shape (n_samples,)
        First clustering assignment (optimized for ~2000 clusters)
    c2 : array-like, shape (n_samples,)
        Second clustering assignment (optimized for ~2000 clusters)
    ground_truth : array-like, shape (n_samples,), optional
        Ground truth labels for comparison
    c1_svgs : pandas.Series, optional
        Series of SVG objects for samples (indexed same as c1)
        For each C1 cluster, will show the SVG of majority C2 character
    c2_labels : array-like, optional
        C2 labels (can be same as c2, used for finding majority character)
        If None, uses c2
    mode : str, default='inline'
        Display mode: 'inline', 'external', or 'jupyterlab'
    port : int, default=8050
        Port number for the server
    height : int, default=900
        Height of inline display in pixels
    
    Returns
    -------
    app : JupyterDash
        The Dash app instance
    """
    
    # Convert to numpy arrays
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    if c2_labels is None:
        c2_labels = c2
    else:
        c2_labels = np.asarray(c2_labels)
    
    if ground_truth is not None:
        ground_truth = np.asarray(ground_truth)
    
    # Validate inputs
    assert len(c1) == len(c2), "Clusterings must have the same length"
    if ground_truth is not None:
        assert len(c1) == len(ground_truth), "Ground truth must have the same length"
    if c1_svgs is not None:
        assert len(c1) == len(c1_svgs), "c1_svgs must have the same length as c1"
    
    print("=" * 60)
    print("COMPUTING METRICS FOR HIGH-DIMENSIONAL CLUSTERING COMPARISON")
    print("=" * 60)
    
    # Get statistics
    c1_stats = get_cluster_statistics(c1)
    c2_stats = get_cluster_statistics(c2)
    
    print(f"\nC1: {c1_stats['n_clusters']} clusters")
    print(f"  - Mean size: {c1_stats['mean_size']:.1f}")
    print(f"  - Median size: {c1_stats['median_size']:.1f}")
    print(f"  - Range: [{c1_stats['min_size']}, {c1_stats['max_size']}]")
    
    print(f"\nC2: {c2_stats['n_clusters']} clusters")
    print(f"  - Mean size: {c2_stats['mean_size']:.1f}")
    print(f"  - Median size: {c2_stats['median_size']:.1f}")
    print(f"  - Range: [{c2_stats['min_size']}, {c2_stats['max_size']}]")
    
    print("\nComputing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(c1, c2)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Store data for callbacks
    global _callback_data
    _callback_data = {
        'similarity_matrix': similarity_matrix,
        'c1_array': c1,
        'c2_array': c2_labels,
        'svg_series': c1_svgs
    }
    
    # Check for empty string clusters
    has_empty_c1 = "" in similarity_matrix.index
    has_empty_c2 = "" in similarity_matrix.columns
    if has_empty_c1 or has_empty_c2:
        empty_c1_size = int(similarity_matrix.loc[""].sum()) if has_empty_c1 else 0
        empty_c2_size = int(similarity_matrix[""].sum()) if has_empty_c2 else 0
        print(f"\n⚠️  Empty string clusters detected:")
        if has_empty_c1:
            print(f"  - C1: {empty_c1_size:,} samples")
        if has_empty_c2:
            print(f"  - C2: {empty_c2_size:,} samples")
        print("  You can filter these out in the network graph settings.")
    
    print("\nComputing metrics...")
    metrics = compute_metrics(c1, c2)
    
    if ground_truth is not None:
        c1_gt_metrics = compute_homogeneity_metrics(c1, ground_truth)
        c2_gt_metrics = compute_homogeneity_metrics(c2, ground_truth)
    else:
        c1_gt_metrics = c2_gt_metrics = None
    
    print(f"\nARI: {metrics['ARI']:.4f}")
    print(f"NMI: {metrics['NMI']:.4f}")
    
    # Get suggested parameters
    suggested = suggest_network_parameters(c1_stats['n_clusters'], c2_stats['n_clusters'])
    print(f"\nSuggested network parameters:")
    print(f"  - Top N clusters: {suggested['top_n']}")
    print(f"  - Min overlap %: {suggested['min_pct']}")
    
    print("\nPreparing visualizations...")
    
    # Precompute all figures EXCEPT network (computed on-demand)
    c1_dist_fig = create_cluster_size_distribution(c1, "C1", "#636EFA")
    c2_dist_fig = create_cluster_size_distribution(c2, "C2", "#EF553B")
    top_clusters_fig = create_top_clusters_comparison(similarity_matrix, top_n=50)
    overlap_dist_fig = create_overlap_distribution(similarity_matrix)
    matching_quality_fig = create_matching_quality_analysis(similarity_matrix)
    fragmentation_fig = analyze_cluster_fragmentation(similarity_matrix)
    
    # Create tables
    c1_matches_df, c2_matches_df = create_best_matches_table(similarity_matrix, top_n=100)
    
    # Find orphan clusters
    c1_orphans_df, c2_orphans_df = find_orphan_clusters(similarity_matrix, threshold=5)
    
    print("=" * 60)
    print("DASHBOARD READY")
    print("=" * 60)
    
    # Create app
    app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Build ground truth section conditionally
    ground_truth_section = []
    if ground_truth is not None:
        ground_truth_section = [
            html.H3("Comparison with Ground Truth", className="mt-4 mb-3"),
            dbc.Card([
                dbc.CardHeader("Ground Truth Metrics"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("C1 vs Ground Truth"),
                            html.P(f"Homogeneity: {c1_gt_metrics['homogeneity']:.3f}"),
                            html.P(f"Completeness: {c1_gt_metrics['completeness']:.3f}"),
                            html.P(f"V-Measure: {c1_gt_metrics['v_measure']:.3f}"),
                        ], width=6),
                        dbc.Col([
                            html.H5("C2 vs Ground Truth"),
                            html.P(f"Homogeneity: {c2_gt_metrics['homogeneity']:.3f}"),
                            html.P(f"Completeness: {c2_gt_metrics['completeness']:.3f}"),
                            html.P(f"V-Measure: {c2_gt_metrics['v_measure']:.3f}"),
                        ], width=6),
                    ])
                ])
            ], className="mb-4")
        ]
    
    app.layout = dbc.Container([
        html.H1("High-Dimensional Clustering Comparison", className="text-center my-4"),
        
        # Summary cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Overall Metrics"),
                    dbc.CardBody([
                        html.H4(f"ARI: {metrics['ARI']:.4f}"),
                        html.H4(f"NMI: {metrics['NMI']:.4f}"),
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("C1 Statistics"),
                    dbc.CardBody([
                        html.P(f"Clusters: {c1_stats['n_clusters']}"),
                        html.P(f"Mean size: {c1_stats['mean_size']:.1f}"),
                        html.P(f"Median size: {c1_stats['median_size']:.1f}"),
                        html.P("🖼️ SVG glyphs: ✓" if c1_svgs is not None else "SVG glyphs: ✗", 
                               className="text-success" if c1_svgs is not None else "text-muted"),
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("C2 Statistics"),
                    dbc.CardBody([
                        html.P(f"Clusters: {c2_stats['n_clusters']}"),
                        html.P(f"Mean size: {c2_stats['mean_size']:.1f}"),
                        html.P(f"Median size: {c2_stats['median_size']:.1f}"),
                    ])
                ])
            ], width=4),
        ], className="mb-4"),
        
        # Ground truth comparison (conditionally included)
        *ground_truth_section,
        
        # Cluster size distributions
        html.H3("Cluster Size Distributions", className="mt-4 mb-3"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=c1_dist_fig)], width=6),
            dbc.Col([dcc.Graph(figure=c2_dist_fig)], width=6),
        ], className="mb-4"),
        
        # Matching quality
        html.H3("Matching Quality Analysis", className="mt-4 mb-3"),
        dcc.Graph(figure=matching_quality_fig, className="mb-4"),
        
        # Fragmentation
        html.H3("Cluster Fragmentation", className="mt-4 mb-3"),
        html.P("How many clusters in one clustering does each cluster in the other map to?"),
        dcc.Graph(figure=fragmentation_fig, className="mb-4"),

        # Cluster network graph (INTERACTIVE - computed on demand)
        html.H3("Interactive Cluster Network", className="mt-4 mb-3"),
        dbc.Card([
            dbc.CardBody([
                dbc.Alert([
                    html.H5("💡 Suggested Parameters", className="alert-heading"),
                    html.P(f"Based on your data size, we suggest: Top {suggested['top_n']} clusters, {suggested['min_pct']}% minimum overlap"),
                    html.P("🖼️ C1 nodes will display SVG glyphs of their majority character!" if c1_svgs is not None else "", 
                           className="mb-0 text-success") if c1_svgs is not None else None,
                ], color="info", className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Top N clusters to show:"),
                        dbc.Input(
                            id='network-top-n',
                            type='number',
                            value=suggested['top_n'],
                            min=10,
                            max=min(500, c1_stats['n_clusters'], c2_stats['n_clusters']),
                            step=10,
                            placeholder=f"Enter 10-{min(500, c1_stats['n_clusters'])}"
                        ),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Minimum overlap % to show edge:"),
                        dbc.Input(
                            id='network-min-pct',
                            type='number',
                            value=suggested['min_pct'],
                            min=0,
                            max=50,
                            step=0.5,
                            placeholder="Enter 0-50"
                        ),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Filter empty strings (\"\"):"),
                        dbc.Checklist(
                            id='filter-empty-checkbox',
                            options=[{'label': ' Remove \"\" clusters', 'value': 'filter'}],
                            value=['filter'] if has_empty_c1 or has_empty_c2 else [],
                            switch=True,
                        ),
                    ], width=4),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Button('Generate Network', id='generate-network-button', color="primary", className="w-100 mt-3", size="lg"),
                    ], width=12),
                ], className="mt-2"),
                html.Div(id='network-stats', className="mt-3"),
            ])
        ], className="mb-3"),
        html.Div(id='network-loading', className="text-center mb-2"),
        dcc.Graph(id='cluster-network', className="mb-4"),
        
        # Top clusters heatmap
        html.H3("Top 50 Clusters Overlap", className="mt-4 mb-3"),
        dcc.Graph(figure=top_clusters_fig, className="mb-4"),
        
        # Overlap distribution
        html.H3("Overlap Distribution", className="mt-4 mb-3"),
        dcc.Graph(figure=overlap_dist_fig, className="mb-4"),
        
        # Orphan clusters
        html.H3("Orphan Clusters (≤5 connections)", className="mt-4 mb-3"),
        dbc.Row([
            dbc.Col([
                html.H5("C1 Orphans"),
                dash_table.DataTable(
                    data=c1_orphans_df.to_dict('records') if not c1_orphans_df.empty else [],
                    columns=[{"name": i, "id": i} for i in ['Cluster', 'Size', 'Connections']],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                ) if not c1_orphans_df.empty else html.P("No orphan clusters found", className="text-muted"),
            ], width=6),
            dbc.Col([
                html.H5("C2 Orphans"),
                dash_table.DataTable(
                    data=c2_orphans_df.to_dict('records') if not c2_orphans_df.empty else [],
                    columns=[{"name": i, "id": i} for i in ['Cluster', 'Size', 'Connections']],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                ) if not c2_orphans_df.empty else html.P("No orphan clusters found", className="text-muted"),
            ], width=6),
        ], className="mb-4"),
        
        # Best matches tables
        html.H3("Best Cluster Matches (Top 100 by size)", className="mt-4 mb-3"),
        dbc.Tabs([
            dbc.Tab([
                html.Div([
                    dash_table.DataTable(
                        data=c1_matches_df.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in c1_matches_df.columns],
                        page_size=20,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ],
                        filter_action="native",
                        sort_action="native",
                        export_format="csv",
                    )
                ], className="p-3")
            ], label="C1 → C2 Mappings"),
            dbc.Tab([
                html.Div([
                    dash_table.DataTable(
                        data=c2_matches_df.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in c2_matches_df.columns],
                        page_size=20,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ],
                        filter_action="native",
                        sort_action="native",
                        export_format="csv",
                    )
                ], className="p-3")
            ], label="C2 → C1 Mappings"),
        ], className="mb-4"),
        
        # Search functionality
        html.H3("Search Specific Clusters", className="mt-4 mb-3"),
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Search C1 Cluster:"),
                        dbc.Input(id='search-c1', type='text', placeholder='Enter C1 cluster ID'),
                    ], width=5),
                    dbc.Col([
                        dbc.Label("Search C2 Cluster:"),
                        dbc.Input(id='search-c2', type='text', placeholder='Enter C2 cluster ID'),
                    ], width=5),
                    dbc.Col([
                        dbc.Label(" "),
                        dbc.Button("Search", id='search-button', color="primary", className="w-100"),
                    ], width=2),
                ]),
                html.Div(id='search-results', className="mt-3")
            ])
        ]),
        
    ], fluid=True)
    
    # Callback for network graph generation
    @app.callback(
        [Output('cluster-network', 'figure'),
         Output('network-loading', 'children'),
         Output('network-stats', 'children')],
        Input('generate-network-button', 'n_clicks'),
        [State('network-top-n', 'value'),
         State('network-min-pct', 'value'),
         State('filter-empty-checkbox', 'value')],
        prevent_initial_call=True
    )
    def generate_network(n_clicks, top_n, min_pct, filter_empty_value):
        if n_clicks:
            try:
                # Validate inputs
                if top_n is None or top_n < 10:
                    return go.Figure(), html.P("⚠️ Please enter a valid number (≥10) for top N", className="text-warning"), ""
                if min_pct is None or min_pct < 0:
                    return go.Figure(), html.P("⚠️ Please enter a valid percentage (≥0)", className="text-warning"), ""
                
                # Check if filtering is enabled
                filter_empty = 'filter' in (filter_empty_value or [])
                
                loading_msg = html.P("⏳ Generating network graph with SVG glyphs (this may take 30-60 seconds)...", className="text-info")
                
                # Generate SVG map if SVGs are available
                c1_svg_map = None
                if _callback_data['svg_series'] is not None:
                    # Get top clusters to render
                    sim_matrix = _callback_data['similarity_matrix']
                    if filter_empty and "" in sim_matrix.index:
                        sim_matrix = sim_matrix.drop("", axis=0)
                    
                    top_c1_labels = sim_matrix.sum(axis=1).nlargest(top_n).index.tolist()
                    
                    c1_svg_map = create_svg_thumbnail_map(
                        _callback_data['c1_array'],
                        _callback_data['c2_array'],
                        _callback_data['svg_series'],
                        top_c1_labels,
                        size=(60, 60)
                    )
                
                fig, stats = create_cluster_network_graph(
                    _callback_data['similarity_matrix'], 
                    top_n=int(top_n), 
                    min_overlap_pct=float(min_pct),
                    filter_empty=filter_empty,
                    c1_svg_map=c1_svg_map
                )
                
                filter_info = ""
                if stats['filtered_c1'] > 0 or stats['filtered_c2'] > 0:
                    filter_info = f" | Filtered: {stats['filtered_c1']} C1 + {stats['filtered_c2']} C2 empty clusters"
                
                svg_info = f" | SVG glyphs: {len(c1_svg_map)} clusters" if c1_svg_map else ""
                
                stats_display = dbc.Alert([
                    html.H6("Network Statistics:", className="alert-heading"),
                    html.P(f"Nodes: {stats['n_nodes']} | Edges: {stats['n_edges']} | Avg Degree: {stats['avg_degree']:.2f} | Density: {stats['density']:.4f}{filter_info}{svg_info}"),
                    html.P(f"Total overlap represented: {stats['total_overlap']:,} samples"),
                ], color="success")
                
                return fig, html.P("✓ Network generated successfully!", className="text-success"), stats_display
            except Exception as e:
                import traceback
                traceback.print_exc()
                return go.Figure(), html.P(f"❌ Error: {str(e)}", className="text-danger"), ""
        return go.Figure(), "", ""
    
    # Callback for search
    @app.callback(
        Output('search-results', 'children'),
        Input('search-button', 'n_clicks'),
        State('search-c1', 'value'),
        State('search-c2', 'value'),
        prevent_initial_call=True
    )
    def search_clusters(n_clicks, c1_id, c2_id):
        results = []
        
        try:
            if c1_id:
                # Convert to appropriate type
                try:
                    if c1_id.isdigit():
                        c1_id = int(c1_id)
                except:
                    pass
                
                if c1_id in similarity_matrix.index:
                    row = similarity_matrix.loc[c1_id]
                    total = row.sum()
                    top_matches = row.nlargest(10)
                    
                    results.append(html.H5(f"C1 Cluster '{c1_id}' (size: {int(total):,})"))
                    results.append(html.P(f"Top 10 C2 matches:"))
                    
                    table_data = []
                    for c2_label, overlap in top_matches.items():
                        if overlap > 0:
                            pct = overlap / total * 100
                            table_data.append({
                                'C2_Cluster': str(c2_label),
                                'Overlap': int(overlap),
                                'Percentage': f"{pct:.1f}%"
                            })
                    
                    results.append(dash_table.DataTable(
                        data=table_data,
                        columns=[{"name": i, "id": i} for i in ['C2_Cluster', 'Overlap', 'Percentage']],
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    ))
                else:
                    results.append(html.P(f"C1 cluster '{c1_id}' not found", className="text-danger"))
            
            if c2_id:
                # Convert to appropriate type
                try:
                    if c2_id.isdigit():
                        c2_id = int(c2_id)
                except:
                    pass
                
                if c2_id in similarity_matrix.columns:
                    col = similarity_matrix[c2_id]
                    total = col.sum()
                    top_matches = col.nlargest(10)
                    
                    results.append(html.H5(f"C2 Cluster '{c2_id}' (size: {int(total):,})", className="mt-3"))
                    results.append(html.P(f"Top 10 C1 matches:"))
                    
                    table_data = []
                    for c1_label, overlap in top_matches.items():
                        if overlap > 0:
                            pct = overlap / total * 100
                            table_data.append({
                                'C1_Cluster': str(c1_label),
                                'Overlap': int(overlap),
                                'Percentage': f"{pct:.1f}%"
                            })
                    
                    results.append(dash_table.DataTable(
                        data=table_data,
                        columns=[{"name": i, "id": i} for i in ['C1_Cluster', 'Overlap', 'Percentage']],
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    ))
                else:
                    results.append(html.P(f"C2 cluster '{c2_id}' not found", className="text-danger"))
        
        except Exception as e:
            results.append(html.P(f"Error: {str(e)}", className="text-danger"))
        
        return results if results else html.P("Enter a cluster ID to search")
    
    # Run the app
    print(f"\nLaunching dashboard in {mode} mode...")
    if mode == 'inline':
        app.run(mode='inline', port=port, height=height)
    elif mode == 'external':
        app.run(mode='external', port=port)
    elif mode == 'jupyterlab':
        app.run(mode='jupyterlab', port=port)
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'inline', 'external', or 'jupyterlab'")
    
    return app