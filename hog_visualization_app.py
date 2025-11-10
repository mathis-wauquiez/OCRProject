"""
Interactive HOG and Patch Matching Visualization Web Application
Supports UMAP and Force-Directed graph layouts with clickable patches
"""

import numpy as np
import torch
from torch import Tensor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from umap import UMAP
import io
import base64
from PIL import Image
from dataclasses import dataclass
from typing import Optional, List, Tuple
import colorsys
import random

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt


# ============================================
# Data Classes (provided by user)
# ============================================

@dataclass
class GradientParameters:
    device: str = "cuda" if torch.cuda.is_available() else 'cpu'
    C: int = 1
    method: str = 'gaussian'
    grdt_sigma: float = 3.5
    ksize_factor: float = 8


@dataclass
class HOGParameters:
    device: str = "cuda" if torch.cuda.is_available() else 'cpu'
    C: int = 1
    partial_output: bool = True
    method: str = 'gaussian'
    grdt_sigma: float = 3.5
    ksize_factor: float = 8
    cell_height: int = 16
    cell_width: int = 16
    psize: int = 128
    num_bins: int = 8
    sigma: float | None = 1
    threshold: float | None = 0.2


@dataclass
class fullHOGOutput:
    dx: Tensor
    dy: Tensor
    patches_grdt_magnitude: Tensor
    patches_grdt_orientation: Tensor
    patches_image: Tensor
    histograms: Tensor


@dataclass
class featureMatchingParameters:
    metric: str = "CEMD"
    epsilon: float = 0.005
    reciprocal_only: bool = True
    partial_output: bool = True


@dataclass
class featureMatchingOutputs:
    match_indices: Tensor
    deltas: Tensor
    total_dissimilarities: Tensor
    deltas2: Optional[Tensor] = None
    total_dissimilarities2: Optional[Tensor] = None


# ============================================
# Utility Functions
# ============================================

def torch_to_pil(tensor: Tensor) -> Image.Image:
    """Convert torch tensor to PIL Image"""
    if len(tensor.shape) == 3:
        # C, H, W -> H, W, C
        tensor = tensor.permute(1, 2, 0)
    
    # Normalize to [0, 255]
    tensor = tensor.cpu().detach()
    if tensor.max() <= 1.0:
        tensor = tensor * 255
    
    array = tensor.numpy().astype(np.uint8)
    if array.shape[-1] == 1:
        array = array.squeeze(-1)
    
    return Image.fromarray(array)


def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string for embedding in HTML"""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{img_str}"


def get_random_colors(n: int, seed: int = 42) -> np.ndarray:
    """Generate n random but distinct colors"""
    random.seed(seed)
    np.random.seed(seed)
    
    colors = []
    for _ in range(n):
        hue = random.random()
        saturation = 0.6 + random.random() * 0.4
        value = 0.7 + random.random() * 0.3
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([r, g, b, 1.0])
    
    return np.array(colors)


# ============================================
# Graph Layout Computation
# ============================================

class GraphLayoutComputer:
    """Computes graph layouts for visualization"""
    
    def __init__(self, match_indices: Tensor, distance_matrix: Tensor):
        self.match_indices = match_indices
        self.D = distance_matrix
        self.N = self._compute_num_nodes()
        self.G = self._build_graph()
        self.communities, self.community_map = self._detect_communities()
        self.degrees = np.array([self.G.degree(i) for i in range(self.N)])
        
    def _compute_num_nodes(self) -> int:
        """Determine actual number of nodes"""
        if len(self.match_indices) > 0:
            max_query_idx = self.match_indices[:, 0].max().item()
            max_candidate_idx = self.match_indices[:, 1].max().item()
            return max(max_query_idx, max_candidate_idx) + 1
        return self.D.shape[0]
    
    def _build_graph(self) -> nx.Graph:
        """Build NetworkX graph from matches"""
        G = nx.Graph()
        G.add_nodes_from(range(self.N))
        edges = [(int(i.item()), int(j.item())) 
                for i, j in self.match_indices if i != j]
        G.add_edges_from(edges)
        return G
    
    def _detect_communities(self) -> Tuple[List, dict]:
        """Detect communities using greedy modularity"""
        if self.G.number_of_edges() > 0:
            from networkx.algorithms import community
            communities = list(community.greedy_modularity_communities(self.G))
        else:
            communities = [[i] for i in range(self.N)]
        
        community_map = {}
        for comm_idx, comm in enumerate(communities):
            for node in comm:
                community_map[int(node)] = comm_idx
        
        return communities, community_map
    
    def compute_umap_layout(self) -> np.ndarray:
        """Compute UMAP embedding"""
        distance_matrix = self.D.cpu().numpy()
        umap_reducer = UMAP(
            n_components=2, 
            metric='precomputed',
            n_neighbors=min(15, self.N - 1),
            random_state=42
        )
        coords = umap_reducer.fit_transform(np.exp(distance_matrix))
        return coords
    
    def compute_force_directed_layout(self) -> np.ndarray:
        """Compute force-directed (spring) layout"""
        if self.G.number_of_edges() > 0:
            iterations = min(50, max(20, 100 - self.N // 10))
            pos = nx.spring_layout(
                self.G,
                k=2/np.sqrt(self.N),
                iterations=iterations,
                seed=42,
                scale=20
            )
        else:
            pos = nx.random_layout(self.G, seed=42)
            for node in pos:
                pos[node] = pos[node] * 20
        
        coords = np.array([pos[i] for i in range(self.N)])
        return coords


# ============================================
# HOG Visualization Generator
# ============================================

class HOGVisualizer:
    """Generates HOG visualizations"""
    
    def __init__(self, hog_params: HOGParameters):
        self.params = hog_params
        
    def create_hog_visualization(self, hog_output, patch_idx: int, hog_params=None):
        """
        Visualize the HOG features for a given patch index.
        Works even if fullHOGOutput does not have Nh, Nw, or Nbins attributes.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        # --- Infer parameters from HOG output ---
        histograms = hog_output.histograms.cpu()

        # Infer cell & bin structure
        if histograms.ndim == 4:
            # (Npatch, C, Ncells, Nbins)
            Npatch, C, Ncells, Nbins = histograms.shape
        elif histograms.ndim == 3:
            # (Npatch, Ncells, Nbins)
            Npatch, Ncells, Nbins = histograms.shape
            C = 1
        else:
            raise ValueError(f"Unexpected histogram shape: {histograms.shape}")

        # Use provided params or fallback defaults
        if hog_params is None:
            cell_h = 16
            cell_w = 16
        else:
            cell_h = getattr(hog_params, "cell_height", 16)
            cell_w = getattr(hog_params, "cell_width", 16)

        # Try to infer Nh, Nw from total number of cells
        side = int(np.sqrt(Ncells))
        Nh_eff, Nw_eff = side, max(1, Ncells // max(1, side))
        bin_angles = np.linspace(0, np.pi, Nbins, endpoint=False)

        # Select patch histogram
        selected_histograms = histograms[patch_idx]
        if selected_histograms.ndim == 3:
            selected_histograms = selected_histograms.mean(0)  # average over channels if needed

        # --- Prepare visualization image ---
        patch_img = hog_output.patches_image[patch_idx].cpu().numpy()
        if patch_img.ndim == 3:
            patch_img = np.transpose(patch_img, (1, 2, 0))
        patch_img = (patch_img - patch_img.min()) / (patch_img.max() - patch_img.min() + 1e-8)

        fig, ax3 = plt.subplots(figsize=(6, 6))
        ax3.imshow(patch_img, cmap='gray')
        ax3.set_title(f'HOG Visualization (Patch {patch_idx})', fontsize=14, fontweight='bold')

        # --- Reshape histograms safely ---
        h_shape = selected_histograms.shape
        if len(h_shape) == 1:
            Nh_eff, Nw_eff = 1, 1
            selected_histograms = selected_histograms.reshape(1, 1, -1)
        elif len(h_shape) == 2:
            Ncells, Nbins_eff = h_shape
            side = int(np.sqrt(Ncells))
            Nh_eff, Nw_eff = side, max(1, Ncells // max(1, side))
            selected_histograms = selected_histograms.reshape(Nh_eff, Nw_eff, Nbins_eff)
        elif len(h_shape) == 3:
            Nh_eff, Nw_eff, _ = h_shape
        else:
            raise ValueError(f"Unexpected selected histogram shape: {h_shape}")

        max_magnitude = selected_histograms.max() if selected_histograms.max() > 0 else 1.0

        # --- Draw arrows per cell ---
        for cell_y in range(Nh_eff):
            for cell_x in range(Nw_eff):
                center_x = (cell_x + 0.5) * cell_w
                center_y = (cell_y + 0.5) * cell_h
                cell_hist = selected_histograms[cell_y, cell_x]

                for bin_idx, magnitude in enumerate(cell_hist):
                    if magnitude > 0.05 * max_magnitude:
                        angle = bin_angles[bin_idx]
                        length = magnitude * min(cell_w, cell_h) * 0.45 / max_magnitude
                        dx = length * np.cos(angle)
                        dy = length * np.sin(angle)
                        color = plt.cm.hsv(angle / (2 * np.pi))

                        ax3.arrow(center_x, center_y, dx, dy,
                                head_width=2.0, head_length=2.0,
                                fc=color, ec=color, alpha=0.85, linewidth=2.0)
                        ax3.arrow(center_x, center_y, -dx, -dy,
                                head_width=2.0, head_length=2.0,
                                fc=color, ec=color, alpha=0.85, linewidth=2.0)

        # --- Draw grid ---
        for i in range(Nh_eff + 1):
            ax3.axhline(y=i * cell_h - 0.5, color='cyan', linewidth=1.0, alpha=0.6)
        for i in range(Nw_eff + 1):
            ax3.axvline(x=i * cell_w - 0.5, color='cyan', linewidth=1.0, alpha=0.6)

        ax3.set_xlim(-0.5, Nw_eff * cell_w - 0.5)
        ax3.set_ylim(Nh_eff * cell_h - 0.5, -0.5)
        ax3.axis('off')
        plt.tight_layout()

        return fig
            
    def create_histogram_chart(self, hog_output, patch_idx: int, hog_params=None):
        """
        Create histogram bar chart for the center cell of a patch.
        Automatically infers grid shape (Nh×Nw) even if histograms are flat or channel-first.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        # --- Extract histograms ---
        histograms = hog_output.histograms.cpu()

        if histograms.ndim == 4:
            # (N, C, Ncells, Nbins)
            N, C, Ncells, Nbins = histograms.shape
        elif histograms.ndim == 3:
            # (N, Ncells, Nbins)
            N, Ncells, Nbins = histograms.shape
            C = 1
        else:
            raise ValueError(f"Unexpected histogram shape: {histograms.shape}")

        # --- Select patch ---
        selected_hist = histograms[patch_idx]
        if selected_hist.ndim == 3:
            selected_hist = selected_hist.mean(0)  # average channels if needed

        # --- Infer Nh, Nw from number of cells ---
        side = int(np.sqrt(selected_hist.shape[0]))
        Nh_eff = side
        Nw_eff = max(1, selected_hist.shape[0] // max(1, side))

        # Reshape safely to (Nh, Nw, Nbins)
        selected_hist = selected_hist.reshape(Nh_eff, Nw_eff, Nbins)

        # --- Pick the center cell ---
        example_cell_y, example_cell_x = Nh_eff // 2, Nw_eff // 2
        example_hist = selected_hist[example_cell_y, example_cell_x]

        # --- Plot histogram ---
        fig, ax = plt.subplots(figsize=(8, 4))
        bin_angles_deg = np.linspace(0, 180, Nbins, endpoint=False)
        colors = [plt.cm.hsv(a / 360) for a in bin_angles_deg]

        ax.bar(range(Nbins), example_hist, color=colors,
            edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Orientation Bin (°)', fontsize=12)
        ax.set_ylabel('Magnitude', fontsize=12)
        ax.set_title(f'HOG Histogram - Center Cell (Patch {patch_idx})',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(range(Nbins))
        ax.set_xticklabels([f'{int(a)}°' for a in bin_angles_deg],
                        rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, example_hist.max() * 1.1 if example_hist.max() > 0 else 1)

        plt.tight_layout()

        import io, base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        return f"data:image/png;base64,{img_str}"

# ============================================
# Nearest Neighbor Analyzer
# ============================================

class NearestNeighborAnalyzer:
    """Analyzes nearest neighbors and statistical tests"""
    
    def __init__(
        self,
        distance_matrix: Tensor,
        match_indices: Tensor,
        deltas: Tensor,
        epsilon: float
    ):
        self.D = distance_matrix
        self.match_indices = match_indices
        self.deltas = deltas
        self.epsilon = epsilon
    
    def get_nearest_neighbors(
        self,
        query_idx: int,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get k nearest neighbors and their acceptance status"""
        
        # Get matches for this query
        query_matches = self.match_indices[
            self.match_indices[:, 0] == query_idx
        ][:, 1]
        
        # Get distances
        D_i = self.D[query_idx].detach().cpu().numpy()
        
        # Sort by distance
        sorted_indices = np.argsort(D_i)[:k]
        sorted_distances = D_i[sorted_indices]
        
        # Determine acceptance
        accepted_mask = np.isin(sorted_indices, query_matches.cpu().numpy())
        
        return sorted_indices, sorted_distances, accepted_mask
    
    def create_statistical_test_plot(
        self,
        query_idx: int,
        k: int = 10
    ) -> str:
        """Create statistical test visualization"""
        
        indices, distances, accepted = self.get_nearest_neighbors(query_idx, k)
        thresh_value = self.deltas[query_idx].item()
        n_matches = (self.match_indices[:, 0] == query_idx).sum().item()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        colors = ['green' if acc else 'red' for acc in accepted]
        ax.scatter(range(1, len(indices) + 1), distances, c=colors, s=100, zorder=3)
        
        ax.axhline(y=thresh_value, color='orange', linestyle='--', linewidth=2,
                  label=f'Threshold δ(ε) = {thresh_value:.3f}')
        
        ax.axhspan(thresh_value, distances.max() * 1.1, alpha=0.2,
                  color='red', label='Rejected region')
        
        ax.set_xlabel('Nearest Neighbor Rank', fontsize=11)
        ax.set_ylabel('Distance', fontsize=11)
        ax.set_title(
            f'Statistical Test: Query {query_idx} '
            f'(ε={self.epsilon}, {n_matches} matches)',
            fontsize=12, fontweight='bold'
        )
        ax.set_xticks(range(1, len(indices) + 1))
        ax.grid(alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        
        return f"data:image/png;base64,{img_str}"


# ============================================
# Main Application Data Container
# ============================================

class AppData:
    """Container for all application data"""
    
    def __init__(
        self,
        hog_output: fullHOGOutput,
        matching_output: featureMatchingOutputs,
        hog_params: HOGParameters,
        extracted_patches: List[Tensor]
    ):
        self.hog_output = hog_output
        self.matching_output = matching_output
        self.hog_params = hog_params
        self.extracted_patches = extracted_patches
        
        # Compute layouts
        self.graph_computer = GraphLayoutComputer(
            matching_output.match_indices,
            matching_output.total_dissimilarities
        )
        
        self.umap_coords = None
        self.force_coords = None
        
        # Initialize visualizers
        self.hog_visualizer = HOGVisualizer(hog_params)
        self.nn_analyzer = NearestNeighborAnalyzer(
            matching_output.total_dissimilarities,
            matching_output.match_indices,
            matching_output.deltas,
            0.005  # epsilon - you can make this configurable
        )
        
        # Compute community colors
        self.community_colors = get_random_colors(
            len(self.graph_computer.communities)
        )


# ============================================
# Dash Application
# ============================================

def create_app(app_data: AppData) -> dash.Dash:
    """Create and configure the Dash application"""
    
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True
    )
    
    # App layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Interactive HOG & Patch Matching Visualization",
                       className="text-center mb-4")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Label("Graph Layout Method:", className="fw-bold"),
                dcc.Dropdown(
                    id='layout-dropdown',
                    options=[
                        {'label': 'UMAP Embedding', 'value': 'umap'},
                        {'label': 'Force-Directed', 'value': 'force'}
                    ],
                    value='umap',
                    clearable=False
                )
            ], width=3),
            
            dbc.Col([
                html.Label("Number of Neighbors:", className="fw-bold"),
                dcc.Slider(
                    id='k-slider',
                    min=5,
                    max=20,
                    step=1,
                    value=10,
                    marks={i: str(i) for i in range(5, 21, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], width=3),
            
            dbc.Col([
                html.Div([
                    html.P("Click on any patch in the graph to view its details and nearest neighbors",
                          className="text-muted mb-0")
                ])
            ], width=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading-graph",
                    type="default",
                    children=[
                        dcc.Graph(
                            id='graph-plot',
                            style={'height': '700px'},
                            config={'displayModeBar': True}
                        )
                    ]
                )
            ], width=12)
        ], className="mb-4"),
        
        html.Hr(),
        
        # Selected patch details
        html.Div(id='selected-patch-info'),
        
        dbc.Row([
            dbc.Col([
                html.Div(id='patch-details')
            ], width=12)
        ])
        
    ], fluid=True, style={'padding': '20px'})
    
    # Callbacks
    @app.callback(
        Output('graph-plot', 'figure'),
        Input('layout-dropdown', 'value')
    )
    def update_graph(layout_method):
        """Update graph visualization based on layout method"""
        
        # Compute coordinates
        if layout_method == 'umap':
            if app_data.umap_coords is None:
                app_data.umap_coords = app_data.graph_computer.compute_umap_layout()
            coords = app_data.umap_coords
            title_suffix = "UMAP Embedding"
        else:
            if app_data.force_coords is None:
                app_data.force_coords = app_data.graph_computer.compute_force_directed_layout()
            coords = app_data.force_coords
            title_suffix = "Force-Directed Layout"
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        G = app_data.graph_computer.G
        community_map = app_data.graph_computer.community_map
        
        # Intra-community edges
        for comm_idx in range(len(app_data.graph_computer.communities)):
            edge_x = []
            edge_y = []
            
            for i, j in G.edges():
                if (community_map.get(int(i), -1) == comm_idx and
                    community_map.get(int(j), -1) == comm_idx):
                    edge_x.extend([coords[i, 0], coords[j, 0], None])
                    edge_y.extend([coords[i, 1], coords[j, 1], None])
            
            if edge_x:
                color = app_data.community_colors[comm_idx]
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(width=1.5, color=f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},0.4)'),
                    hoverinfo='none',
                    showlegend=False
                ))
        
        # Inter-community edges
        inter_edge_x = []
        inter_edge_y = []
        for i, j in G.edges():
            comm_i = community_map.get(int(i), -1)
            comm_j = community_map.get(int(j), -1)
            if comm_i != comm_j:
                inter_edge_x.extend([coords[i, 0], coords[j, 0], None])
                inter_edge_y.extend([coords[i, 1], coords[j, 1], None])
        
        if inter_edge_x:
            fig.add_trace(go.Scatter(
                x=inter_edge_x, y=inter_edge_y,
                mode='lines',
                line=dict(width=0.5, color='rgba(200,200,200,0.2)'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes colored by community
        node_colors = []
        node_sizes = []
        hover_text = []
        
        degrees = app_data.graph_computer.degrees
        
        for i in range(app_data.graph_computer.N):
            comm_idx = community_map.get(i, 0)
            color = app_data.community_colors[comm_idx]
            node_colors.append(f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})')
            
            # Size by degree
            size = 10 + min(degrees[i] * 2, 30)
            node_sizes.append(size)
            
            hover_text.append(
                f'Patch {i}<br>'
                f'Degree: {degrees[i]}<br>'
                f'Community: {comm_idx}<br>'
                f'Click to view details'
            )
        
        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='white')
            ),
            text=hover_text,
            hoverinfo='text',
            customdata=np.arange(app_data.graph_computer.N),
            showlegend=False
        ))
        
        fig.update_layout(
            title=dict(
                text=f'Patch Matching Graph - {title_suffix}<br>'
                     f'<sub>{app_data.graph_computer.N} patches, '
                     f'{G.number_of_edges()} connections, '
                     f'{len(app_data.graph_computer.communities)} communities</sub>',
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=700
        )
        
        return fig
    
    @app.callback(
        [Output('selected-patch-info', 'children'),
         Output('patch-details', 'children')],
        [Input('graph-plot', 'clickData'),
         Input('k-slider', 'value')],
        prevent_initial_call=True
    )
    def display_patch_details(clickData, k):
        """Display details when a patch is clicked"""
        
        if clickData is None:
            return "", ""
        
        # Get clicked patch index
        patch_idx = clickData['points'][0]['customdata']
        
        # Get nearest neighbors
        nn_indices, nn_distances, nn_accepted = app_data.nn_analyzer.get_nearest_neighbors(
            patch_idx, k
        )
        
        # Create info header
        info_header = dbc.Alert([
            html.H4(f"Selected Patch: {patch_idx}", className="alert-heading"),
            html.P(f"Showing {k} nearest neighbors with HOG visualizations")
        ], color="info")
        
        # Create patch detail panels
        detail_panels = []
        
        # Query patch panel
        query_patch = app_data.extracted_patches[patch_idx]
        query_img = torch_to_pil(query_patch).resize((256, 256))
        query_img_b64 = pil_to_base64(query_img)
        
        query_hog = app_data.hog_visualizer.create_hog_visualization(
            app_data.hog_output, patch_idx
        )
        query_hist = app_data.hog_visualizer.create_histogram_chart(
            app_data.hog_output, patch_idx
        )
        
        query_panel = dbc.Card([
            dbc.CardHeader(html.H5("Query Patch", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Img(src=query_img_b64, style={'width': '100%'})
                    ], width=3),
                    dbc.Col([
                        html.Img(src=query_hog, style={'width': '100%'})
                    ], width=9)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Img(src=query_hist, style={'width': '100%', 'marginTop': '10px'})
                    ], width=12)
                ])
            ])
        ], className="mb-4")
        
        detail_panels.append(query_panel)
        
        # Statistical test plot
        stat_plot = app_data.nn_analyzer.create_statistical_test_plot(patch_idx, k)
        stat_panel = dbc.Card([
            dbc.CardHeader(html.H5("Statistical Test Results", className="mb-0")),
            dbc.CardBody([
                html.Img(src=stat_plot, style={'width': '100%'})
            ])
        ], className="mb-4")
        
        detail_panels.append(stat_panel)
        
        # Nearest neighbors
        nn_header = html.H5(f"Top {k} Nearest Neighbors", className="mb-3")
        detail_panels.append(nn_header)
        
        nn_cards = []
        for idx, (nn_idx, distance, accepted) in enumerate(
            zip(nn_indices, nn_distances, nn_accepted), 1
        ):
            nn_patch = app_data.extracted_patches[nn_idx]
            nn_img = torch_to_pil(nn_patch).resize((256, 256))
            nn_img_b64 = pil_to_base64(nn_img)
            
            nn_hog = app_data.hog_visualizer.create_hog_visualization(
                app_data.hog_output, nn_idx
            )
            
            status = "✓ ACCEPTED" if accepted else "❌ REJECTED"
            color = "success" if accepted else "danger"
            
            nn_card = dbc.Card([
                dbc.CardHeader([
                    html.H6(f"Neighbor #{idx} - Patch {nn_idx}", className="mb-0"),
                    dbc.Badge(status, color=color, className="float-end")
                ]),
                dbc.CardBody([
                    html.P(f"Distance: {distance:.4f}", className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Img(src=nn_img_b64, style={'width': '100%'})
                        ], width=3),
                        dbc.Col([
                            html.Img(src=nn_hog, style={'width': '100%'})
                        ], width=9)
                    ])
                ])
            ], className="mb-3", color=color, outline=True)
            
            nn_cards.append(nn_card)
        
        detail_panels.extend(nn_cards)
        
        return info_header, detail_panels
    
    return app


# ============================================
# Entry Point
# ============================================

def run_visualization_app(
    hog_output: fullHOGOutput,
    matching_output: featureMatchingOutputs,
    hog_params: HOGParameters,
    extracted_patches: List[Tensor],
    port: int = 8050,
    debug: bool = True
):
    """
    Run the interactive visualization application
    
    Args:
        hog_output: Full HOG computation output
        matching_output: Feature matching results
        hog_params: HOG parameters used
        extracted_patches: List of image patches as tensors
        port: Port to run the server on
        debug: Whether to run in debug mode
    """
    
    print("Initializing application data...")
    app_data = AppData(hog_output, matching_output, hog_params, extracted_patches)
    
    print("Creating application...")
    app = create_app(app_data)
    
    print(f"\nStarting server on http://localhost:{port}")
    print("Open this URL in your web browser to view the visualization")
    print("\nInstructions:")
    print("  1. Choose a graph layout method (UMAP or Force-Directed)")
    print("  2. Click on any patch in the graph")
    print("  3. View the patch details, HOG visualization, and nearest neighbors")
    print("  4. Adjust the number of neighbors with the slider")
    print("\nPress Ctrl+C to stop the server")
    
    app.run_server(debug=debug, port=port)


if __name__ == "__main__":
    print("This module should be imported and used with your data.")
    print("Example usage:")
    print("""
    from hog_visualization_app import run_visualization_app
    
    run_visualization_app(
        hog_output=your_hog_output,
        matching_output=your_matching_output,
        hog_params=your_hog_params,
        extracted_patches=your_extracted_patches
    )
    """)