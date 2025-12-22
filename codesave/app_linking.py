"""
Interactive HOG and Patch Matching Visualization with Parameter Configuration
Allows users to configure parameters, run the pipeline, and visualize results
"""
import matplotlib
matplotlib.use("Agg")  # must be before importing pyplot
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import Tensor
import plotly.graph_objects as go
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
from einops import rearrange

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
from matplotlib.colors import hsv_to_rgb
from src.data.patch_database import PatchDatabase  # 🔹 NEW

# Import your existing classes
from src.character_linking.feature_matching import (
    HOG, HOGParameters, 
    featureMatching, featureMatchingParameters
)

# Import utility functions from the original app
from hog_visualization_app import (
    torch_to_pil, pil_to_base64, get_random_colors,
    GraphLayoutComputer, HOGVisualizer, NearestNeighborAnalyzer
)


# ============================================
# Enhanced App Data Container
# ============================================

class EnhancedAppData:
    """Container for application data with pipeline execution"""
    
    def __init__(self):
        self.database = None  # 🔹 NEW: store PatchDatabase
        self.img_torch = None
        self.img_comp_torch = None  # 🔹 NEW: connected component mask
        self.hog_output = None
        self.matching_output = None
        self.hog_params = None
        self.matching_params = None
        self.extracted_patches = None
        self.histograms = None
        
        # Computed on demand
        self.graph_computer = None
        self.umap_coords = None
        self.force_coords = None
        self.hog_visualizer = None
        self.nn_analyzer = None
        self.community_colors = None
    
    def run_pipeline(
        self,
        img_torch: Tensor,
        img_comp_torch: Optional[Tensor],  # 🔹 NEW
        hog_params: HOGParameters,
        matching_params: featureMatchingParameters
    ):
        """Run the complete HOG and matching pipeline"""
        print("Running HOG computation...")
        self.hog_params = hog_params
        self.matching_params = matching_params
        self.img_torch = img_torch
        self.img_comp_torch = img_comp_torch  # 🔹 NEW
        
        hog = HOG(hog_params)
        self.hog_output = hog(img_torch, img_comp_torch)  # 🔸 CHANGED (previously None)
        
        print(f"HOG output: {len(self.hog_output.patches_image)} patches")
        
        # Reshape histograms: Npatch C Ncells Nbins -> Npatch (C Ncells) Nbins
        self.histograms = rearrange(
            self.hog_output.histograms,
            'Npatch C Ncells Nbins -> Npatch (C Ncells) Nbins'
        )
        
        print("Running feature matching...")
        # Create feature matcher and run
        feature_matcher = featureMatching(matching_params)
        self.matching_output = feature_matcher(
            query_histograms=self.histograms,
            key_histograms=self.histograms
        )
        
        print(f"Matches: {len(self.matching_output.match_indices)}")
        
        # Extract patches
        self.extracted_patches = [
            self.hog_output.patches_image[i] 
            for i in range(len(self.hog_output.patches_image))
        ]
        
        # Initialize computed objects
        self._initialize_visualizers()
        
        return True
    
    def _initialize_visualizers(self):
        """Initialize visualization components"""
        
        self.graph_computer = GraphLayoutComputer(
            self.matching_output.match_indices,
            self.matching_output.total_dissimilarities
        )
        
        self.hog_visualizer = HOGVisualizer(self.hog_params)
        
        self.nn_analyzer = NearestNeighborAnalyzer(
            self.matching_output.total_dissimilarities,
            self.matching_output.match_indices,
            self.matching_output.deltas,
            self.matching_params.epsilon
        )
        
        self.community_colors = get_random_colors(
            len(self.graph_computer.communities)
        )
    
    def is_ready(self) -> bool:
        """Check if pipeline has been run"""
        return self.hog_output is not None and self.matching_output is not None


# ============================================
# Create Enhanced Dash Application
# ============================================

def create_enhanced_app() -> dash.Dash:
    """Create the enhanced Dash application with parameter configuration"""
    
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True
    )
    
    # Global app data
    app_data = EnhancedAppData()
    
    # Store in app for access in callbacks
    app.app_data = app_data
    
    # App layout
    app.layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Interactive HOG & Patch Matching Pipeline",
                       className="text-center mb-2"),
                html.P("Configure parameters, run analysis, and explore results interactively",
                      className="text-center text-muted mb-4")
            ])
        ]),
        
        # Main tabs
        dcc.Tabs(id='main-tabs', value='config-tab', children=[
            
            # Configuration Tab
            dcc.Tab(label='⚙️ Configuration & Run', value='config-tab', children=[
                dbc.Container([
                    html.Br(),
                    
                    # Image Upload Section
                    dbc.Card([
                        dbc.CardHeader(html.H4("1. Load Patch Database", className="mb-0")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Database Path:", className="fw-bold"),
                                    dbc.Input(
                                        id='database-path',
                                        type='text',
                                        value='data/datasets/databases/book1',  # default
                                        placeholder='Enter path to PatchDatabase directory'
                                    )
                                ], width=8),
                                dbc.Col([
                                    dbc.Button("📂 Load Database", id='load-database-btn', color="secondary", className="w-100")
                                ], width=4)
                            ]),
                            html.Div(id='database-status', className='mt-3'),
                            html.Div(id='image-selectors', className='mt-3')
                        ])
                    ], className="mb-4"),
                    
                    # HOG Parameters Section
                    dbc.Card([
                        dbc.CardHeader(html.H4("2. HOG Parameters", className="mb-0")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Device:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='hog-device',
                                        options=[
                                            {'label': 'CUDA (GPU)', 'value': 'cuda'},
                                            {'label': 'CPU', 'value': 'cpu'}
                                        ],
                                        value='cuda' if torch.cuda.is_available() else 'cpu'
                                    )
                                ], width=3),
                                
                                dbc.Col([
                                    html.Label("Method:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='hog-method',
                                        options=[
                                            {'label': 'Gaussian', 'value': 'gaussian'},
                                            {'label': 'Farid 3x3', 'value': 'farid_3x3'},
                                            {'label': 'Farid 5x5', 'value': 'farid_5x5'},
                                            {'label': 'Central Differences', 'value': 'central'}
                                        ],
                                        value='gaussian'
                                    )
                                ], width=3),
                                
                                dbc.Col([
                                    html.Label("Gradient Sigma:", className="fw-bold"),
                                    dbc.Input(
                                        id='hog-grdt-sigma',
                                        type='number',
                                        value=2.5,
                                        min=0.1,
                                        max=10,
                                        step=0.1
                                    )
                                ], width=3),
                                
                                dbc.Col([
                                    html.Label("Kernel Size Factor:", className="fw-bold"),
                                    dbc.Input(
                                        id='hog-ksize-factor',
                                        type='number',
                                        value=6,
                                        min=1,
                                        max=20,
                                        step=1
                                    )
                                ], width=3)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Cell Height:", className="fw-bold"),
                                    dbc.Input(
                                        id='hog-cell-height',
                                        type='number',
                                        value=16,
                                        min=4,
                                        max=64,
                                        step=2
                                    )
                                ], width=3),
                                
                                dbc.Col([
                                    html.Label("Cell Width:", className="fw-bold"),
                                    dbc.Input(
                                        id='hog-cell-width',
                                        type='number',
                                        value=16,
                                        min=4,
                                        max=64,
                                        step=2
                                    )
                                ], width=3),
                                
                                dbc.Col([
                                    html.Label("Patch Size:", className="fw-bold"),
                                    dbc.Input(
                                        id='hog-psize',
                                        type='number',
                                        value=112,
                                        min=32,
                                        max=256,
                                        step=16
                                    )
                                ], width=3),
                                
                                dbc.Col([
                                    html.Label("Number of Bins:", className="fw-bold"),
                                    dbc.Input(
                                        id='hog-num-bins',
                                        type='number',
                                        value=16,
                                        min=4,
                                        max=32,
                                        step=1
                                    )
                                ], width=3)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Sigma (Weighting):", className="fw-bold"),
                                    dbc.Input(
                                        id='hog-sigma',
                                        type='number',
                                        value=100,
                                        min=0.1,
                                        max=200,
                                        step=0.1
                                    )
                                ], width=4),
                                
                                dbc.Col([
                                    html.Label("Threshold:", className="fw-bold"),
                                    dbc.Input(
                                        id='hog-threshold',
                                        type='number',
                                        value=0.2,
                                        min=0.0,
                                        max=1.0,
                                        step=0.05
                                    )
                                ], width=4),
                                
                                dbc.Col([
                                    html.Label("Partial Output:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='hog-partial-output',
                                        options=[
                                            {'label': 'True', 'value': True},
                                            {'label': 'False', 'value': False}
                                        ],
                                        value=False
                                    )
                                ], width=4)
                            ])
                        ])
                    ], className="mb-4"),
                    
                    # Feature Matching Parameters Section
                    dbc.Card([
                        dbc.CardHeader(html.H4("3. Feature Matching Parameters", className="mb-0")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Metric:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='matching-metric',
                                        options=[
                                            {'label': 'CEMD (Cross-bin EMD)', 'value': 'CEMD'},
                                            {'label': 'L1', 'value': 'L1'},
                                            {'label': 'L2', 'value': 'L2'},
                                            {'label': 'EMD', 'value': 'EMD'}
                                        ],
                                        value='CEMD'
                                    )
                                ], width=4),
                                
                                dbc.Col([
                                    html.Label("Epsilon (ε):", className="fw-bold"),
                                    dbc.Input(
                                        id='matching-epsilon',
                                        type='number',
                                        value=0.005,
                                        min=0.0001,
                                        max=0.1,
                                        step=0.0001
                                    )
                                ], width=4),
                                
                                dbc.Col([
                                    html.Label("Reciprocal Only:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='matching-reciprocal',
                                        options=[
                                            {'label': 'True', 'value': True},
                                            {'label': 'False', 'value': False}
                                        ],
                                        value=True
                                    )
                                ], width=4)
                            ])
                        ])
                    ], className="mb-4"),
                    
                    # Run Button
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "🚀 Run Pipeline",
                                id='run-pipeline-btn',
                                color="primary",
                                size="lg",
                                className="w-100",
                                disabled=True
                            )
                        ], width=12)
                    ], className="mb-3"),
                    
                    # Progress and Results
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='pipeline-status')
                        ], width=12)
                    ])
                    
                ], fluid=True)
            ]),
            
            # Visualization Tab
            dcc.Tab(label='📊 Visualization', value='viz-tab', children=[
                dbc.Container([
                    html.Br(),
                    
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
                                html.P("Click on any patch to view details and neighbors",
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
                    
                    html.Div(id='selected-patch-info'),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='patch-details')
                        ], width=12)
                    ])
                    
                ], fluid=True)
            ])
        ])
        
    ], fluid=True, style={'padding': '20px'})
    
    # ============================================
    # Callbacks
    # ============================================
    

    @app.callback(
        [Output('database-status', 'children'),
        Output('image-selectors', 'children'),
        Output('run-pipeline-btn', 'disabled')],
        Input('load-database-btn', 'n_clicks'),
        State('database-path', 'value'),
        prevent_initial_call=True
    )
    def load_database(n_clicks, db_path):
        """Load a PatchDatabase from disk"""
        try:
            db = PatchDatabase.load(db_path)
            app.app_data.database = db  # store it globally

            n_imgs = len(db._images)
            app.app_data.database = db
            
            status = dbc.Alert([
                html.I(className="bi bi-check-circle-fill me-2"),
                f"✓ Loaded database from {db_path} ({n_imgs} images)"
            ], color="success")

            selector = html.Div([
                html.Label("Select Image Index:", className="fw-bold mt-2"),
                dcc.Slider(
                    id='db-image-index',
                    min=0,
                    max=n_imgs - 1,
                    step=1,
                    value=0,
                    marks={i: str(i) for i in range(n_imgs)}
                )
            ])

            return status, selector, False

        except Exception as e:
            error = dbc.Alert(f"❌ Failed to load database: {e}", color="danger")
            return error, "", True


    @app.callback(
        Output('pipeline-status', 'children'),
        Input('run-pipeline-btn', 'n_clicks'),
        [State('db-image-index', 'value'),  # 🔹 add this
         State('hog-device', 'value'),
         State('hog-method', 'value'),
         State('hog-grdt-sigma', 'value'),
         State('hog-ksize-factor', 'value'),
         State('hog-cell-height', 'value'),
         State('hog-cell-width', 'value'),
         State('hog-psize', 'value'),
         State('hog-num-bins', 'value'),
         State('hog-sigma', 'value'),
         State('hog-threshold', 'value'),
         State('hog-partial-output', 'value'),
         State('matching-metric', 'value'),
         State('matching-epsilon', 'value'),
         State('matching-reciprocal', 'value')],
        prevent_initial_call=True
    )
    def run_pipeline(n_clicks, img_idx, device, method, grdt_sigma, ksize_factor,
                     cell_height, cell_width, psize, num_bins, sigma, threshold,
                     partial_output, metric, epsilon, reciprocal):
        """Run the HOG and feature matching pipeline"""
        
        if n_clicks is None:
            return ""
        
        try:
            # Get uploaded image
            db = app.app_data.database
            img_np = db._images[img_idx]
            img_comp = db._components[img_idx]

            img_idx = 0
            ctx = callback_context
            if ctx.triggered and 'db-image-index' in ctx.states:
                img_idx = ctx.states['db-image-index.value']
            img_np = db._images[img_idx]
            img_comp = db._components[img_idx]

            # Convert to grayscale if needed
            import cv2
            if img_np.ndim == 3 and img_np.shape[-1] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)[..., None]

            # Convert to torch
            device = device or 'cpu'
            img_torch = torch.tensor(img_np, device=device).permute(2, 0, 1).float() / 255
            img_comp_torch = img_comp
            
            # Convert to torch
            if len(img_np.shape) == 2:
                img_np = img_np[:, :, np.newaxis]
                        
            # Create HOG parameters
            hog_params = HOGParameters(
                device=device,
                C=img_np.shape[-1],
                partial_output=partial_output,
                method=method,
                grdt_sigma=float(grdt_sigma),
                ksize_factor=int(ksize_factor),
                cell_height=int(cell_height),
                cell_width=int(cell_width),
                psize=int(psize),
                num_bins=int(num_bins),
                sigma=float(sigma),
                threshold=float(threshold)
            )
            
            # Create matching parameters
            matching_params = featureMatchingParameters(
                metric=metric,
                epsilon=float(epsilon),
                reciprocal_only=reciprocal,
                partial_output=False
            )
            
            # Run pipeline
            success = app.app_data.run_pipeline(img_torch, img_comp_torch, hog_params, matching_params)

            
            if success:
                n_patches = len(app.app_data.extracted_patches)
                n_matches = len(app.app_data.matching_output.match_indices)
                
                result = dbc.Alert([
                    html.H5("✓ Pipeline Complete!", className="alert-heading"),
                    html.Hr(),
                    html.P([
                        html.Strong("Patches extracted: "), f"{n_patches}", html.Br(),
                        html.Strong("Matches found: "), f"{n_matches}", html.Br(),
                        html.Strong("Match density: "), f"{n_matches / (n_patches * n_patches) * 100:.2f}%"
                    ]),
                    html.Hr(),
                    html.P("Switch to the 'Visualization' tab to explore results!", className="mb-0")
                ], color="success")
                
                return result
            
        except Exception as e:
            import traceback
            error = dbc.Alert([
                html.H5("❌ Pipeline Failed", className="alert-heading"),
                html.Hr(),
                html.P(f"Error: {str(e)}"),
                html.Details([
                    html.Summary("Show traceback"),
                    html.Pre(traceback.format_exc(), style={'fontSize': '10px'})
                ])
            ], color="danger")
            return error
    
    @app.callback(
        Output('graph-plot', 'figure'),
        Input('layout-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_graph(layout_method):
        """Update graph visualization"""
        
        if not app.app_data.is_ready():
            return go.Figure().add_annotation(
                text="Run the pipeline first!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
        
        # Compute coordinates
        if layout_method == 'umap':
            if app.app_data.umap_coords is None:
                app.app_data.umap_coords = app.app_data.graph_computer.compute_umap_layout()
            coords = app.app_data.umap_coords
            title_suffix = "UMAP Embedding"
        else:
            if app.app_data.force_coords is None:
                app.app_data.force_coords = app.app_data.graph_computer.compute_force_directed_layout()
            coords = app.app_data.force_coords
            title_suffix = "Force-Directed Layout"
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        G = app.app_data.graph_computer.G
        community_map = app.app_data.graph_computer.community_map
        
        # Intra-community edges
        for comm_idx in range(len(app.app_data.graph_computer.communities)):
            edge_x, edge_y = [], []
            
            for i, j in G.edges():
                if (community_map.get(int(i), -1) == comm_idx and
                    community_map.get(int(j), -1) == comm_idx):
                    edge_x.extend([coords[i, 0], coords[j, 0], None])
                    edge_y.extend([coords[i, 1], coords[j, 1], None])
            
            if edge_x:
                color = app.app_data.community_colors[comm_idx]
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(width=1.5, color=f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},0.4)'),
                    hoverinfo='none',
                    showlegend=False
                ))
        
        # Inter-community edges
        inter_edge_x, inter_edge_y = [], []
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
        
        # Add nodes
        node_colors, node_sizes, hover_text = [], [], []
        degrees = app.app_data.graph_computer.degrees
        
        for i in range(app.app_data.graph_computer.N):
            comm_idx = community_map.get(i, 0)
            color = app.app_data.community_colors[comm_idx]
            node_colors.append(f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})')
            
            size = 10 + min(degrees[i] * 2, 30)
            node_sizes.append(size)
            
            hover_text.append(
                f'Patch {i}<br>Degree: {degrees[i]}<br>Community: {comm_idx}<br>Click to view'
            )
        
        fig.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            mode='markers',
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color='white')),
            text=hover_text,
            hoverinfo='text',
            customdata=np.arange(app.app_data.graph_computer.N),
            showlegend=False
        ))
        
        fig.update_layout(
            title=dict(
                text=f'Patch Matching Graph - {title_suffix}<br>'
                     f'<sub>{app.app_data.graph_computer.N} patches, '
                     f'{G.number_of_edges()} connections, '
                     f'{len(app.app_data.graph_computer.communities)} communities</sub>',
                x=0.5, xanchor='center'
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
        """Display patch details when clicked"""
        
        if clickData is None or not app.app_data.is_ready():
            return "", ""
        
        patch_idx = clickData['points'][0]['customdata']
        
        # Get nearest neighbors
        nn_indices, nn_distances, nn_accepted = app.app_data.nn_analyzer.get_nearest_neighbors(
            patch_idx, k
        )
        
        # Info header
        info_header = dbc.Alert([
            html.H4(f"Selected Patch: {patch_idx}", className="alert-heading"),
            html.P(f"Showing {k} nearest neighbors with HOG visualizations")
        ], color="info")
        
        # Query patch panel
        query_patch = app.app_data.extracted_patches[patch_idx]
        query_img = torch_to_pil(query_patch).resize((256, 256))
        query_img_b64 = pil_to_base64(query_img)
        
        query_hog = app.app_data.hog_visualizer.create_hog_visualization(
            app.app_data.hog_output, patch_idx
        )
        query_hist = app.app_data.hog_visualizer.create_histogram_chart(
            app.app_data.hog_output, patch_idx
        )
        
        query_panel = dbc.Card([
            dbc.CardHeader(html.H5("Query Patch", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([html.Img(src=query_img_b64, style={'width': '100%'})], width=3),
                    dbc.Col([html.Img(src=query_hog, style={'width': '100%'})], width=9)
                ]),
                dbc.Row([
                    dbc.Col([html.Img(src=query_hist, style={'width': '100%', 'marginTop': '10px'})], width=12)
                ])
            ])
        ], className="mb-4")
        
        detail_panels = [query_panel]
        
        # Statistical test plot
        stat_plot = app.app_data.nn_analyzer.create_statistical_test_plot(patch_idx, k)
        stat_panel = dbc.Card([
            dbc.CardHeader(html.H5("Statistical Test Results", className="mb-0")),
            dbc.CardBody([html.Img(src=stat_plot, style={'width': '100%'})])
        ], className="mb-4")
        
        detail_panels.append(stat_panel)
        detail_panels.append(html.H5(f"Top {k} Nearest Neighbors", className="mb-3"))
        
        # Nearest neighbors
        for idx, (nn_idx, distance, accepted) in enumerate(
            zip(nn_indices, nn_distances, nn_accepted), 1
        ):
            nn_patch = app.app_data.extracted_patches[nn_idx]
            nn_img = torch_to_pil(nn_patch).resize((256, 256))
            nn_img_b64 = pil_to_base64(nn_img)
            
            nn_hog = app.app_data.hog_visualizer.create_hog_visualization(
                app.app_data.hog_output, nn_idx
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
                        dbc.Col([html.Img(src=nn_img_b64, style={'width': '100%'})], width=3),
                        dbc.Col([html.Img(src=nn_hog, style={'width': '100%'})], width=9)
                    ])
                ])
            ], className="mb-3", color=color, outline=True)
            
            detail_panels.append(nn_card)
        
        return info_header, detail_panels
    
    return app


# ============================================
# Entry Point
# ============================================

def run_enhanced_app(port: int = 8050, debug: bool = True):
    """
    Run the enhanced interactive application with parameter configuration
    
    Args:
        port: Port to run the server on
        debug: Whether to run in debug mode
    """
    
    print("\n" + "="*60)
    print("Enhanced HOG Visualization Application")
    print("="*60)
    print(f"\nStarting server on http://localhost:{port}")
    print("\nFeatures:")
    print("  ✓ Configure HOG parameters")
    print("  ✓ Configure feature matching parameters")
    print("  ✓ Upload and process images")
    print("  ✓ Interactive visualization")
    print("  ✓ Click to explore patches and neighbors")
    print("\nInstructions:")
    print("  1. Upload an image")
    print("  2. Configure parameters (or use defaults)")
    print("  3. Click 'Run Pipeline'")
    print("  4. Switch to 'Visualization' tab")
    print("  5. Explore results!")
    print("\nPress Ctrl+C to stop the server\n")
    
    app = create_enhanced_app()
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    run_enhanced_app()