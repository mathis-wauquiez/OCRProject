
# graph-related
import networkx as nx
import igraph as ig
import leidenalg as la

# "array imports"
import torch
import numpy as np
import pandas as pd

# sci-stuff
from scipy.stats import entropy

# path things
from pathlib import Path

# type hinting
from .graph_clustering import communityDetectionBase
from .feature_matching import featureMatching
from typing import List, Dict
from .params import featureMatchingParameters

# Reporting
from ..auto_report import AutoReport, ReportConfig, Theme
import logging
import matplotlib.pyplot as plt
from .graph_visu import matches_per_treshold, random_match_figure, size_distribution_figure, purity_figure, completeness_figure, report_community, plot_nearest_neighbors
from .tsne_plot import plot_community_tsne
from tqdm import tqdm

import io
import base64

UNKNOWN_LABEL = '▯'  # U+25AF - represents unrecognized characters


import warnings
def _get_b64(fig, dpi=75, quality=70):
    # Convert figure to base64
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Glyph.*missing from font')
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi, 
                    pil_kwargs={'quality': quality})
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_str

class graphClusteringSweep(AutoReport):
    def __init__(
            self,
            feature: str,

            epsilons: List[float],
            gammas: List[float],

            target_lbl: str,
            
            edges_type: str,
            metric: str = "CEMD",
            keep_reciprocal: bool = True,
            device: str = "cuda",
            output_dir: str = "./outputs/clustering_results",

            embed_images: bool = False,  # False = save as files
            image_dpi: int = 100,
            thumbnail_dpi: int = 50,
            use_jpeg: bool = True,
            jpeg_quality: int = 70,

    ):
        
        config = ReportConfig(
            dpi=300,
            theme=Theme.DEFAULT,
            show_progress=False,
            max_image_size=(1920, 1080),
            include_toc=True
        )
        
        # Create report with configuration
        super().__init__(
            title="Clustering Sweep",
            author="Mathis",
            output_dir=Path(output_dir) / "reports",
            config=config,
            log_level=logging.INFO
        )

        
        featureMatcher = featureMatching(featureMatchingParameters(
            metric = metric,
            epsilon= epsilons[0],
            partial_output=False
        ))

        self.feature            = feature
        self.featureMatcher     = featureMatcher
        self.epsilons           = epsilons
        self.edges_type         = edges_type
        self.gammas             = gammas
        self.target_lbl         = target_lbl
        self.keep_reciprocal    = keep_reciprocal
        self.device             = device

        self.embed_images = embed_images
        self.image_dpi = image_dpi
        self.thumbnail_dpi = thumbnail_dpi
        self.use_jpeg = use_jpeg
        self.jpeg_quality = jpeg_quality

        self.report_name = self.metadata.report_id
        self.image_counter = 0
        if not self.embed_images:
            self.assets_dir = self.output_dir / f"assets_{self.report_name}"
            self.assets_dir.mkdir(parents=True, exist_ok=True)


    def _evaluate_membership(self, target_labels, membership, exclude_unknown=True):
        if exclude_unknown:
            mask = target_labels != UNKNOWN_LABEL
            target_labels = target_labels[mask]
            membership = [m for m, keep in zip(membership, mask) if keep]
        
        from .metrics import compute_metrics
        metrics = compute_metrics(
            reference_labels=target_labels,
            predicted_labels=membership
        )
        return metrics
    
    def get_graph(self, nlfa, dissimilarities, epsilon):

        if self.edges_type == 'nlfa':
            weight_matrix = nlfa
        elif self.edges_type in ['dissim', 'dissimilarities']:
            weight_matrix = dissimilarities
        elif self.edges_type in ['link', 'constant']:
            weight_matrix = torch.ones_like(nlfa)

        return self._get_matrix_graph(
            linkage_matrix=nlfa,
            threshold=epsilon,
            weight_matrix=weight_matrix
        )

    def _get_matrix_graph(self, linkage_matrix, threshold, weight_matrix):
        # Build the edges list
        N = len(linkage_matrix)
        if self.edges_type == 'nlfa':
            # = side note: we could modify 2 --> 3 depending on our stats model =
            threshold = -(np.log(threshold) - 2 * np.log(N))

        connected = linkage_matrix >= threshold
        if self.keep_reciprocal:
            connected &= linkage_matrix.T >= threshold
            weight_matrix = .5 * (weight_matrix + weight_matrix.T)

        # Build the graph
        edges = torch.nonzero(connected, as_tuple=False)
        edges = edges[edges[:,0] != edges[:, 1]]
        edges_list = [
            (int(i.item()),
             int(j.item()),
             weight_matrix[i, j].item())
             for i, j in edges]
        
        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_weighted_edges_from(edges_list)

        return G, edges


    def __call__(
            self, 
            dataframe
    ):
        
        # -- Compute the matches thanks to the featureMatching class --
        if dataframe[self.feature].ndim == 1:
            dataframe[self.feature] = dataframe[self.feature].map(lambda x: x.reshape(-1, 8))
        
        features = dataframe[self.feature]
        features = torch.tensor(features, device=self.device)

        _, nlfa, dissimilarities = self.featureMatcher.match(features, features)

        results = []

        for epsilon in tqdm(self.epsilons, desc="Epsilon", colour="blue"):
            graph, edges = self.get_graph(nlfa, dissimilarities, epsilon)
            G_ig = ig.Graph.from_networkx(graph)

            for gamma in tqdm(self.gammas, desc="Gamma", leave=False, colour="green"):

                partition = la.find_partition(G_ig, la.RBConfigurationVertexPartition, resolution_parameter=gamma, n_iterations=10, seed=42)
                metrics = self._evaluate_membership(target_labels=dataframe[self.target_lbl], membership=partition.membership)
                results.append({
                    'epsilon': epsilon,
                    'gamma': gamma,
                    **metrics
                })


        results_df = pd.DataFrame(results)
        
        # Find best combination by v_measure
        best_idx = results_df['adjusted_rand_index'].idxmax()
        best_row = results_df.loc[best_idx]
        best_epsilon = best_row['epsilon']
        best_gamma = best_row['gamma']

        print(best_epsilon, best_gamma)

        metric_names = [col for col in results_df.columns if col not in ['epsilon', 'gamma']]
    
        for metric in metric_names:
            pivot = results_df.pivot_table(values=metric, index='gamma', columns='epsilon')
            self.report_table(pivot, title=f'{metric} (gamma × epsilon)')

        fig = self.report_parameter_heatmaps(results_df)
        self.report_figure(fig, title="Parameter Sweep Heatmaps")


        dataframe, filtered_dataframe, label_representatives_dataframe = self.report_graph(
            dataframe, nlfa, dissimilarities, best_epsilon, best_gamma
        )
        
        return dataframe, filtered_dataframe, label_representatives_dataframe


    def report_graph(self, dataframe, nlfa, dissimilarities, best_epsilon, best_gamma):
            
            # == Build the grah / evaluate ==
            graph, edges = self.get_graph(nlfa, dissimilarities, best_epsilon)
            G_ig = ig.Graph.from_networkx(graph)
            partition = la.find_partition(G_ig, la.RBConfigurationVertexPartition, resolution_parameter=best_gamma, n_iterations=-1, seed=42)
            dataframe['membership'] = partition.membership
            degree_centrality = nx.degree_centrality(graph)
            dataframe['degree_centrality'] = dataframe.index.map(degree_centrality)

            best_metrics = self._evaluate_membership(target_labels=dataframe[self.target_lbl], membership=partition.membership)

            #! Report the metrics
            best_metrics_df = pd.DataFrame([{
                'epsilon': best_epsilon,
                'gamma': best_gamma,
                **best_metrics
            }])

            #? Compute the purity stats

            purity_data = []
            representatives = {}

            for cluster, cluster_data in tqdm(dataframe.groupby('membership'), desc="Computing purity", colour='green'):
                cluster_size = len(cluster_data)
                
                # Separate known vs unknown labels
                known_mask = cluster_data[self.target_lbl] != UNKNOWN_LABEL
                known_data = cluster_data[known_mask]
                unknown_count = (~known_mask).sum()
                
                if len(known_data) > 0:
                    label_counts = known_data[self.target_lbl].value_counts()
                    known_size = len(known_data)
                    
                    label_probs = label_counts / known_size
                    cluster_entropy = entropy(label_probs, base=2)
                    
                    if len(label_counts) > 1:
                        cluster_ne = cluster_entropy / np.log2(len(label_counts))
                    else:
                        cluster_ne = 0
                    
                    purity = label_counts.iloc[0] / known_size
                    dominant_label = label_counts.index[0]
                    unique_labels = len(label_counts)
                else:
                    # Cluster contains only unknowns
                    cluster_entropy = np.nan
                    cluster_ne = np.nan
                    purity = np.nan
                    dominant_label = UNKNOWN_LABEL
                    unique_labels = 0
                    label_counts = pd.Series(dtype=int)
                
                # Find representatives only among known labels
                label_representatives = {}
                for label in label_counts.index:
                    if label == UNKNOWN_LABEL:
                        continue

                    label_nodes = cluster_data[cluster_data[self.target_lbl] == label]
                    most_central_idx = label_nodes['degree_centrality'].idxmax()
                    label_representatives[label] = most_central_idx
                
                representatives[cluster] = label_representatives
                
                purity_data.append({
                    'Cluster': cluster,
                    'Size': cluster_size,
                    'Known_Size': len(known_data),
                    'Unknown_Count': unknown_count,
                    'Purity': purity,
                    'Entropy': cluster_entropy,
                    'Normalized entropy': cluster_ne,
                    'Dominant_Label': dominant_label,
                    'Unique_Labels': unique_labels
                })

            purity_dataframe = pd.DataFrame(purity_data)
            purity_dataframe.set_index('Cluster', inplace=True)

            #? Compute the completeness stats

            label_data = []

            for label, label_rows in tqdm(dataframe.groupby(self.target_lbl), desc="Computing completeness", colour='blue'):
                label_size = len(label_rows)
                cluster_counts = label_rows['membership'].value_counts()
                
                cluster_probs = cluster_counts / label_size
                label_entropy = entropy(cluster_probs, base=2)
                if len(cluster_counts) > 1:
                    label_ne = label_entropy / np.log2(len(cluster_counts))
                else:
                    label_ne = 0
                best_share = cluster_counts.iloc[0] / label_size
                
                label_data.append({
                    'Label': label,
                    'Size': label_size,
                    'Best share': best_share,
                    'Entropy': label_entropy,
                    'Normalized entropy': label_ne,
                    'Dominant_Cluster': cluster_counts.index[0],
                    'Unique_Clusters': len(cluster_counts)
                })

            label_dataframe = pd.DataFrame(label_data)


            # == Report general metrics about the clustering ==

            # a. Metrics
            self.report_table(best_metrics_df.T, title=f'Best Parameters (ε={best_epsilon:.4f}, γ={best_gamma:.4f})')

            # b. Summary
            self.report_executive_summary(dataframe, purity_dataframe, label_dataframe, 
                                best_epsilon, best_gamma)

            # b. Connectivity
            self.report(matches_per_treshold(nlfa, best_epsilon), title="Average number of matches = f(epsilon)")

            # c. Cluster size distribution
            self.report(size_distribution_figure(dataframe['membership'], dataframe[self.target_lbl]), title="Cluster Size Distribution")

            # d. Purity - share that the dominant label takes in each cluster

            self.report_figure(purity_figure(purity_dataframe), title="Purity of the clusters")

            # e. Completeness measure - across how much clusters does one label spread?

            self.report_figure(completeness_figure(label_dataframe), title="Compleness")

            # f. Happax statistics

            # Find singleton clusters
            cluster_sizes = dataframe['membership'].value_counts()
            hapax_clusters = cluster_sizes[cluster_sizes == 1].index
            hapax_df = dataframe[dataframe['membership'].isin(hapax_clusters)]

            # Statistics
            hapax_stats = pd.DataFrame({
                'Count': [len(hapax_df)],
                'Percentage': [100 * len(hapax_df) / len(dataframe)],
                'Unique_Labels': [hapax_df[self.target_lbl].nunique()]
            })
            self.report_table(hapax_stats, title="Hapax (Singleton Clusters)")

            # g. Stats about unknown characters

            unknown_df = dataframe[dataframe[self.target_lbl] == UNKNOWN_LABEL]
            unknown_stats = pd.DataFrame({
                'Total Unknown': [len(unknown_df)],
                'Percentage': [100 * len(unknown_df) / len(dataframe)],
                'Clusters Containing Unknown': [unknown_df['membership'].nunique()],
                'Pure Unknown Clusters': [(dataframe.groupby('membership')[self.target_lbl]
                                        .apply(lambda x: (x == UNKNOWN_LABEL).all())).sum()]
            })
            self.report_table(unknown_stats, title="Unknown Character Statistics (▯)")
            
            # Label distribution
            # label_dist = hapax_df[self.target_lbl].value_counts()
            # self.report_table(label_dist.head(20), title="Hapax Label Distribution (Top 20)")

            # == Report random instances of characters and their NNs ==

            nn_examples_html = '<div style="display: grid; gap: 30px;">'

            for i in tqdm(range(30), desc="Random NN examples", colour='cyan'):
                idx = np.random.randint(len(dataframe))
                
                fig = plot_nearest_neighbors(
                    query_idx=idx,
                    dataframe=dataframe,
                    dissimilarities=dissimilarities,
                    graph=graph,
                    n_to_show=23
                )
                
                img_tag = self._save_figure(fig, prefix="nn_example")
                
                nn_examples_html += f'''
                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #667eea; margin-top: 0;">Example {i+1}: Nearest Neighbors for Sample {idx}</h3>
                    {img_tag}
                </div>
                '''

            nn_examples_html += '</div>'

            self.report_raw_html(nn_examples_html, title="Random Nearest Neighbor Examples (10 samples)")

            # == Report the communities ==

            # a. Report each community's labels and stats
            cluster_sizes = dataframe.groupby('membership').size().sort_values(ascending=False)
            clusters_html = '<div style="display: grid; gap: 30px;">'
            
            min_cluster_size = 2

            for cluster in tqdm(cluster_sizes[cluster_sizes >= min_cluster_size].index, 
                                desc="Reporting clusters", colour='magenta'):
                cluster_df = dataframe[dataframe['membership'] == cluster]

                cluster_stats = purity_dataframe.loc[cluster]
                label_representatives = representatives[cluster]

                fig = report_community(cluster, cluster_stats, cluster_df, label_representatives, self.target_lbl)

                img_tag = self._save_figure(fig, prefix="cluster")
                
                purity_display = f"{cluster_stats['Purity']:.2%}" if not np.isnan(cluster_stats['Purity']) else "N/A"
                
                clusters_html += f'''
                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #667eea; margin-top: 0;">Cluster {cluster} - Representatives</h3>
                    <p><strong>Size:</strong> {cluster_stats["Size"]} | <strong>Purity:</strong> {purity_display}</p>
                    {img_tag}
                '''

                if len(cluster_df) >= 10:
                    fig = plot_community_tsne(
                        cluster_id=cluster,
                        dataframe=dataframe,
                        graph=graph,
                        target_lbl=self.target_lbl
                    )

                    tsne_img_tag = self._save_figure(fig, prefix="tsne")
                    
                    clusters_html += f'''
                        <h4 style="color: #667eea; margin-top: 20px;">t-SNE Visualization</h4>
                        {tsne_img_tag}
                    '''
                
                clusters_html += '</div>'
            clusters_html += '</div>'
            self.report_raw_html(clusters_html, title=f"All Clusters Analysis ({len(cluster_sizes)} clusters)")


            # b. Report some interesting 

            # == Report some examples on specific instances of Hanzi ==

            match_examples_html = '<div style="display: grid; gap: 30px;">'

            for i in tqdm(range(30), desc="Random match figures", colour='yellow'):
                fig1, fig2, idx = random_match_figure(self.featureMatcher, dataframe['histogram'], best_epsilon, svgs=dataframe['svg'])

                match_examples_html += f'''
                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #667eea; margin-top: 0;">Example {i+1}: Sample {idx}</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <h4>Distribution</h4>
                            <img src="data:image/png;base64,{_get_b64(fig1)}" style="max-width: 100%; height: auto;">
                        </div>
                        <div>
                            <h4>Matches</h4>
                            <img src="data:image/png;base64,{_get_b64(fig2)}" style="max-width: 100%; height: auto;">
                        </div>
                    </div>
                </div>
                '''

            match_examples_html += '</div>'

            self.report_raw_html(match_examples_html, title="Random Match Examples (10 samples)")


                # self.report(fig1, title=f'Distribution for sample {idx}')
                # self.report(fig2, title=f'Matches for sample {idx}')

            # == Hapax Report ==

            # a. NNs

            idxs = list(np.random.permutation(hapax_df.index))[:10]
            hapax_nn_html = '<div style="display: grid; gap: 30px;">'

            for idx in tqdm(idxs, desc="Hapax NN examples", colour='red'):            
                fig = plot_nearest_neighbors(
                    query_idx=idx,
                    dataframe=dataframe,
                    dissimilarities=dissimilarities,
                    graph=graph,
                    n_to_show=23
                )
                
                # self.report(fig, title=f'Nearest neighbors for happax sample {idx}')
                hapax_nn_html += f'''
                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #667eea; margin-top: 0;">Hapax Example - Sample {idx}</h3>
                    <img src="data:image/png;base64,{_get_b64(fig)}" style="max-width: 100%; height: auto;">
                </div>
                '''

            hapax_nn_html += '</div>'

            self.report_raw_html(hapax_nn_html, title="Hapax Nearest Neighbor Examples (10 samples)")

            
            # b. All Happax

            html_parts = ['<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 15px;">']
            
            for idx, row in hapax_df.iterrows():
                svg_string = row['svg'].to_string()
                label = row[self.target_lbl]
                
                html_parts.append(f'''
                <div style="border: 1px solid #ddd; padding: 10px; text-align: center; border-radius: 5px;">
                    <div style="width: 100px; height: 100px; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                        {svg_string}
                    </div>
                    <div style="margin-top: 5px; font-size: 11px; font-weight: bold;">{label}</div>
                    <div style="font-size: 9px; color: #666;">idx: {idx}</div>
                </div>
                ''')
            
            html_parts.append('</div>')
            
            self.report_raw_html(''.join(html_parts), title=f"All Hapax ({len(hapax_df)} items)")

            # == Save the label representatives for Edwin ==

            # keep only the characters that are present more than a few times in the document
            # for each "pseudo-gt" label, yield the most connected in the graph
            min_size = 2

            filtered_dataframe = dataframe[
                dataframe.groupby(self.target_lbl)[self.target_lbl].transform('size') >= min_size
            ]

            label_representatives_dataframe = filtered_dataframe.loc[
                filtered_dataframe.groupby(self.target_lbl)['degree_centrality'].idxmax()
            ]

            #! Save these dataframes!

            return dataframe, filtered_dataframe, label_representatives_dataframe
    
    """ ================================================================== """
    """ ====                                                          ==== """
    """ = Under this line is GPT code only. Above, it was human-redacted = """
    """ ====                                                          ==== """
    """ ================================================================== """


    def report_parameter_heatmaps(self, results_df):
        """
        Create heatmaps for parameter sweep results.
        """
        import seaborn as sns
        
        metric_names = [col for col in results_df.columns if col not in ['epsilon', 'gamma']]
        
        # Create a 2x3 grid of heatmaps
        n_metrics = len(metric_names)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, metric in enumerate(metric_names):
            ax = axes[idx]
            pivot = results_df.pivot_table(values=metric, index='gamma', columns='epsilon')
            
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                    ax=ax, cbar_kws={'label': metric})
            ax.set_title(f'{metric} vs Parameters', fontweight='bold')
            ax.set_xlabel('Epsilon (ε)')
            ax.set_ylabel('Gamma (γ)')
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def report_executive_summary(self, dataframe, purity_dataframe, label_dataframe, best_epsilon, best_gamma):
        """
        Generate an executive summary with key findings.
        """
        n_clusters = len(purity_dataframe)
        avg_purity = purity_dataframe['Purity'].mean()
        avg_size = purity_dataframe['Size'].mean()
        n_hapax = (purity_dataframe['Size'] == 1).sum()
        
        # Find best and worst clusters
        best_cluster = purity_dataframe['Purity'].idxmax()
        worst_cluster = purity_dataframe['Purity'].idxmin()
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
                    <small>Range: {purity_dataframe['Purity'].min():.1%} - {purity_dataframe['Purity'].max():.1%}</small>
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
                    <li>Best Parameters: ε={best_epsilon:.4f}, γ={best_gamma:.4f}</li>
                    <li>Purest Cluster: #{best_cluster} ({purity_dataframe.loc[best_cluster, 'Purity']:.1%} purity)</li>
                    <li>Most Scattered Cluster: #{worst_cluster} ({purity_dataframe.loc[worst_cluster, 'Purity']:.1%} purity)</li>
                    <li>Largest Cluster: #{largest_cluster} ({purity_dataframe.loc[largest_cluster, 'Size']} patches)</li>
                </ul>
            </div>
        </div>
        """
        
        self.report_raw_html(summary_html, title="Executive Summary")



    def _save_figure(self, fig, prefix="fig"):
        """
        Save figure efficiently. Returns HTML img tag.
        """
        self.image_counter += 1
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Glyph.*missing from font')
            
            if self.embed_images:
                # Embed as base64 (larger file, single HTML)
                buf = io.BytesIO()
                fmt = 'jpeg' if self.use_jpeg else 'png'
                save_kwargs = {'format': fmt, 'bbox_inches': 'tight', 'dpi': self.image_dpi}
                if self.use_jpeg:
                    save_kwargs['pil_kwargs'] = {'quality': self.jpeg_quality}
                
                fig.savefig(buf, **save_kwargs)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode()
                plt.close(fig)
                
                mime = 'jpeg' if self.use_jpeg else 'png'
                return f'<img src="data:image/{mime};base64,{img_str}" style="max-width: 100%; height: auto;" loading="lazy">'
            
            else:
                # Save as external file (smaller HTML, multiple files)
                ext = 'jpg' if self.use_jpeg else 'png'
                filename = f"{prefix}_{self.image_counter:04d}.{ext}"
                filepath = self.assets_dir / filename
                
                save_kwargs = {'format': 'jpeg' if self.use_jpeg else 'png', 
                              'bbox_inches': 'tight', 'dpi': self.image_dpi}
                if self.use_jpeg:
                    save_kwargs['pil_kwargs'] = {'quality': self.jpeg_quality}
                
                fig.savefig(filepath, **save_kwargs)
                plt.close(fig)
                
                return f'<img src="assets_{self.report_name}/{filename}" style="max-width: 100%; height: auto;" loading="lazy">'



