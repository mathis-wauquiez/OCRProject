"""
graphClusteringSweep — HOG sweep, graph building, metrics evaluation.

Splitting / refinement lives in ``refinement.py``.
Reporting is delegated to ``ClusteringSweepReporter``.
"""

# graph-related
import networkx as nx
import igraph as ig
import leidenalg as la

# arrays
import torch
import numpy as np
import pandas as pd

# sci
from scipy.stats import entropy

# path / typing
from pathlib import Path
from typing import List, Dict, Optional

# internal
from .feature_matching import featureMatching
from .params import featureMatchingParameters
from .clustering_sweep_report import ClusteringSweepReporter, UNKNOWN_LABEL

# Reporting base
from ..auto_report import AutoReport, ReportConfig, Theme
import logging

# HOG
from src.patch_processing.configs import get_hog_cfg
from src.patch_processing.hog import HOG


# Refinement pipeline
from .refinement import (
    RefinementResult,
    HausdorffSplitStep, OCRRematchStep, PCAZScoreRematchStep,
)

# New refinement methods
from .kmedoids_refinement import KMedoidsSplitMergeStep

from tqdm import tqdm


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
            output_dir: str = "./outputs/clustering/results",

            cell_sizes: None | List[int] = None,
            normalization_methods: None | List[None | str] = ['patch'],
            grdt_sigmas: None | List[float] = None,
            nums_bins: None | List[int] = None,

            # ── Cluster splitting parameters ──
            split_thresholds: Optional[List[float]] = None,
            split_linkage_method: str = 'average',
            split_min_cluster_size: int = 2,
            split_batch_size: int = 256,

            # ── Post-split rematching parameters ──
            rematch_max_cluster_size: int = 3,
            rematch_pca_k: int = 5,
            rematch_z_max: float = 3.0,
            rematch_n_candidates: int = 5,
            rematch_min_target_size: int = 10,

            # ── Explicit refinement pipeline (overrides above if given) ──
            refinement_steps: Optional[list] = None,

            embed_images: bool = False,
            image_dpi: int = 100,
            thumbnail_dpi: int = 50,
            use_jpeg: bool = True,
            jpeg_quality: int = 70,

            # ── Post-clustering refinement (optional) ──
            enable_chat_split: bool = False,
            chat_split_purity_threshold: float = 0.90,
            chat_split_min_size: int = 3,
            chat_split_min_label_count: int = 2,

            enable_hapax_association: bool = False,
            hapax_min_confidence: float = 0.3,
            hapax_max_dissimilarity: Optional[float] = None,

            enable_glossary: bool = False,
    ):
        config = ReportConfig(
            dpi=image_dpi,
            output_format='jpeg' if use_jpeg else 'png',
            image_quality=jpeg_quality,
            theme=Theme.DEFAULT,
            show_progress=False,
            max_image_size=(1920, 1080),
            include_toc=True
        )
        super().__init__(
            title="Clustering Sweep",
            author="Mathis",
            output_dir=Path(output_dir) / "reports",
            config=config,
            log_level=logging.INFO
        )

        featureMatcher = featureMatching(featureMatchingParameters(
            metric=metric, epsilon=epsilons[0], partial_output=False
        ))

        self.feature            = feature
        self.featureMatcher     = featureMatcher

        self.epsilons           = epsilons
        self.gammas             = gammas
        self.cell_sizes         = cell_sizes
        self.normalization_methods = normalization_methods
        self.grdt_sigmas        = grdt_sigmas
        self.nums_bins          = nums_bins

        self.edges_type         = edges_type
        self.target_lbl         = target_lbl
        self.keep_reciprocal    = keep_reciprocal
        self.device             = device

        # ── Cluster splitting config ──
        self.split_thresholds       = split_thresholds if split_thresholds is not None else [21.5]
        self.split_linkage_method   = split_linkage_method
        self.split_min_cluster_size = split_min_cluster_size
        self.split_batch_size       = split_batch_size

        # ── Post-split rematching config ──
        self.rematch_max_cluster_size   = rematch_max_cluster_size
        self.rematch_pca_k              = rematch_pca_k
        self.rematch_z_max              = rematch_z_max
        self.rematch_n_candidates       = rematch_n_candidates
        self.rematch_min_target_size    = rematch_min_target_size

        # ── Refinement pipeline ──
        self.refinement_steps           = refinement_steps

        # ── Post-clustering refinement (optional) ──
        self.enable_chat_split          = enable_chat_split
        self.chat_split_purity_threshold = chat_split_purity_threshold
        self.chat_split_min_size        = chat_split_min_size
        self.chat_split_min_label_count = chat_split_min_label_count

        self.enable_hapax_association    = enable_hapax_association
        self.hapax_min_confidence        = hapax_min_confidence
        self.hapax_max_dissimilarity     = hapax_max_dissimilarity

        self.enable_glossary             = enable_glossary

        self.embed_images   = embed_images
        self.image_dpi      = image_dpi
        self.thumbnail_dpi  = thumbnail_dpi
        self.use_jpeg       = use_jpeg
        self.jpeg_quality   = jpeg_quality

        # ── Build refinement pipeline ──────────────────────────────────────
        # If an explicit list is provided, use it; otherwise build from params.
        if refinement_steps is not None:
            self.refinement_steps = refinement_steps
        else:
            steps: list = []
            if self.split_thresholds:
                steps.append(HausdorffSplitStep(
                    thresholds=self.split_thresholds,
                    linkage_method=self.split_linkage_method,
                    min_cluster_size=self.split_min_cluster_size,
                    batch_size=self.split_batch_size,
                    evaluate_fn=self._evaluate_membership,
                ))
            steps.append(OCRRematchStep(
                max_cluster_size=rematch_max_cluster_size,
            ))
            steps.append(PCAZScoreRematchStep(
                max_cluster_size=rematch_max_cluster_size,
                pca_k=rematch_pca_k,
                z_max=rematch_z_max,
                n_candidates=rematch_n_candidates,
                min_target_size=rematch_min_target_size,
                device=device,
            ))
            self.refinement_steps = steps

        self.report_name   = self.metadata.report_id
        self.image_counter = 0
        if not self.embed_images:
            self.assets_dir = self.output_dir / f"assets_{self.report_name}"
            self.assets_dir.mkdir(parents=True, exist_ok=True)

        # ── Create the reporter ──
        self.reporter = ClusteringSweepReporter(self)

    # ================================================================
    #  Evaluation
    # ================================================================

    def _evaluate_membership(self, target_labels, membership, exclude_unknown=True):
        if exclude_unknown:
            mask = pd.Series(target_labels).fillna(UNKNOWN_LABEL) != UNKNOWN_LABEL
            target_labels = target_labels[mask]
            membership = [m for m, keep in zip(membership, mask) if keep]

        from .metrics import compute_metrics
        return compute_metrics(
            reference_labels=target_labels,
            predicted_labels=membership
        )

    # ================================================================
    #  Graph construction
    # ================================================================

    def get_graph(self, nlfa, dissimilarities, epsilon):
        if self.edges_type == 'nlfa':
            weight_matrix = nlfa
        elif self.edges_type in ['dissim', 'dissimilarities']:
            weight_matrix = dissimilarities
        elif self.edges_type in ['link', 'constant']:
            weight_matrix = torch.ones_like(nlfa)

        return self._get_matrix_graph(
            linkage_matrix=nlfa, threshold=epsilon,
            weight_matrix=weight_matrix
        )

    def _get_matrix_graph(self, linkage_matrix, threshold, weight_matrix):
        N = len(linkage_matrix)
        if self.edges_type == 'nlfa':
            threshold = -(np.log(threshold) - 2 * np.log(N))

        connected = linkage_matrix >= threshold
        if self.keep_reciprocal:
            connected &= linkage_matrix.T >= threshold
            weight_matrix = .5 * (weight_matrix + weight_matrix.T)

        edges = torch.nonzero(connected, as_tuple=False)
        edges = edges[edges[:, 0] != edges[:, 1]]
        edges_list = [
            (int(i.item()), int(j.item()), weight_matrix[i, j].item())
            for i, j in edges
        ]

        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_weighted_edges_from(edges_list)
        return G, edges

    # ================================================================
    #  Purity & representatives (pure computation)
    # ================================================================

    def _compute_purity_and_representatives(self, dataframe, membership_col):
        purity_data = []
        representatives = {}

        for cluster, cluster_data in dataframe.groupby(membership_col):
            cluster_size = len(cluster_data)
            known_mask = cluster_data[self.target_lbl].fillna(UNKNOWN_LABEL) != UNKNOWN_LABEL
            known_data = cluster_data[known_mask]
            unknown_count = (~known_mask).sum()

            label_counts = (
                known_data[self.target_lbl].value_counts()
                if len(known_data) > 0
                else pd.Series(dtype=int)
            )
            if len(label_counts) > 0:
                known_size = len(known_data)
                label_probs = label_counts / known_size
                cluster_entropy = entropy(label_probs, base=2)
                cluster_ne = (cluster_entropy / np.log2(len(label_counts))
                              if len(label_counts) > 1 else 0)
                purity = label_counts.iloc[0] / known_size
                dominant_label = label_counts.index[0]
                unique_labels = len(label_counts)
            else:
                cluster_entropy = np.nan
                cluster_ne = np.nan
                purity = np.nan
                dominant_label = UNKNOWN_LABEL
                unique_labels = 0
                label_counts = pd.Series(dtype=int)

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
                'Unique_Labels': unique_labels,
            })

        purity_df = pd.DataFrame(purity_data)
        purity_df.set_index('Cluster', inplace=True)
        return purity_df, representatives

    # ================================================================
    #  Completeness (label-level stats)
    # ================================================================

    def _compute_label_dataframe(self, dataframe):
        label_data = []
        known_df = dataframe[
            dataframe[self.target_lbl].fillna(UNKNOWN_LABEL) != UNKNOWN_LABEL
        ]
        for label, label_rows in tqdm(known_df.groupby(self.target_lbl),
                                      desc="Computing completeness", colour='blue'):
            label_size = len(label_rows)
            cluster_counts = label_rows['membership'].value_counts()
            cluster_probs = cluster_counts / label_size
            label_entropy = entropy(cluster_probs, base=2)
            label_ne = (label_entropy / np.log2(len(cluster_counts))
                        if len(cluster_counts) > 1 else 0)
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
        return pd.DataFrame(label_data)

    # ================================================================
    #  report_graph: ALL computation, then ONE reporting call
    # ================================================================

    def report_graph(self, dataframe, nlfa, dissimilarities,
                     best_epsilon, best_gamma, renderer=None):

        # ── 1. Build graph & partition ──
        graph, edges = self.get_graph(nlfa, dissimilarities, best_epsilon)
        G_ig = ig.Graph.from_networkx(graph)
        partition = la.find_partition(
            G_ig, la.RBConfigurationVertexPartition,
            resolution_parameter=best_gamma, n_iterations=-1, seed=42
        )

        # ── 2. Degree centrality ──
        degree_centrality = nx.degree_centrality(graph)
        dataframe['degree_centrality'] = pd.Series(
            [degree_centrality[i] for i in range(len(dataframe))],
            index=dataframe.index
        )

        # ── 3. Pre-split membership ──
        pre_split_membership = np.array(partition.membership)
        dataframe.loc[:, 'membership_pre_split'] = pd.Series(
            pre_split_membership, index=dataframe.index
        )

        # ── 4. Refinement pipeline ──
        #   Each step takes the current membership and returns a new one.
        #   Results are collected for per-step reporting.
        current_membership = pre_split_membership.copy()
        refinement_results: list[RefinementResult] = []
        refinement_step_names: list[str] = []

        has_renderer = renderer is not None
        for step in self.refinement_steps:
            if not has_renderer and isinstance(step, (HausdorffSplitStep, PCAZScoreRematchStep)):
                continue  # skip GPU steps when no renderer available
            result = step.run(
                dataframe, current_membership, renderer,
                target_lbl=self.target_lbl,
                graph=graph,
            )
            current_membership = result.membership
            refinement_results.append(result)
            refinement_step_names.append(step.name)

        post_split_membership = current_membership
        dataframe.loc[:, 'membership'] = pd.Series(
            post_split_membership, index=dataframe.index
        )
        did_split = len(refinement_results) > 0

        # Extract split-specific metadata for backward-compat reporting
        split_result = next(
            (r for r, n in zip(refinement_results, refinement_step_names)
             if n == 'hausdorff_split'), None
        )
        best_threshold = (split_result.metadata.get('best_threshold')
                          if split_result else None)
        best_split_log = split_result.log if split_result else None
        sweep_results_df = (split_result.metadata.get('sweep_df')
                            if split_result else None)

        # ── 5. Reorder by membership ──
        dataframe['_graph_node_id'] = range(len(dataframe))
        dataframe = dataframe.sort_values(
            by=['membership', 'degree_centrality'],
            ascending=[True, False]
        ).reset_index(drop=True)

        # ── 5a. Realign graph, matrices, and membership arrays ──
        # After sorting + reset_index the dataframe has new 0..N-1 indices,
        # but the graph nodes, distance matrices, and raw membership arrays
        # still use the OLD row order.  Re-map everything to the new order
        # so that downstream code (subgraph extraction, NN lookup, etc.)
        # can use dataframe indices directly as graph node IDs.
        perm = dataframe.pop('_graph_node_id').values
        old_to_new = {int(old): new for new, old in enumerate(perm)}
        graph = nx.relabel_nodes(graph, old_to_new)

        perm_t = torch.tensor(perm, device=nlfa.device, dtype=torch.long)
        nlfa = nlfa[perm_t][:, perm_t]
        dissimilarities = dissimilarities[perm_t][:, perm_t]

        pre_split_membership = pre_split_membership[perm]
        post_split_membership = np.asarray(post_split_membership)[perm]

        # ── 5b. Post-clustering refinement (optional) ──
        from .post_clustering import chat_split_clusters, associate_hapax, build_glossary

        chat_split_log = None
        hapax_log = None
        glossary_df = None

        if self.enable_chat_split and self.target_lbl in dataframe.columns:
            dataframe, chat_split_log = chat_split_clusters(
                dataframe, dissimilarities,
                purity_threshold=self.chat_split_purity_threshold,
                min_split_size=self.chat_split_min_size,
                min_label_count=self.chat_split_min_label_count,
                target_lbl=self.target_lbl,
            )

        if self.enable_hapax_association and self.target_lbl in dataframe.columns:
            dataframe, hapax_log = associate_hapax(
                dataframe, dissimilarities,
                target_lbl=self.target_lbl,
                min_confidence=self.hapax_min_confidence,
                max_dissimilarity=self.hapax_max_dissimilarity,
            )

        if self.enable_glossary and self.target_lbl in dataframe.columns:
            glossary_df = build_glossary(dataframe, target_lbl=self.target_lbl)

        # ── 5c. Compute purity & representatives AFTER index reset ──
        #     so that stored representative indices match the new index.
        pre_purity_df, pre_representatives = \
            self._compute_purity_and_representatives(dataframe, 'membership_pre_split')

        if did_split:
            purity_dataframe, representatives = \
                self._compute_purity_and_representatives(dataframe, 'membership')
        else:
            purity_dataframe = pre_purity_df
            representatives = pre_representatives

        # ── 6. Post-split metrics & label completeness ──
        best_metrics = self._evaluate_membership(
            target_labels=dataframe[self.target_lbl],
            membership=dataframe['membership'].tolist()
        )
        label_dataframe = self._compute_label_dataframe(dataframe)

        # ── 7. Delegate ALL reporting in one call ──
        self.reporter.report_graph_results(
            dataframe=dataframe,
            graph=graph,
            partition=partition,
            nlfa=nlfa,
            dissimilarities=dissimilarities,
            best_epsilon=best_epsilon,
            best_gamma=best_gamma,
            pre_split_membership=pre_split_membership,
            post_split_membership=post_split_membership,
            best_threshold=best_threshold,
            best_split_log=best_split_log,
            sweep_results_df=sweep_results_df,
            did_split=did_split,
            purity_dataframe=purity_dataframe,
            representatives=representatives,
            pre_purity_df=pre_purity_df,
            pre_representatives=pre_representatives,
            best_metrics=best_metrics,
            label_dataframe=label_dataframe,
            refinement_results=refinement_results,
            refinement_step_names=refinement_step_names,
            chat_split_log=chat_split_log,
            hapax_log=hapax_log,
            glossary_df=glossary_df,
        )

        # ── 8. Build filtered / representative dataframes ──
        min_size = 2
        filtered_dataframe = dataframe[
            dataframe.groupby(self.target_lbl)[self.target_lbl]
            .transform('size') >= min_size
        ]
        label_representatives_dataframe = filtered_dataframe.loc[
            filtered_dataframe.groupby(self.target_lbl)['degree_centrality'].idxmax()
        ]

        return (dataframe, filtered_dataframe,
                label_representatives_dataframe, graph, partition)

    # ================================================================
    #  __call__: HOG config sweep → best config → report_graph
    # ================================================================

    def __call__(self, dataframe):
        from itertools import product
        from torch.utils.data import DataLoader

        if self.cell_sizes is None:
            self.cell_sizes = [24]
        if self.grdt_sigmas is None:
            self.grdt_sigmas = [5]
        if self.nums_bins is None:
            self.nums_bins = [16]

        # Generate all HOG configurations
        hog_configs = {}
        for cell_size, grdt_sigma, num_bins, normalize in product(
            self.cell_sizes, self.grdt_sigmas, self.nums_bins,
            self.normalization_methods
        ):
            normalization_name = ('unnormalized'
                                  if normalize is False or normalize is None
                                  else normalize)
            config_name = f'hog_{cell_size}_{num_bins}_{grdt_sigma}_{normalization_name}'
            renderer, hog_params = get_hog_cfg(
                cell_size, grdt_sigma, num_bins, normalize, device=self.device
            )
            hog_configs[config_name] = {
                'renderer': renderer, 'hog_params': hog_params,
                'cell_size': cell_size, 'grdt_sigma': grdt_sigma,
                'num_bins': num_bins, 'normalization': normalize,
            }

        all_results = []
        best_overall_score = -np.inf
        best_overall_config = None

        for config_name, hog_cfg in tqdm(hog_configs.items(),
                                         desc="HOG Config", colour="cyan"):
            print(f"\n{'='*60}\nProcessing HOG Configuration: {config_name}\n{'='*60}")

            hog_processor = HOG(hog_cfg['hog_params'])
            hog_renderer = hog_cfg['renderer'](dataframe['svg'])

            dataloader = DataLoader(
                hog_renderer, batch_size=256, shuffle=False,
                num_workers=0, pin_memory=True
            )

            # Preallocate histograms
            first_batch = next(iter(dataloader))
            sample_output = hog_processor(
                first_batch[:1].unsqueeze(1).to(dtype=torch.float32, device=self.device)
            )
            total_samples = len(dataframe)
            histogram_shape = sample_output.histograms[0, 0].shape
            histograms = torch.zeros(
                (total_samples, *histogram_shape), device=self.device
            )

            start_idx = 0
            for batch in tqdm(dataloader, desc="Computing HOG",
                              colour="red", leave=False):
                hogOutput = hog_processor(
                    batch.unsqueeze(1).to(dtype=torch.float32, device=self.device)
                )
                histogram_batch = hogOutput.histograms[:, 0]
                batch_size = histogram_batch.shape[0]
                histograms[start_idx:start_idx + batch_size] = histogram_batch
                start_idx += batch_size

            features = histograms


            # ==== A contrario ====

            _, nlfa, dissimilarities, mu_tot, var_tot = \
                self.featureMatcher.match(features, features)
            dataframe['mu_tot'] = mu_tot.cpu().numpy()  # store for use in other scripts
            dataframe['var_tot'] = var_tot.cpu().numpy()


            # ==== Sweep epsilon × gamma ====
            config_results = []
            for epsilon in tqdm(self.epsilons, desc="Epsilon",
                                leave=False, colour="blue"):
                
                graph, edges = self.get_graph(nlfa, dissimilarities, epsilon)
                G_ig = ig.Graph.from_networkx(graph)

                for gamma in tqdm(self.gammas, desc="Gamma",
                                  leave=False, colour="green"):
                    
                    partition = la.find_partition(
                        G_ig, la.RBConfigurationVertexPartition,
                        resolution_parameter=gamma, n_iterations=10, seed=42
                    )
                    metrics = self._evaluate_membership(
                        target_labels=dataframe[self.target_lbl],
                        membership=partition.membership
                    )
                    result = {
                        'hog_config': config_name,
                        'cell_size': hog_cfg['cell_size'],
                        'grdt_sigma': hog_cfg['grdt_sigma'],
                        'num_bins': hog_cfg['num_bins'],
                        'epsilon': epsilon,
                        'gamma': gamma,
                        **metrics
                    }
                    config_results.append(result)
                    all_results.append(result)

            config_df = pd.DataFrame(config_results)
            best_idx = config_df['adjusted_rand_index'].idxmax()
            best_score = config_df.loc[best_idx, 'adjusted_rand_index']

            if best_score > best_overall_score:
                best_overall_score = best_score
                best_overall_config = {
                    'config_name': config_name,
                    'hog_cfg': hog_cfg,
                    'features': features,
                    'nlfa': nlfa,
                    'dissimilarities': dissimilarities,
                    **config_df.loc[best_idx].to_dict()
                }

            del features, nlfa, dissimilarities, histograms, hog_processor, hog_renderer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results_df = pd.DataFrame(all_results)

        print(f"\n{'='*60}\nBEST OVERALL CONFIGURATION\n{'='*60}")
        print(f"HOG Config: {best_overall_config['config_name']}")
        print(f"Epsilon: {best_overall_config['epsilon']:.4f}")
        print(f"Gamma: {best_overall_config['gamma']:.4f}")
        print(f"ARI: {best_overall_config['adjusted_rand_index']:.4f}")

        # ── HOG sweep report section ──
        self.reporter.report_hog_sweep(results_df, hog_configs)

        # ── Use best config for detailed graph report ──
        dataframe['histogram'] = list(best_overall_config['features'].cpu().numpy())
        # Re-instantiate the renderer for the best config (the factory was stored,
        # not the instance, which was deleted earlier to free GPU memory).
        best_renderer = best_overall_config['hog_cfg']['renderer'](dataframe['svg'])

        dataframe, filtered_dataframe, label_representatives_dataframe, \
            graph, partition = self.report_graph(
                dataframe,
                best_overall_config['nlfa'],
                best_overall_config['dissimilarities'],
                best_overall_config['epsilon'],
                best_overall_config['gamma'],
                renderer=best_renderer,
            )

        # Annotate outputs with best config info
        for df in [dataframe, filtered_dataframe, label_representatives_dataframe]:
            df['best_hog_config'] = best_overall_config['config_name']
            df['best_cell_size']  = best_overall_config['cell_size']
            df['best_grdt_sigma'] = best_overall_config['grdt_sigma']
            df['best_num_bins']   = best_overall_config['num_bins']

        return (dataframe, filtered_dataframe,
                label_representatives_dataframe, graph, partition)