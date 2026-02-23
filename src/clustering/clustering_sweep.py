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

# typing
from typing import List, Dict, Optional

# internal
from .feature_matching import featureMatching
from .params import featureMatchingParameters
from .clustering_sweep_report import (
    ClusteringSweepReporter,
    RefinementReport, ClusterQuality,
)
from .graph import build_graph
from .metrics import UNKNOWN_LABEL, compute_metrics, compute_cluster_purity, compute_label_completeness

# HOG
from src.patch_processing.configs import get_hog_cfg
from src.patch_processing.hog import HOG

# Refinement pipeline
from .refinement import (
    RefinementResult,
    HausdorffSplitStep, PCAZScoreRematchStep,
)

from tqdm import tqdm


class graphClusteringSweep:
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

            cell_sizes: None | List[int] = None,
            normalization_methods: None | List[None | str] = ['patch'],
            grdt_sigmas: None | List[float] = None,
            nums_bins: None | List[int] = None,

            # ── Injected components (instantiated externally, e.g. by Hydra) ──
            reporter: Optional[ClusteringSweepReporter] = None,
            refinement_steps: Optional[list] = None,

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
        if reporter is None:
            raise ValueError("reporter (ClusteringSweepReporter) must be provided")
        if refinement_steps is None:
            raise ValueError("refinement_steps must be provided")

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

        # ── Injected components ──
        self.reporter           = reporter
        self.refinement_steps   = refinement_steps

        # ── Post-clustering refinement (optional) ──
        self.enable_chat_split          = enable_chat_split
        self.chat_split_purity_threshold = chat_split_purity_threshold
        self.chat_split_min_size        = chat_split_min_size
        self.chat_split_min_label_count = chat_split_min_label_count

        self.enable_hapax_association    = enable_hapax_association
        self.hapax_min_confidence        = hapax_min_confidence
        self.hapax_max_dissimilarity     = hapax_max_dissimilarity

        self.enable_glossary             = enable_glossary

    # ================================================================
    #  Evaluation
    # ================================================================

    def _evaluate_membership(self, target_labels, membership, exclude_unknown=True):
        if exclude_unknown:
            mask = pd.Series(target_labels).fillna(UNKNOWN_LABEL) != UNKNOWN_LABEL
            target_labels = target_labels[mask]
            membership = [m for m, keep in zip(membership, mask) if keep]

        return compute_metrics(
            reference_labels=target_labels,
            predicted_labels=membership
        )

    # ================================================================
    #  Graph construction (delegates to clustering.graph)
    # ================================================================

    def get_graph(self, nlfa, dissimilarities, epsilon):
        return build_graph(nlfa, dissimilarities, epsilon,
                           self.edges_type, self.keep_reciprocal)

    # ================================================================
    #  Purity & completeness (delegates to clustering.metrics)
    # ================================================================

    def _compute_purity_and_representatives(self, dataframe, membership_col):
        return compute_cluster_purity(dataframe, membership_col, self.target_lbl)

    def _compute_label_dataframe(self, dataframe):
        return compute_label_completeness(dataframe, self.target_lbl)

    # ================================================================
    #  report_graph sub-steps
    # ================================================================

    def _run_refinement(self, dataframe, pre_split_membership, renderer, graph):
        """Run the refinement pipeline and return post-split membership + logs."""
        current_membership = pre_split_membership.copy()
        results: list[RefinementResult] = []
        step_names: list[str] = []

        has_renderer = renderer is not None
        for step in self.refinement_steps:
            if not has_renderer and isinstance(step, (HausdorffSplitStep, PCAZScoreRematchStep)):
                continue  # skip GPU steps when no renderer available
            result = step.run(
                dataframe, current_membership, renderer,
                target_lbl=self.target_lbl,
                graph=graph,
                evaluate_fn=self._evaluate_membership,
            )
            current_membership = result.membership
            results.append(result)
            step_names.append(step.name)

        return current_membership, results, step_names

    def _reorder_by_membership(self, dataframe, graph, nlfa, dissimilarities,
                               pre_split_membership, post_split_membership):
        """Sort dataframe by membership, realign graph / matrices / arrays."""
        dataframe['_graph_node_id'] = range(len(dataframe))
        dataframe = dataframe.sort_values(
            by=['membership', 'degree_centrality'],
            ascending=[True, False]
        ).reset_index(drop=True)

        perm = dataframe.pop('_graph_node_id').values
        old_to_new = {int(old): new for new, old in enumerate(perm)}
        graph = nx.relabel_nodes(graph, old_to_new)

        perm_t = torch.tensor(perm, device=nlfa.device, dtype=torch.long)
        nlfa = nlfa[perm_t][:, perm_t]
        dissimilarities = dissimilarities[perm_t][:, perm_t]

        pre_split_membership = pre_split_membership[perm]
        post_split_membership = np.asarray(post_split_membership)[perm]

        return (dataframe, graph, nlfa, dissimilarities,
                pre_split_membership, post_split_membership)

    def _run_post_clustering(self, dataframe, dissimilarities):
        """Run optional post-clustering steps (CHAT split, hapax, glossary)."""
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

        return dataframe, chat_split_log, hapax_log, glossary_df

    def _compute_quality(self, dataframe, pre_split_membership,
                         post_split_membership, did_split):
        """Compute purity, metrics, and label completeness."""
        pre_purity_df, pre_representatives = \
            self._compute_purity_and_representatives(dataframe, 'membership_pre_split')

        if did_split:
            purity_dataframe, representatives = \
                self._compute_purity_and_representatives(dataframe, 'membership')
        else:
            purity_dataframe = pre_purity_df
            representatives = pre_representatives

        _to_list = lambda m: m if isinstance(m, list) else m.tolist()
        best_metrics = self._evaluate_membership(
            target_labels=dataframe[self.target_lbl],
            membership=dataframe['membership'].tolist()
        )
        label_dataframe = self._compute_label_dataframe(dataframe)

        pre_split_metrics = self._evaluate_membership(
            target_labels=dataframe[self.target_lbl],
            membership=_to_list(pre_split_membership),
        )
        post_split_metrics = self._evaluate_membership(
            target_labels=dataframe[self.target_lbl],
            membership=_to_list(post_split_membership),
        )

        quality = ClusterQuality(
            purity_dataframe=purity_dataframe,
            representatives=representatives,
            pre_purity_df=pre_purity_df,
            pre_representatives=pre_representatives,
            best_metrics=best_metrics,
            label_dataframe=label_dataframe,
        )
        return quality, pre_split_metrics, post_split_metrics

    # ================================================================
    #  report_graph: orchestrator
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
        post_split_membership, refinement_results, refinement_step_names = \
            self._run_refinement(dataframe, pre_split_membership, renderer, graph)

        dataframe.loc[:, 'membership'] = pd.Series(
            post_split_membership, index=dataframe.index
        )
        did_split = len(refinement_results) > 0

        # ── 5. Reorder by membership ──
        dataframe, graph, nlfa, dissimilarities, \
            pre_split_membership, post_split_membership = \
            self._reorder_by_membership(
                dataframe, graph, nlfa, dissimilarities,
                pre_split_membership, post_split_membership,
            )

        # ── 6. Post-clustering refinement (optional) ──
        dataframe, chat_split_log, hapax_log, glossary_df = \
            self._run_post_clustering(dataframe, dissimilarities)

        # ── 7. Compute quality metrics ──
        quality, pre_split_metrics, post_split_metrics = \
            self._compute_quality(
                dataframe, pre_split_membership,
                post_split_membership, did_split,
            )

        # ── 8. Build refinement report ──
        split_result = next(
            (r for r, n in zip(refinement_results, refinement_step_names)
             if n == 'hausdorff_split'), None
        )
        refinement = RefinementReport(
            pre_split_membership=pre_split_membership,
            post_split_membership=post_split_membership,
            did_split=did_split,
            results=refinement_results,
            step_names=refinement_step_names,
            best_threshold=(split_result.metadata.get('best_threshold')
                            if split_result else None),
            best_split_log=split_result.log if split_result else None,
            sweep_results_df=(split_result.metadata.get('sweep_df')
                              if split_result else None),
            pre_split_metrics=pre_split_metrics,
            post_split_metrics=post_split_metrics,
        )

        # ── 9. Delegate ALL reporting in one call ──
        self.reporter.report_graph_results(
            dataframe=dataframe,
            graph=graph,
            partition=partition,
            nlfa=nlfa,
            dissimilarities=dissimilarities,
            best_epsilon=best_epsilon,
            best_gamma=best_gamma,
            refinement=refinement,
            quality=quality,
            feature_matcher=self.featureMatcher,
            chat_split_log=chat_split_log,
            hapax_log=hapax_log,
            glossary_df=glossary_df,
        )

        # ── 10. Build filtered / representative dataframes ──
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