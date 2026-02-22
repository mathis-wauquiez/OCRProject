"""
Evaluation protocol and ablation framework for character clustering.

Implements:
  - Comprehensive metrics (ARI, NMI, V-measure, per-class F1, noise fraction)
  - Statistical significance testing (bootstrap, Wilcoxon, Cohen's d)
  - Ablation study infrastructure
  - Scalability benchmarking
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from scipy.stats import wilcoxon

from .metrics import compute_metrics


# ===========================================================================
#  Extended evaluation metrics
# ===========================================================================


def compute_per_class_f1(
    reference_labels: np.ndarray,
    predicted_labels: np.ndarray,
    exclude_label: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute precision, recall, and F1 per character class using
    optimal Hungarian matching between predicted clusters and true classes.

    Returns a DataFrame with columns:
        class, count, precision, recall, f1, matched_cluster
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.cluster import contingency_matrix

    ref = np.array(reference_labels)
    pred = np.array(predicted_labels)

    if exclude_label is not None:
        mask = ref != exclude_label
        ref = ref[mask]
        pred = pred[mask]

    C = contingency_matrix(ref, pred)
    # C[i, j] = number of samples in true class i AND predicted cluster j
    # Rows = true classes, Cols = predicted clusters
    true_classes = np.unique(ref)
    pred_clusters = np.unique(pred)

    # Hungarian matching: maximize overlap
    row_idx, col_idx = linear_sum_assignment(-C)

    rows = []
    for i, true_class in enumerate(true_classes):
        class_size = int((ref == true_class).sum())
        if i < len(row_idx):
            matched_col = col_idx[np.where(row_idx == i)[0]]
            if len(matched_col) > 0:
                j = matched_col[0]
                tp = int(C[i, j])
                fp = int(C[:, j].sum() - tp)
                fn = int(C[i, :].sum() - tp)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall / (precision + recall)
                       if (precision + recall) > 0 else 0.0)
                matched = int(pred_clusters[j]) if j < len(pred_clusters) else -1
            else:
                precision = recall = f1 = 0.0
                matched = -1
        else:
            precision = recall = f1 = 0.0
            matched = -1

        rows.append({
            'class': true_class,
            'count': class_size,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'matched_cluster': matched,
        })

    return pd.DataFrame(rows).sort_values('count', ascending=False).reset_index(drop=True)


def compute_noise_fraction(labels: np.ndarray) -> float:
    """Fraction of points classified as noise (label = -1)."""
    return float((labels == -1).sum()) / len(labels) if len(labels) > 0 else 0.0


# ===========================================================================
#  Statistical significance testing
# ===========================================================================


def bootstrap_metrics(
    reference_labels: np.ndarray,
    predicted_labels: np.ndarray,
    n_bootstrap: int = 5,
    metric_fn: Optional[Callable] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap evaluation: subsample the evaluation set N times and
    report mean ± std of all metrics.

    Parameters
    ----------
    reference_labels : array-like
        Ground truth labels.
    predicted_labels : array-like
        Predicted cluster labels.
    n_bootstrap : int
        Number of bootstrap samples.
    metric_fn : callable or None
        Function(ref, pred) → dict of metrics.  Defaults to ``compute_metrics``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict : {metric_name: {'mean': float, 'std': float, 'values': list}}
    """
    rng = np.random.RandomState(seed)
    ref = np.array(reference_labels)
    pred = np.array(predicted_labels)
    N = len(ref)

    if metric_fn is None:
        metric_fn = compute_metrics

    all_metrics = []
    for _ in range(n_bootstrap):
        indices = rng.choice(N, size=N, replace=True)
        m = metric_fn(ref[indices], pred[indices])
        all_metrics.append(m)

    # Aggregate
    result = {}
    keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
    for key in keys:
        values = [m[key] for m in all_metrics if key in m]
        result[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values,
        }
    return result


def paired_significance_test(
    ref_labels: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_bootstrap: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Paired Wilcoxon signed-rank test and Cohen's d for comparing two methods.

    Parameters
    ----------
    ref_labels : array-like
        Ground truth.
    pred_a, pred_b : array-like
        Predictions from method A and B.
    n_bootstrap : int
        Bootstrap samples for paired comparison.

    Returns
    -------
    dict with keys:
        ``ari_a``, ``ari_b`` : bootstrap stats for each method
        ``wilcoxon_p`` : p-value from Wilcoxon signed-rank test
        ``cohens_d`` : effect size
    """
    from .metrics import compute_metrics

    rng = np.random.RandomState(seed)
    ref = np.array(ref_labels)
    pred_a_ = np.array(pred_a)
    pred_b_ = np.array(pred_b)
    N = len(ref)

    ari_a_vals = []
    ari_b_vals = []
    for _ in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        ma = compute_metrics(ref[idx], pred_a_[idx])
        mb = compute_metrics(ref[idx], pred_b_[idx])
        ari_a_vals.append(ma.get('adjusted_rand_index', 0))
        ari_b_vals.append(mb.get('adjusted_rand_index', 0))

    ari_a = np.array(ari_a_vals)
    ari_b = np.array(ari_b_vals)

    # Wilcoxon test
    try:
        stat, p_value = wilcoxon(ari_a, ari_b)
    except Exception:
        stat, p_value = 0, 1.0

    # Cohen's d
    diff = ari_a - ari_b
    cohens_d = diff.mean() / max(diff.std(), 1e-10)

    return {
        'ari_a': {'mean': float(ari_a.mean()), 'std': float(ari_a.std())},
        'ari_b': {'mean': float(ari_b.mean()), 'std': float(ari_b.std())},
        'wilcoxon_statistic': float(stat),
        'wilcoxon_p_value': float(p_value),
        'cohens_d': float(cohens_d),
    }


# ===========================================================================
#  Ablation study framework
# ===========================================================================


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str
    parameter_name: str
    parameter_values: list
    default_value: Any = None


@dataclass
class AblationResult:
    """Result of a single ablation run."""
    config_name: str
    parameter_name: str
    parameter_value: Any
    metrics: Dict[str, float]
    runtime_seconds: float = 0.0
    n_clusters: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class AblationRunner:
    """
    Run ablation studies that isolate one variable while holding
    all others at their default/best setting.

    Usage:
        runner = AblationRunner()
        runner.add_ablation(AblationConfig(...))
        results = runner.run(pipeline_fn, data, default_params)
    """

    def __init__(self):
        self.ablations: List[AblationConfig] = []
        self.results: List[AblationResult] = []

    def add_ablation(self, config: AblationConfig):
        self.ablations.append(config)

    def run(
        self,
        pipeline_fn: Callable,
        data: Any,
        default_params: Dict[str, Any],
        reference_labels: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Run all registered ablation studies.

        Parameters
        ----------
        pipeline_fn : callable
            ``pipeline_fn(data, **params) -> labels``
        data : Any
            Input data (e.g., dataframe, NLFA matrix).
        default_params : dict
            Default parameter values.
        reference_labels : ndarray or None
            Ground truth for metrics.

        Returns
        -------
        DataFrame with all results.
        """
        self.results = []

        for ablation in self.ablations:
            for value in ablation.parameter_values:
                params = default_params.copy()
                params[ablation.parameter_name] = value

                t0 = time.time()
                try:
                    labels = pipeline_fn(data, **params)
                except Exception as e:
                    self.results.append(AblationResult(
                        config_name=ablation.name,
                        parameter_name=ablation.parameter_name,
                        parameter_value=value,
                        metrics={'error': str(e)},
                    ))
                    continue
                runtime = time.time() - t0

                metrics = {}
                n_clusters = len(np.unique(labels[labels >= 0])) if labels is not None else 0

                if reference_labels is not None and labels is not None:
                    metrics = compute_metrics(reference_labels, labels)

                self.results.append(AblationResult(
                    config_name=ablation.name,
                    parameter_name=ablation.parameter_name,
                    parameter_value=value,
                    metrics=metrics,
                    runtime_seconds=runtime,
                    n_clusters=n_clusters,
                ))

        return self.results_to_dataframe()

    def results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to a flat DataFrame for plotting."""
        rows = []
        for r in self.results:
            row = {
                'ablation': r.config_name,
                'parameter': r.parameter_name,
                'value': r.parameter_value,
                'runtime_s': r.runtime_seconds,
                'n_clusters': r.n_clusters,
            }
            for k, v in r.metrics.items():
                if isinstance(v, (int, float)):
                    row[k] = v
            rows.append(row)
        return pd.DataFrame(rows)


# ===========================================================================
#  Predefined ablation configurations (Section 9 of the spec)
# ===========================================================================

def get_feature_ablations() -> List[AblationConfig]:
    """Section 9.1: Feature representation ablations."""
    return [
        AblationConfig(
            name='A1_hog_cell_size',
            description='HOG cell size sweep',
            parameter_name='cell_size',
            parameter_values=[8, 12, 16, 24],
            default_value=24,
        ),
        AblationConfig(
            name='A2_orientation_bins',
            description='Number of HOG orientation bins',
            parameter_name='num_bins',
            parameter_values=[4, 8, 12, 16, 24],
            default_value=16,
        ),
        AblationConfig(
            name='A4_ot_vs_cemd',
            description='OT vs CEMD dissimilarity metric',
            parameter_name='metric',
            parameter_values=['CEMD', 'L2'],
            default_value='CEMD',
        ),
    ]


def get_acontrario_ablations() -> List[AblationConfig]:
    """Section 9.2: A contrario framework ablations."""
    return [
        AblationConfig(
            name='A7_nfa_threshold',
            description='NFA threshold epsilon sweep',
            parameter_name='epsilon',
            parameter_values=[0.01, 0.1, 1.0, 10.0, 100.0],
            default_value=0.005,
        ),
    ]


def get_clustering_ablations() -> List[AblationConfig]:
    """Section 9.3: Clustering method ablations."""
    return [
        AblationConfig(
            name='A9_hdbscan_min_cluster_size',
            description='HDBSCAN min_cluster_size sweep',
            parameter_name='min_cluster_size',
            parameter_values=[2, 3, 5, 10, 20],
            default_value=3,
        ),
        AblationConfig(
            name='A10_hdbscan_min_samples',
            description='HDBSCAN min_samples sweep',
            parameter_name='min_samples',
            parameter_values=[1, 3, 5, 10, 15],
            default_value=3,
        ),
    ]


def get_refinement_ablations() -> List[AblationConfig]:
    """Section 9.4: Refinement ablations."""
    return [
        AblationConfig(
            name='A12_mrf_beta',
            description='MRF coupling weight beta sweep',
            parameter_name='mrf_beta',
            parameter_values=[0.1, 0.5, 1.0, 2.0, 5.0],
            default_value=1.0,
        ),
    ]


# ===========================================================================
#  Scalability benchmarking
# ===========================================================================


@dataclass
class ScalabilityResult:
    """Result of a single scalability benchmark run."""
    n_characters: int
    stage: str
    wall_clock_seconds: float
    memory_mb: float = 0.0


def run_scalability_benchmark(
    pipeline_stages: Dict[str, Callable],
    data_subsets: Dict[int, Any],
) -> pd.DataFrame:
    """
    Run the pipeline on subsets of increasing size and measure
    wall-clock time and memory per stage.

    Parameters
    ----------
    pipeline_stages : dict {stage_name: callable(data) -> result}
    data_subsets : dict {n_characters: data_subset}

    Returns
    -------
    DataFrame with columns: n_characters, stage, wall_clock_seconds, memory_mb
    """
    import tracemalloc

    results = []
    for n, data in sorted(data_subsets.items()):
        for stage_name, stage_fn in pipeline_stages.items():
            tracemalloc.start()
            t0 = time.time()
            try:
                stage_fn(data)
            except Exception:
                pass
            elapsed = time.time() - t0
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            results.append(ScalabilityResult(
                n_characters=n,
                stage=stage_name,
                wall_clock_seconds=elapsed,
                memory_mb=peak / (1024 * 1024),
            ))

    rows = [
        {'n_characters': r.n_characters, 'stage': r.stage,
         'wall_clock_seconds': r.wall_clock_seconds, 'memory_mb': r.memory_mb}
        for r in results
    ]
    return pd.DataFrame(rows)
