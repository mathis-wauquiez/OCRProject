"""
Ablation framework, significance testing, and scalability benchmarking.

Metrics live in metrics.py — this module provides the experiment
infrastructure around them.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from scipy.stats import wilcoxon

from .metrics import compute_metrics


# ===========================================================================
#  Statistical significance
# ===========================================================================

def bootstrap_metrics(ref, pred, n_bootstrap=5, seed=42):
    """Bootstrap mean +/- std of all metrics."""
    rng = np.random.RandomState(seed)
    ref, pred = np.asarray(ref), np.asarray(pred)
    N = len(ref)
    all_m = [compute_metrics(ref[rng.choice(N, N, replace=True)],
                             pred[rng.choice(N, N, replace=True)])
             for _ in range(n_bootstrap)]
    keys = [k for k in all_m[0] if isinstance(all_m[0][k], (int, float))]
    return {k: {'mean': float(np.mean(v := [m[k] for m in all_m])),
                'std': float(np.std(v))} for k in keys}


def paired_significance_test(ref, pred_a, pred_b, n_bootstrap=5, seed=42):
    """Paired Wilcoxon + Cohen's d comparing two methods on ARI."""
    rng = np.random.RandomState(seed)
    ref, a, b = np.asarray(ref), np.asarray(pred_a), np.asarray(pred_b)
    N = len(ref)
    ari_a, ari_b = [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(N, N, replace=True)
        ari_a.append(compute_metrics(ref[idx], a[idx]).get('adjusted_rand_index', 0))
        ari_b.append(compute_metrics(ref[idx], b[idx]).get('adjusted_rand_index', 0))
    aa, ab = np.array(ari_a), np.array(ari_b)
    try:
        stat, p = wilcoxon(aa, ab)
    except Exception:
        stat, p = 0, 1.0
    diff = aa - ab
    return {'ari_a_mean': float(aa.mean()), 'ari_b_mean': float(ab.mean()),
            'wilcoxon_p': float(p), 'cohens_d': float(diff.mean() / max(diff.std(), 1e-10))}


# ===========================================================================
#  Ablation framework
# ===========================================================================

@dataclass
class AblationConfig:
    name: str
    description: str
    parameter_name: str
    parameter_values: list
    default_value: Any = None


class AblationRunner:
    """Run ablation studies: vary one parameter, hold others at defaults."""

    def __init__(self):
        self.ablations: List[AblationConfig] = []

    def add_ablation(self, config: AblationConfig):
        self.ablations.append(config)

    def run(self, pipeline_fn, data, default_params, reference_labels=None):
        """pipeline_fn(data, **params) → labels.  Returns DataFrame."""
        rows = []
        for abl in self.ablations:
            for val in abl.parameter_values:
                params = {**default_params, abl.parameter_name: val}
                t0 = time.time()
                try:
                    labels = pipeline_fn(data, **params)
                except Exception as e:
                    rows.append({'ablation': abl.name, 'parameter': abl.parameter_name,
                                 'value': val, 'error': str(e)})
                    continue
                elapsed = time.time() - t0
                metrics = compute_metrics(reference_labels, labels) if reference_labels is not None else {}
                row = {'ablation': abl.name, 'parameter': abl.parameter_name,
                       'value': val, 'runtime_s': elapsed,
                       'n_clusters': len(np.unique(labels[labels >= 0])) if labels is not None else 0}
                row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                rows.append(row)
        return pd.DataFrame(rows)


# Predefined ablation configs (Section 9 of spec)

def get_feature_ablations():
    return [
        AblationConfig('A1_cell_size', 'HOG cell size', 'cell_size', [8, 12, 16, 24], 24),
        AblationConfig('A2_bins', 'Orientation bins', 'num_bins', [4, 8, 12, 16, 24], 16),
        AblationConfig('A4_metric', 'Dissimilarity metric', 'metric', ['CEMD', 'L2'], 'CEMD'),
    ]

def get_acontrario_ablations():
    return [AblationConfig('A7_epsilon', 'NFA threshold', 'epsilon', [0.01, 0.1, 1.0, 10.0, 100.0], 0.005)]

def get_clustering_ablations():
    return [
        AblationConfig('A9_min_cluster', 'HDBSCAN min_cluster_size', 'min_cluster_size', [2, 3, 5, 10, 20], 3),
        AblationConfig('A10_min_samples', 'HDBSCAN min_samples', 'min_samples', [1, 3, 5, 10, 15], 3),
    ]

def get_refinement_ablations():
    return [AblationConfig('A12_mrf_beta', 'MRF beta', 'mrf_beta', [0.1, 0.5, 1.0, 2.0, 5.0], 1.0)]


# ===========================================================================
#  Scalability benchmark
# ===========================================================================

def run_scalability_benchmark(pipeline_stages, data_subsets):
    """Wall-clock + memory per stage at increasing N. Returns DataFrame."""
    import tracemalloc
    rows = []
    for n, data in sorted(data_subsets.items()):
        for name, fn in pipeline_stages.items():
            tracemalloc.start()
            t0 = time.time()
            try:
                fn(data)
            except Exception:
                pass
            elapsed = time.time() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append({'n_characters': n, 'stage': name,
                         'wall_clock_s': elapsed, 'memory_mb': peak / 1024 / 1024})
    return pd.DataFrame(rows)
