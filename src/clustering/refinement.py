"""
Cluster refinement pipeline — composable post-partition operations.

Every refinement step (split, merge, rematch) inherits from
``ClusterRefinementStep`` and implements ``run()``.  Steps are
chained sequentially in ``graphClusteringSweep.report_graph``:
the output membership of step N feeds into step N+1.

Each step returns a ``RefinementResult`` that carries the new
membership array plus structured diagnostics for reporting.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm import tqdm

from .bin_image_metrics import (
    compute_distance_matrices_batched,
    compute_hausdorff,
    registeredMetric,
)
from .metrics import UNKNOWN_LABEL


# ════════════════════════════════════════════════════════════════════
#  Data structures
# ════════════════════════════════════════════════════════════════════

@dataclass
class RefinementResult:
    """Output of a single refinement step."""
    membership: np.ndarray                # new cluster IDs (one per row)
    log: List[Dict[str, Any]]             # per-cluster action log
    metadata: Dict[str, Any] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════
#  Abstract base
# ════════════════════════════════════════════════════════════════════

class ClusterRefinementStep(ABC):
    """One composable step in a post-partition refinement pipeline."""

    name: str = "abstract"

    @abstractmethod
    def run(
        self,
        dataframe: pd.DataFrame,
        membership: np.ndarray,
        renderer,
        *,
        target_lbl: str,
        **ctx,
    ) -> RefinementResult:
        """
        Args:
            dataframe: full DataFrame (columns include *target_lbl*,
                ``'svg'``, ``'histogram'``, ``'degree_centrality'``, …).
            membership: current cluster-ID array (len = len(dataframe)).
            renderer: ``Renderer`` instance (callable Dataset).
            target_lbl: column name of the ground-truth label.
            **ctx: extra context forwarded by the orchestrator (e.g.
                ``graph``, ``features``, ``reg_metric``).

        Returns:
            ``RefinementResult`` with the updated membership.
        """
        ...


# ════════════════════════════════════════════════════════════════════
#  1. Hausdorff-based cluster splitting
# ════════════════════════════════════════════════════════════════════

class HausdorffSplitStep(ClusterRefinementStep):
    """
    Split large clusters using pairwise Hausdorff distances and
    hierarchical linkage.

    **All** patches participate in the distance matrix and linkage tree
    regardless of label status: the dendrogram cut alone decides which
    sub-cluster each patch belongs to.  This avoids blind spots where
    OCR-unknown characters are silently absorbed into the dominant group.
    """

    name = "hausdorff_split"

    def __init__(
        self,
        thresholds: List[float],
        linkage_method: str = 'average',
        min_cluster_size: int = 5,
        batch_size: int = 256,
    ):
        self.thresholds = thresholds
        self.linkage_method = linkage_method
        self.min_cluster_size = min_cluster_size
        self.batch_size = batch_size

    # ── per-cluster worker (static for thread-pool) ─────────────────

    @staticmethod
    def _compute_one(
        cid, indices, dataframe, reg_metric, renderer,
        batch_size, linkage_method, min_cluster_size, gpu_sem,
    ):
        size = len(indices)

        if size < min_cluster_size:
            return cid, dict(indices=indices, linkage=None, size=size)

        subdf = dataframe.iloc[indices]

        if gpu_sem is not None:
            gpu_sem.acquire()
        try:
            D_dict = compute_distance_matrices_batched(
                reg_metric, renderer, subdf,
                batch_size=batch_size,
                sym_registration=False,
            )
        finally:
            if gpu_sem is not None:
                gpu_sem.release()

        D = D_dict['hausdorff']
        condensed = squareform(D, checks=False)
        bad = ~np.isfinite(condensed)
        if bad.any():
            fmax = np.nanmax(condensed[np.isfinite(condensed)]) \
                if np.isfinite(condensed).any() else 1.0
            condensed[bad] = fmax * 10
        Z = linkage(condensed, method=linkage_method)

        return cid, dict(indices=indices, linkage=Z, size=size)

    # ── linkage computation (parallel) ──────────────────────────────

    def _compute_linkages(self, dataframe, membership, renderer):
        metrics_dict = {'hausdorff': compute_hausdorff}
        reg_metric = registeredMetric(metrics=metrics_dict, sym=True)

        cluster_ids = np.unique(membership)
        cluster_indices = {
            cid: np.where(membership == cid)[0] for cid in cluster_ids
        }

        sorted_cids = sorted(cluster_ids,
                             key=lambda c: len(cluster_indices[c]),
                             reverse=True)

        linkages = {}
        gpu_sem = threading.Semaphore(1)

        with ThreadPoolExecutor(max_workers=None) as pool:
            futs = {
                pool.submit(
                    self._compute_one,
                    cid,
                    cluster_indices[cid],
                    dataframe, reg_metric, renderer,
                    self.batch_size, self.linkage_method,
                    self.min_cluster_size, gpu_sem,
                ): cid
                for cid in sorted_cids
            }
            with tqdm(total=len(futs),
                      desc="Hausdorff linkages", colour="yellow") as pbar:
                for fut in as_completed(futs):
                    cid, result = fut.result()
                    linkages[cid] = result
                    pbar.update(1)

        return linkages

    # ── threshold sweep ─────────────────────────────────────────────

    def _apply_threshold(self, linkages, threshold):
        n_total = max(
            idx.max() for info in linkages.values()
            for idx in [info['indices']] if len(idx)
        ) + 1
        new_mem = np.full(n_total, -1, dtype=int)
        log = []
        next_id = 0

        for cid, info in linkages.items():
            indices = info['indices']
            Z = info['linkage']

            if Z is None:
                # Cluster too small to split — assign everything to one id
                new_mem[indices] = next_id
                log.append(dict(
                    original_cluster=int(cid),
                    original_size=info['size'],
                    n_subclusters=1,
                    subcluster_sizes=[info['size']],
                ))
                next_id += 1
                continue

            sub_labels = fcluster(Z, threshold, criterion='distance')
            unique_subs = np.unique(sub_labels)
            n_sub = len(unique_subs)

            sub_sizes = []
            for s in unique_subs:
                mask_s = sub_labels == s
                new_mem[indices[mask_s]] = next_id
                sub_sizes.append(int(mask_s.sum()))
                next_id += 1

            log.append(dict(
                original_cluster=int(cid),
                original_size=info['size'],
                n_subclusters=n_sub,
                subcluster_sizes=sub_sizes,
            ))

        assert (new_mem >= 0).all(), "Unassigned samples after splitting!"
        return new_mem, log

    # ── public interface ────────────────────────────────────────────

    def run(self, dataframe, membership, renderer, *,
            target_lbl, **ctx) -> RefinementResult:

        evaluate_fn = ctx.get('evaluate_fn')

        linkages = self._compute_linkages(
            dataframe, membership, renderer,
        )

        best_score = -np.inf
        best_mem = None
        best_log = None
        best_thresh = self.thresholds[0]
        sweep_rows = []

        for thresh in tqdm(self.thresholds,
                           desc="Split threshold sweep", colour="magenta"):
            mem, log = self._apply_threshold(linkages, thresh)

            if evaluate_fn is not None:
                metrics = evaluate_fn(
                    target_labels=dataframe[target_lbl],
                    membership=mem.tolist(),
                )
            else:
                metrics = {}

            n_clusters = len(np.unique(mem))
            n_split = sum(1 for e in log if e['n_subclusters'] > 1)
            sweep_rows.append({
                'split_threshold': thresh,
                'n_clusters_post_split': n_clusters,
                'n_clusters_actually_split': n_split,
                **metrics,
            })

            score = metrics.get('adjusted_rand_index', 0)
            if score > best_score:
                best_score = score
                best_mem = mem
                best_log = log
                best_thresh = thresh

        sweep_df = pd.DataFrame(sweep_rows)

        return RefinementResult(
            membership=best_mem,
            log=best_log,
            metadata=dict(
                best_threshold=best_thresh,
                sweep_df=sweep_df,
                linkages=linkages,
            ),
        )


# ════════════════════════════════════════════════════════════════════
#  2. Label-based cluster splitting
# ════════════════════════════════════════════════════════════════════

class LabelSplitStep(ClusterRefinementStep):
    """
    Split each cluster by ground-truth label.

    For each cluster, members are grouped by their known label.
    Groups with >= ``min_label_size`` members become their own
    sub-cluster.  Smaller groups are dissolved into individual
    singleton clusters.  Unknowns stay with the dominant sub-cluster
    (or become singletons if no large group exists).
    """

    name = "label_split"

    def __init__(self, min_label_size: int = 2):
        self.min_label_size = min_label_size

    def run(self, dataframe, membership, renderer, *,
            target_lbl, **ctx) -> RefinementResult:

        labels = dataframe[target_lbl].fillna(UNKNOWN_LABEL).values
        new_mem = np.empty_like(membership)
        log = []
        next_id = 0

        for cid in np.unique(membership):
            pos = np.where(membership == cid)[0]
            clabels = labels[pos]
            known_mask = clabels != UNKNOWN_LABEL
            known_pos = pos[known_mask]
            unknown_pos = pos[~known_mask]

            if len(known_pos) == 0:
                # All unknown — keep as one cluster
                new_mem[pos] = next_id
                next_id += 1
                log.append(dict(cluster=int(cid), action='keep',
                                size=len(pos), n_sub=1))
                continue

            # Group known members by label
            ulabels, inv = np.unique(clabels[known_mask], return_inverse=True)
            counts = np.bincount(inv)

            # Dominant label → will receive unknowns
            dominant_new_id = None
            dominant_count = 0

            for i, lbl in enumerate(ulabels):
                lbl_pos = known_pos[inv == i]
                if counts[i] >= self.min_label_size:
                    new_mem[lbl_pos] = next_id
                    if counts[i] > dominant_count:
                        dominant_count = counts[i]
                        dominant_new_id = next_id
                    next_id += 1
                else:
                    # Each member becomes a singleton
                    for p in lbl_pos:
                        new_mem[p] = next_id
                        next_id += 1

            # Assign unknowns
            if dominant_new_id is not None and len(unknown_pos):
                new_mem[unknown_pos] = dominant_new_id
            else:
                for p in unknown_pos:
                    new_mem[p] = next_id
                    next_id += 1

            log.append(dict(
                cluster=int(cid), action='split',
                size=len(pos), n_labels=len(ulabels),
                n_sub=len(np.unique(new_mem[pos])),
            ))

        return RefinementResult(
            membership=new_mem,
            log=log,
            metadata=dict(
                min_label_size=self.min_label_size,
                n_split=sum(1 for e in log if e.get('n_sub', 1) > 1),
            ),
        )
