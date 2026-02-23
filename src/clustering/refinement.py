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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from tqdm import tqdm

from .bin_image_metrics import (
    compute_distance_matrices_batched,
    compute_hausdorff,
    registeredMetric,
)

UNKNOWN_LABEL = '\u25af'  # ▯


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

    Unknown-label patches (``UNKNOWN_LABEL``) are excluded from the
    distance-matrix computation and linkage tree, then reassigned to
    the largest sub-cluster of their original Leiden cluster.
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
        cid, known_idx, unknown_idx, dataframe, reg_metric, renderer,
        batch_size, linkage_method, min_cluster_size, gpu_sem,
    ):
        all_idx = np.concatenate([known_idx, unknown_idx]) \
            if len(unknown_idx) else known_idx
        size = len(all_idx)
        n_known = len(known_idx)

        if n_known < min_cluster_size:
            return cid, dict(
                all_idx=all_idx, known_idx=known_idx,
                unknown_idx=unknown_idx, linkage=None, size=size,
            )

        subdf = dataframe.iloc[known_idx]

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

        return cid, dict(
            all_idx=all_idx, known_idx=known_idx,
            unknown_idx=unknown_idx, linkage=Z, size=size,
        )

    # ── linkage computation (parallel) ──────────────────────────────

    def _compute_linkages(self, dataframe, membership, renderer, target_lbl):
        metrics_dict = {'hausdorff': compute_hausdorff}
        reg_metric = registeredMetric(metrics=metrics_dict, sym=True)

        cluster_ids = np.unique(membership)
        labels = dataframe[target_lbl].fillna(UNKNOWN_LABEL).values

        cluster_info = {}
        for cid in cluster_ids:
            all_pos = np.where(membership == cid)[0]
            known_mask = labels[all_pos] != UNKNOWN_LABEL
            cluster_info[cid] = {
                'known_idx': all_pos[known_mask],
                'unknown_idx': all_pos[~known_mask],
            }

        sorted_cids = sorted(cluster_ids,
                             key=lambda c: len(cluster_info[c]['known_idx']),
                             reverse=True)

        linkages = {}
        gpu_sem = threading.Semaphore(1)

        with ThreadPoolExecutor(max_workers=None) as pool:
            futs = {
                pool.submit(
                    self._compute_one,
                    cid,
                    cluster_info[cid]['known_idx'],
                    cluster_info[cid]['unknown_idx'],
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
            for idx in [info['all_idx']] if len(idx)
        ) + 1
        new_mem = np.full(n_total, -1, dtype=int)
        log = []
        next_id = 0

        for cid, info in linkages.items():
            known_idx = info['known_idx']
            unknown_idx = info['unknown_idx']
            Z = info['linkage']

            if Z is None:
                # Cluster too small to split — assign everything to one id
                new_mem[info['all_idx']] = next_id
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

            # Map each sub-label to a new global id
            sub_id_map = {}
            sub_sizes = []
            for s in unique_subs:
                mask_s = sub_labels == s
                new_mem[known_idx[mask_s]] = next_id
                sub_id_map[s] = next_id
                sub_sizes.append(int(mask_s.sum()))
                next_id += 1

            # Assign unknowns to the largest sub-cluster
            if len(unknown_idx):
                largest_sub = unique_subs[np.argmax(
                    [int((sub_labels == s).sum()) for s in unique_subs]
                )]
                new_mem[unknown_idx] = sub_id_map[largest_sub]

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
            dataframe, membership, renderer, target_lbl,
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
#  2. OCR-based rematching (label-driven, no GPU)
# ════════════════════════════════════════════════════════════════════

class OCRRematchStep(ClusterRefinementStep):
    """
    Stage-1 rematch: merge small / singleton clusters into larger
    clusters when their dominant recognised-character label matches.

    For each cluster of size <= ``max_cluster_size``:
      1. Find the dominant known label in that cluster.
      2. Find the largest cluster whose dominant label is the same.
      3. Reassign all patches in the small cluster to that target.

    Unknowns and clusters without any known label are left untouched.
    """

    name = "ocr_rematch"

    def __init__(self, max_cluster_size: int = 3):
        self.max_cluster_size = max_cluster_size

    def run(self, dataframe, membership, renderer, *,
            target_lbl, **ctx) -> RefinementResult:

        mem = membership.copy()
        labels = dataframe[target_lbl].fillna(UNKNOWN_LABEL).values
        log = []

        # ── Pre-compute per-cluster dominant label & size ───────────
        cluster_ids = np.unique(mem)
        cluster_dom = {}   # cid → dominant known label (or None)
        cluster_size = {}  # cid → total size

        for cid in cluster_ids:
            mask = mem == cid
            cluster_size[cid] = int(mask.sum())
            known = labels[mask]
            known = known[known != UNKNOWN_LABEL]
            if len(known) == 0:
                cluster_dom[cid] = None
            else:
                vals, counts = np.unique(known, return_counts=True)
                cluster_dom[cid] = vals[counts.argmax()]

        # ── Build label → largest cluster mapping ───────────────────
        label_to_largest = {}  # label → (cid, size)
        for cid in cluster_ids:
            dom = cluster_dom[cid]
            if dom is None:
                continue
            sz = cluster_size[cid]
            if dom not in label_to_largest or sz > label_to_largest[dom][1]:
                label_to_largest[dom] = (cid, sz)

        # ── Rematch small clusters ──────────────────────────────────
        for cid in cluster_ids:
            if cluster_size[cid] > self.max_cluster_size:
                continue
            dom = cluster_dom[cid]
            if dom is None:
                continue
            target_cid, target_size = label_to_largest.get(dom, (None, 0))
            if target_cid is None or target_cid == cid:
                continue

            mask = mem == cid
            mem[mask] = target_cid
            log.append(dict(
                action='merge',
                source_cluster=int(cid),
                source_size=cluster_size[cid],
                target_cluster=int(target_cid),
                target_size=target_size,
                label=dom,
            ))

        # ── Re-number contiguously ──────────────────────────────────
        unique_ids = np.unique(mem)
        remap = {old: new for new, old in enumerate(unique_ids)}
        mem = np.array([remap[v] for v in mem])

        return RefinementResult(
            membership=mem,
            log=log,
            metadata=dict(
                n_merged=len(log),
                max_cluster_size=self.max_cluster_size,
            ),
        )


# ════════════════════════════════════════════════════════════════════
#  3. PCA z-score rematching
# ════════════════════════════════════════════════════════════════════

class PCAZScoreRematchStep(ClusterRefinementStep):
    """
    Stage-2 rematch: for each remaining small cluster or singleton,
    test compatibility with candidate clusters via PCA z-score.

    Algorithm for each query cluster Q (size <= ``max_cluster_size``):
      1. Compute Q's mean HOG feature vector.
      2. Find the ``n_candidates`` nearest large clusters by L2
         distance between mean feature vectors.
      3. For each candidate cluster C:
         a. Register the query patch against C's most-central patch
            (the "anchor") using IC registration.
         b. Run PCA on C's registered patch images (first ``pca_k``
            components).
         c. Project the registered query into C's PCA space.
         d. Compute z-score on each of the first ``pca_k`` components.
         e. If max |z| < ``z_max``, Q is compatible with C.
      4. Merge Q into the compatible C with the lowest max |z|.
    """

    name = "pca_zscore_rematch"

    def __init__(
        self,
        max_cluster_size: int = 3,
        pca_k: int = 5,
        z_max: float = 3.0,
        n_candidates: int = 5,
        min_target_size: int = 10,
        batch_size: int = 64,
        device: str = 'cuda',
    ):
        self.max_cluster_size = max_cluster_size
        self.pca_k = pca_k
        self.z_max = z_max
        self.n_candidates = n_candidates
        self.min_target_size = min_target_size
        self.batch_size = batch_size
        self.device = device

    def _register_pair(self, reg_metric, renderer, idx_anchor, idx_query):
        """Register query onto anchor, return warped query image (H,W)."""
        I1 = renderer[idx_anchor].unsqueeze(0).unsqueeze(0).to(self.device)
        I2 = renderer[idx_query].unsqueeze(0).unsqueeze(0).to(self.device)
        try:
            T = reg_metric.ic.run(I1, I2)
            I2w = T.warp(I2)[:, 0]
            mask = T.visibility_mask(I1.shape[2], I1.shape[3], delta=0)
            I2w[~mask] = 0
            return I2w[0].cpu()
        except Exception:
            return renderer[idx_query]

    def _build_cluster_pca(self, reg_metric, renderer, member_indices,
                            anchor_idx, pca_k):
        """Register all cluster members onto anchor, fit PCA, return model."""
        registered = []
        for idx in member_indices:
            if idx == anchor_idx:
                img = renderer[idx]
            else:
                img = self._register_pair(reg_metric, renderer, anchor_idx, idx)
            registered.append(img.flatten().numpy())

        X = np.stack(registered)
        k = min(pca_k, X.shape[0] - 1, X.shape[1])
        if k < 1:
            return None, None, None
        pca = PCA(n_components=k)
        Z = pca.fit_transform(X)
        mean = Z.mean(axis=0)
        std = Z.std(axis=0)
        std[std < 1e-8] = 1.0
        return pca, mean, std

    def run(self, dataframe, membership, renderer, *,
            target_lbl, **ctx) -> RefinementResult:

        mem = membership.copy()
        labels = dataframe[target_lbl].fillna(UNKNOWN_LABEL).values
        log = []

        # Build reg_metric for registration
        metrics_dict = {'hausdorff': compute_hausdorff}
        reg_metric = registeredMetric(metrics=metrics_dict, sym=True)

        # ── Cluster stats ───────────────────────────────────────────
        cluster_ids = np.unique(mem)
        cluster_members = {cid: np.where(mem == cid)[0] for cid in cluster_ids}
        cluster_size = {cid: len(v) for cid, v in cluster_members.items()}

        # Mean HOG features per cluster (for candidate selection)
        histograms = dataframe['histogram'].values
        cluster_mean_feat = {}
        for cid in cluster_ids:
            feats = np.stack([histograms[i] for i in cluster_members[cid]])
            cluster_mean_feat[cid] = feats.mean(axis=0).flatten()

        # Find most-central member per cluster (anchor for registration)
        dc = dataframe['degree_centrality'].values
        cluster_anchor = {}
        for cid in cluster_ids:
            idxs = cluster_members[cid]
            cluster_anchor[cid] = idxs[dc[idxs].argmax()]

        # ── Identify small clusters to rematch ──────────────────────
        small_cids = [cid for cid in cluster_ids
                      if cluster_size[cid] <= self.max_cluster_size]
        large_cids = [cid for cid in cluster_ids
                      if cluster_size[cid] >= self.min_target_size]

        if not large_cids:
            return RefinementResult(membership=mem, log=log,
                                   metadata=dict(n_merged=0))

        large_feats = np.stack([cluster_mean_feat[c] for c in large_cids])

        # Cache PCA models for target clusters (computed lazily)
        pca_cache = {}

        for qcid in tqdm(small_cids, desc="PCA z-score rematch",
                         colour="cyan"):
            q_feat = cluster_mean_feat[qcid].reshape(1, -1)
            dists = np.linalg.norm(large_feats - q_feat, axis=1)
            top_k = np.argsort(dists)[:self.n_candidates]
            candidates = [large_cids[i] for i in top_k]

            q_members = cluster_members[qcid]
            # Use the first member as the query representative
            q_idx = cluster_anchor.get(qcid, q_members[0])

            best_z = np.inf
            best_target = None

            for tcid in candidates:
                # Build or retrieve PCA model for target cluster
                if tcid not in pca_cache:
                    pca_cache[tcid] = self._build_cluster_pca(
                        reg_metric, renderer,
                        cluster_members[tcid],
                        cluster_anchor[tcid],
                        self.pca_k,
                    )
                pca_model, pca_mean, pca_std = pca_cache[tcid]
                if pca_model is None:
                    continue

                # Register query onto target anchor
                anchor = cluster_anchor[tcid]
                q_img = self._register_pair(
                    reg_metric, renderer, anchor, q_idx,
                )
                q_proj = pca_model.transform(
                    q_img.flatten().numpy().reshape(1, -1)
                )[0]
                z_scores = np.abs((q_proj - pca_mean) / pca_std)
                max_z = z_scores.max()

                if max_z < self.z_max and max_z < best_z:
                    best_z = max_z
                    best_target = tcid

            if best_target is not None:
                mem[q_members] = best_target
                log.append(dict(
                    action='pca_merge',
                    source_cluster=int(qcid),
                    source_size=len(q_members),
                    target_cluster=int(best_target),
                    target_size=cluster_size[best_target],
                    max_z_score=float(best_z),
                ))

        # Re-number
        unique_ids = np.unique(mem)
        remap = {old: new for new, old in enumerate(unique_ids)}
        mem = np.array([remap[v] for v in mem])

        return RefinementResult(
            membership=mem,
            log=log,
            metadata=dict(
                n_merged=len(log),
                pca_k=self.pca_k,
                z_max=self.z_max,
                n_candidates=self.n_candidates,
            ),
        )
