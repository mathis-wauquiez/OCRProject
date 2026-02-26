"""
Archived refinement steps — moved from src/clustering/refinement.py.

These steps were part of the post-partition pipeline but are not used
in the current active configuration.  Kept here for reference / reuse.

Classes:
    OCRRematchStep       — merge small clusters by dominant OCR label
    PCAZScoreRematchStep — merge singletons via PCA z-score + IC registration
    HapaxAssociationStep — merge singletons via HOG dissimilarity gate
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sklearn.decomposition import PCA
from tqdm import tqdm

# These imports assume the module lives next to refinement.py.
# Adjust paths if you ever re-activate these steps.
from src.clustering.refinement import (
    ClusterRefinementStep,
    RefinementResult,
)
from src.clustering.bin_image_metrics import (
    compute_distance_matrices_batched,
    compute_hausdorff,
    registeredMetric,
)
from src.clustering.metrics import UNKNOWN_LABEL


# ════════════════════════════════════════════════════════════════════
#  OCR-based rematching (label-driven, no GPU)
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
#  PCA z-score rematching
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


# ════════════════════════════════════════════════════════════════════
#  Hapax-to-cluster association
# ════════════════════════════════════════════════════════════════════

class HapaxAssociationStep(ClusterRefinementStep):
    """
    Merge singleton clusters into larger clusters with matching
    dominant label, using HOG dissimilarity as an acceptance gate.

    For each singleton whose known label has confidence >=
    ``min_confidence``:
      1. Find non-singleton clusters whose dominant label matches.
      2. Pick the candidate with lowest mean HOG dissimilarity.
      3. Accept if that dissimilarity <= threshold (per-cluster
         median intra-cluster dissimilarity, or a fixed cap).
    """

    name = "hapax_association"

    def __init__(
        self,
        min_confidence: float = 0.3,
        max_dissimilarity: Optional[float] = None,
    ):
        self.min_confidence = min_confidence
        self.max_dissimilarity = max_dissimilarity

    def run(self, dataframe, membership, renderer, *,
            target_lbl, **ctx) -> RefinementResult:

        mem = membership.copy()
        labels = dataframe[target_lbl].fillna(UNKNOWN_LABEL).values
        confidences = (dataframe['conf_chat'].values
                       if 'conf_chat' in dataframe.columns
                       else np.ones(len(dataframe)))

        # We need the dissimilarity matrix from context
        dissimilarities = ctx.get('dissimilarities')
        if dissimilarities is None:
            return RefinementResult(
                membership=mem, log=[],
                metadata=dict(n_merged=0, reason='no_dissimilarities'),
            )
        dissim_np = dissimilarities
        if hasattr(dissimilarities, 'cpu'):
            dissim_np = dissimilarities.cpu().numpy()

        log = []

        # Identify singletons and non-singletons
        cluster_sizes = pd.Series(mem).value_counts()
        singleton_cids = set(cluster_sizes[cluster_sizes == 1].index)
        nonsingleton_cids = set(cluster_sizes[cluster_sizes > 1].index)

        # Build dominant label per non-singleton cluster
        cluster_dominant = {}
        for cid in nonsingleton_cids:
            cmask = mem == cid
            clabels = labels[cmask]
            known = clabels[clabels != UNKNOWN_LABEL]
            if len(known) > 0:
                vals, counts = np.unique(known, return_counts=True)
                cluster_dominant[cid] = vals[counts.argmax()]

        label_to_clusters = defaultdict(list)
        for cid, dom in cluster_dominant.items():
            label_to_clusters[dom].append(cid)

        # Adaptive threshold: median intra-cluster dissimilarity
        cluster_median_dissim = {}
        if self.max_dissimilarity is None:
            for cid in nonsingleton_cids:
                cidx = np.where(mem == cid)[0]
                if len(cidx) < 2:
                    continue
                pairwise = dissim_np[np.ix_(cidx, cidx)]
                triu_vals = pairwise[np.triu_indices(len(cidx), k=1)]
                cluster_median_dissim[cid] = float(np.median(triu_vals))

        # Try to merge each singleton
        for cid in list(singleton_cids):
            h_idx = np.where(mem == cid)[0]
            if len(h_idx) != 1:
                continue
            h = h_idx[0]

            h_label = labels[h]
            h_conf = confidences[h]

            if h_label == UNKNOWN_LABEL or h_conf < self.min_confidence:
                log.append(dict(
                    hapax_idx=int(h), label=h_label,
                    target_cluster=None, accepted=False,
                    reason='unknown_or_low_conf',
                ))
                continue

            candidates = label_to_clusters.get(h_label, [])
            if not candidates:
                log.append(dict(
                    hapax_idx=int(h), label=h_label,
                    target_cluster=None, accepted=False,
                    reason='no_candidate_cluster',
                ))
                continue

            # Best candidate by mean dissimilarity
            best_cid = None
            best_mean = np.inf
            for tcid in candidates:
                tidx = np.where(mem == tcid)[0]
                mean_d = float(dissim_np[h, tidx].mean())
                if mean_d < best_mean:
                    best_mean = mean_d
                    best_cid = tcid

            threshold = (self.max_dissimilarity
                         if self.max_dissimilarity is not None
                         else cluster_median_dissim.get(best_cid, np.inf))

            accepted = best_mean <= threshold
            if accepted:
                mem[h] = best_cid

            log.append(dict(
                hapax_idx=int(h), label=h_label,
                target_cluster=int(best_cid) if best_cid is not None else None,
                mean_dissim=float(best_mean),
                threshold=float(threshold),
                accepted=accepted,
                reason='accepted' if accepted else 'dissimilarity_too_high',
            ))

        # Re-number contiguously
        unique_ids = np.unique(mem)
        remap = {old: new for new, old in enumerate(unique_ids)}
        mem = np.array([remap[v] for v in mem])

        return RefinementResult(
            membership=mem,
            log=log,
            metadata=dict(
                n_merged=sum(1 for e in log if e.get('accepted')),
                min_confidence=self.min_confidence,
                max_dissimilarity=self.max_dissimilarity,
            ),
        )
