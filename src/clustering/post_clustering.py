"""
Post-clustering refinement operations:
  1. chat_split_clusters  — split clusters with mixed CHAT labels
  2. associate_hapax      — merge singletons into matching clusters
  3. build_glossary       — produce a character inventory sorted by count
"""

import numpy as np
import pandas as pd
import torch
from collections import defaultdict

UNKNOWN_LABEL = '\u25af'  # ▯


# ================================================================
#  1. CHAT-based cluster splitting
# ================================================================

def chat_split_clusters(dataframe, dissimilarities=None,
                        purity_threshold=0.90,
                        min_split_size=3,
                        min_label_count=2,
                        target_lbl='char_chat'):
    """Split clusters where CHAT predicts 2+ distinct characters.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must have columns: ``membership``, *target_lbl*.
    dissimilarities : torch.Tensor or None, shape (N, N)
        Pairwise HOG dissimilarity matrix.  Used to assign unknown (▯)
        patches to the nearest sub-cluster.  If *None*, unknowns go to
        the largest sub-cluster.
    purity_threshold : float
        If the most-frequent known label accounts for >= this fraction
        of known labels in the cluster, skip (treat as pure).
    min_split_size : int
        Clusters smaller than this are never split.
    min_label_count : int
        A CHAT label needs >= this many patches to form its own
        sub-cluster; labels below this are folded into the dominant group.
    target_lbl : str
        Column containing CHAT predictions.

    Returns
    -------
    dataframe : pd.DataFrame
        ``membership`` updated; original saved as
        ``membership_pre_chat_split``.
    split_log : list[dict]
        One entry per cluster actually split.
    """
    dataframe = dataframe.copy()
    dataframe['membership_pre_chat_split'] = dataframe['membership'].copy()

    membership = dataframe['membership'].values.copy()
    labels = dataframe[target_lbl].fillna(UNKNOWN_LABEL).values

    split_log = []
    next_id = int(membership.max()) + 1

    for cid in np.unique(membership):
        mask = membership == cid
        idx = np.where(mask)[0]
        if len(idx) < min_split_size:
            continue

        cluster_labels = labels[idx]
        known_mask = cluster_labels != UNKNOWN_LABEL
        known_labels = cluster_labels[known_mask]

        if len(known_labels) == 0:
            continue

        label_counts = pd.Series(known_labels).value_counts()
        unique_labels = label_counts.index.tolist()

        if len(unique_labels) <= 1:
            continue

        # Check purity — skip if dominant label is overwhelming
        purity = label_counts.iloc[0] / len(known_labels)
        if purity >= purity_threshold:
            continue

        # Only keep labels with enough occurrences
        surviving = [l for l in unique_labels if label_counts[l] >= min_label_count]
        if len(surviving) <= 1:
            continue

        dominant = label_counts.index[0]

        # Build sub-clusters for each surviving label
        sub_ids = {}
        sub_sizes = []
        for lbl in surviving:
            sub_ids[lbl] = next_id
            next_id += 1

        # Assign known patches to their sub-cluster
        for i, lbl in zip(idx, cluster_labels):
            if lbl in sub_ids:
                membership[i] = sub_ids[lbl]
            elif lbl != UNKNOWN_LABEL:
                # Rare label → fold into dominant
                membership[i] = sub_ids[dominant]

        # Assign unknowns
        unknown_idx = idx[~known_mask]
        if len(unknown_idx) > 0:
            if dissimilarities is not None:
                _assign_unknowns_by_dissimilarity(
                    membership, unknown_idx, sub_ids, idx, cluster_labels,
                    dissimilarities
                )
            else:
                # Fallback: assign to largest sub-cluster
                largest_lbl = max(surviving, key=lambda l: label_counts[l])
                for i in unknown_idx:
                    membership[i] = sub_ids[largest_lbl]

        split_log.append({
            'original_cluster': int(cid),
            'original_size': int(len(idx)),
            'labels_found': {l: int(label_counts[l]) for l in surviving},
            'sub_sizes': [int((membership[idx] == sub_ids[l]).sum())
                          for l in surviving],
        })

    # Renumber contiguously
    membership = _renumber(membership)
    dataframe['membership'] = membership
    return dataframe, split_log


def _assign_unknowns_by_dissimilarity(membership, unknown_idx, sub_ids,
                                       cluster_idx, cluster_labels,
                                       dissimilarities):
    """Assign unknown patches to the nearest sub-cluster by mean dissimilarity."""
    dissim_np = dissimilarities
    if isinstance(dissimilarities, torch.Tensor):
        dissim_np = dissimilarities.cpu().numpy()

    for u in unknown_idx:
        best_id = None
        best_mean = np.inf
        for lbl, sid in sub_ids.items():
            members = cluster_idx[cluster_labels == lbl]
            if len(members) == 0:
                continue
            mean_d = float(dissim_np[u, members].mean())
            if mean_d < best_mean:
                best_mean = mean_d
                best_id = sid
        if best_id is not None:
            membership[u] = best_id


# ================================================================
#  2. Hapax-to-cluster association
# ================================================================

def associate_hapax(dataframe, dissimilarities,
                    target_lbl='char_chat',
                    min_confidence=0.3,
                    max_dissimilarity=None):
    """Merge singleton clusters into matching non-singleton clusters.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must have ``membership``, *target_lbl*, ``conf_chat``.
    dissimilarities : torch.Tensor, shape (N, N)
        Pairwise HOG dissimilarity matrix.
    target_lbl : str
        Column containing CHAT predictions.
    min_confidence : float
        Ignore hapax whose CHAT confidence is below this.
    max_dissimilarity : float or None
        Hard cap on mean dissimilarity for acceptance.
        *None* → use per-target-cluster median intra-cluster dissimilarity.

    Returns
    -------
    dataframe : pd.DataFrame
        ``membership`` updated; original saved as ``membership_pre_hapax``.
    association_log : list[dict]
        Per-hapax entry with match details.
    """
    dataframe = dataframe.copy()
    dataframe['membership_pre_hapax'] = dataframe['membership'].copy()

    membership = dataframe['membership'].values.copy()
    labels = dataframe[target_lbl].fillna(UNKNOWN_LABEL).values
    confidences = dataframe['conf_chat'].values if 'conf_chat' in dataframe else np.ones(len(dataframe))

    dissim_np = dissimilarities
    if isinstance(dissimilarities, torch.Tensor):
        dissim_np = dissimilarities.cpu().numpy()

    # Identify singletons and non-singletons
    cluster_sizes = pd.Series(membership).value_counts()
    singleton_clusters = set(cluster_sizes[cluster_sizes == 1].index)
    nonsingleton_clusters = set(cluster_sizes[cluster_sizes > 1].index)

    # Build label → list of non-singleton cluster IDs (and their dominant label)
    cluster_dominant_label = {}
    for cid in nonsingleton_clusters:
        cmask = membership == cid
        clabels = labels[cmask]
        known = clabels[clabels != UNKNOWN_LABEL]
        if len(known) > 0:
            cluster_dominant_label[cid] = pd.Series(known).value_counts().index[0]

    label_to_clusters = defaultdict(list)
    for cid, dom in cluster_dominant_label.items():
        label_to_clusters[dom].append(cid)

    # Precompute median intra-cluster dissimilarity for adaptive threshold
    cluster_median_dissim = {}
    if max_dissimilarity is None:
        for cid in nonsingleton_clusters:
            cidx = np.where(membership == cid)[0]
            if len(cidx) < 2:
                continue
            pairwise = dissim_np[np.ix_(cidx, cidx)]
            # Upper triangle only (exclude diagonal)
            triu_vals = pairwise[np.triu_indices(len(cidx), k=1)]
            cluster_median_dissim[cid] = float(np.median(triu_vals))

    association_log = []

    for cid in list(singleton_clusters):
        h_idx = np.where(membership == cid)[0]
        if len(h_idx) != 1:
            continue
        h = h_idx[0]

        h_label = labels[h]
        h_conf = confidences[h]

        # Skip unknowns and low-confidence
        if h_label == UNKNOWN_LABEL or h_conf < min_confidence:
            association_log.append({
                'hapax_idx': int(h), 'char_chat': h_label,
                'target_cluster': None, 'mean_dissim': None,
                'accepted': False, 'reason': 'unknown_or_low_conf',
            })
            continue

        candidates = label_to_clusters.get(h_label, [])
        if not candidates:
            association_log.append({
                'hapax_idx': int(h), 'char_chat': h_label,
                'target_cluster': None, 'mean_dissim': None,
                'accepted': False, 'reason': 'no_candidate_cluster',
            })
            continue

        # Find best candidate by mean dissimilarity
        best_cid = None
        best_mean = np.inf
        for target_cid in candidates:
            target_idx = np.where(membership == target_cid)[0]
            mean_d = float(dissim_np[h, target_idx].mean())
            if mean_d < best_mean:
                best_mean = mean_d
                best_cid = target_cid

        # Acceptance test
        if max_dissimilarity is not None:
            threshold = max_dissimilarity
        else:
            threshold = cluster_median_dissim.get(best_cid, np.inf)

        accepted = best_mean <= threshold
        if accepted:
            membership[h] = best_cid

        association_log.append({
            'hapax_idx': int(h), 'char_chat': h_label,
            'target_cluster': int(best_cid) if best_cid is not None else None,
            'mean_dissim': float(best_mean),
            'threshold': float(threshold),
            'accepted': accepted,
            'reason': 'accepted' if accepted else 'dissimilarity_too_high',
        })

    # Renumber contiguously
    membership = _renumber(membership)
    dataframe['membership'] = membership
    return dataframe, association_log


# ================================================================
#  3. Glossary
# ================================================================

def build_glossary(dataframe, target_lbl='char_chat'):
    """Build a character glossary from refined clusters.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must have ``membership``, *target_lbl*, ``conf_chat``,
        ``degree_centrality``, ``page``.

    Returns
    -------
    glossary_df : pd.DataFrame
        One row per cluster, sorted by ``count`` descending.
    """
    labels = dataframe[target_lbl].fillna(UNKNOWN_LABEL)
    rows = []

    for cid, grp in dataframe.groupby('membership'):
        clabels = labels.loc[grp.index]
        known = clabels[clabels != UNKNOWN_LABEL]

        if len(known) > 0:
            counts = known.value_counts()
            character = counts.index[0]
            purity = counts.iloc[0] / len(known)
        else:
            character = UNKNOWN_LABEL
            purity = np.nan

        mean_conf = np.nan
        if 'conf_chat' in grp.columns:
            known_conf = grp.loc[known.index, 'conf_chat'] if len(known) > 0 else pd.Series(dtype=float)
            if len(known_conf) > 0:
                mean_conf = float(known_conf.mean())

        rep_idx = None
        if 'degree_centrality' in grp.columns:
            rep_idx = int(grp['degree_centrality'].idxmax())

        pages = sorted(grp['page'].unique().tolist()) if 'page' in grp.columns else []

        rows.append({
            'character': character,
            'count': len(grp),
            'cluster_id': int(cid),
            'purity': purity,
            'mean_confidence': mean_conf,
            'representative_idx': rep_idx,
            'pages': pages,
            'n_unknown': int((clabels == UNKNOWN_LABEL).sum()),
        })

    glossary_df = pd.DataFrame(rows).sort_values('count', ascending=False).reset_index(drop=True)
    return glossary_df


# ================================================================
#  Helpers
# ================================================================

def _renumber(membership):
    """Renumber membership IDs to be contiguous 0..K-1."""
    unique_ids = np.unique(membership)
    mapping = {old: new for new, old in enumerate(unique_ids)}
    return np.array([mapping[m] for m in membership])
