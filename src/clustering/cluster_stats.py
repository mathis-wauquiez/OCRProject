"""Cluster quality statistics — purity, entropy, completeness."""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm

UNKNOWN_LABEL = '\u25af'  # ▯


def compute_purity(dataframe, membership_col, target_lbl):
    """Per-cluster purity, entropy, and representative patch indices.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must contain *membership_col*, *target_lbl*, and
        ``'degree_centrality'`` columns.
    membership_col : str
        Column holding cluster IDs.
    target_lbl : str
        Column holding ground-truth labels (may contain NaN / unknowns).

    Returns
    -------
    purity_df : pd.DataFrame
        One row per cluster, indexed by cluster ID.
    representatives : dict[int, dict[str, int]]
        ``{cluster_id: {label: most_central_row_index, ...}}``.
    """
    purity_data = []
    representatives = {}

    for cluster, cluster_data in dataframe.groupby(membership_col):
        cluster_size = len(cluster_data)
        known_mask = cluster_data[target_lbl].fillna(UNKNOWN_LABEL) != UNKNOWN_LABEL
        known_data = cluster_data[known_mask]
        unknown_count = (~known_mask).sum()

        label_counts = (
            known_data[target_lbl].value_counts()
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
            label_nodes = cluster_data[cluster_data[target_lbl] == label]
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


def compute_label_completeness(dataframe, target_lbl):
    """Per-label spread across clusters.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must contain ``'membership'`` and *target_lbl* columns.
    target_lbl : str
        Column holding ground-truth labels.

    Returns
    -------
    pd.DataFrame
        One row per label with entropy, best share, dominant cluster, etc.
    """
    label_data = []
    known_df = dataframe[
        dataframe[target_lbl].fillna(UNKNOWN_LABEL) != UNKNOWN_LABEL
    ]
    for label, label_rows in tqdm(known_df.groupby(target_lbl),
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
