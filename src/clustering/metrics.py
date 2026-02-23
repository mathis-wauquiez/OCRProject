import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
import pandas as pd
from tqdm import tqdm


UNKNOWN_LABEL = '▯'  # U+25AF - represents unrecognized characters


def compute_metrics(reference_labels, predicted_labels, exclude_label=None):
    """
    Compute classical clustering evaluation metrics.
    
    Args:
        reference_labels: Ground truth cluster labels (array-like)
        predicted_labels: Predicted cluster labels (array-like)
    
    Returns:
        dict: Dictionary containing all clustering metrics
    """
    # Convert to numpy arrays
    ref = np.array(reference_labels)
    pred = np.array(predicted_labels)
    
    # Remove any NaN values
    valid_mask = ~(pd.isna(ref) | pd.isna(pred))
    
    # Exclude specified label if provided
    if exclude_label is not None:
        exclude_mask = ref != exclude_label
        valid_mask = valid_mask & exclude_mask
    ref = ref[valid_mask]
    pred = pred[valid_mask]
    
    if len(ref) == 0:
        return {
            "error": "No valid samples to compute metrics",
            "n_samples": 0
        }
    
    metrics = {}
    
    # Basic statistics
    metrics['n_samples'] = len(ref)
    metrics['n_clusters_reference'] = len(np.unique(ref))
    metrics['n_clusters_predicted'] = len(np.unique(pred))
    
    # === EXTERNAL METRICS (require ground truth) ===
    
    # Adjusted Rand Index (ARI)
    # Range: [-1, 1], higher is better, 1 = perfect match
    # Adjusted for chance (random labeling gives ~0)
    metrics['adjusted_rand_index'] = float(adjusted_rand_score(ref, pred))
    
    # Normalized Mutual Information (NMI)
    # Range: [0, 1], higher is better, 1 = perfect match
    # Measures shared information between clusterings
    metrics['normalized_mutual_info'] = float(normalized_mutual_info_score(
        ref, pred, average_method='arithmetic'
    ))
    
    # Adjusted Mutual Information (AMI)
    # Range: [-1, 1], higher is better, adjusted for chance
    metrics['adjusted_mutual_info'] = float(adjusted_mutual_info_score(
        ref, pred, average_method='arithmetic'
    ))
    
    # Fowlkes-Mallows Index (FMI)
    # Range: [0, 1], higher is better
    # Geometric mean of precision and recall for pairs
    metrics['fowlkes_mallows_index'] = float(fowlkes_mallows_score(ref, pred))
    
    # Homogeneity, Completeness, V-measure
    # Range: [0, 1], higher is better
    # Homogeneity: each cluster contains only members of a single class
    # Completeness: all members of a class are in the same cluster
    # V-measure: harmonic mean of homogeneity and completeness
    metrics['homogeneity'] = float(homogeneity_score(ref, pred))
    metrics['completeness'] = float(completeness_score(ref, pred))
    metrics['v_measure'] = float(v_measure_score(ref, pred))
    
    # Purity (custom implementation)
    metrics['purity'] = float(compute_purity(ref, pred))
    
    # Inverse Purity
    metrics['inverse_purity'] = float(compute_purity(pred, ref))
    
    # F1 Score (for clustering)
    metrics['f1_score'] = float(compute_clustering_f1(ref, pred))
    
    # Accuracy with optimal matching (Hungarian algorithm)
    metrics['accuracy_optimal_match'] = float(compute_accuracy_hungarian(ref, pred))
    
    return metrics


def compute_purity(labels_true, labels_pred):
    """
    Compute purity score.
    
    Purity = (1/N) * sum_k max_j |cluster_k ∩ class_j|
    
    Range: [0, 1], higher is better
    """
    contingency = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.max(contingency, axis=0)) / np.sum(contingency)


def compute_clustering_f1(labels_true, labels_pred):
    """
    Compute F1 score for clustering using pairwise comparisons.

    Uses the contingency matrix for O(n_classes * n_clusters) computation
    instead of O(n^2) pairwise enumeration.

    Considers each pair of samples:
    - TP: same cluster in both reference and prediction
    - FP: same cluster in prediction but different in reference
    - FN: different cluster in prediction but same in reference
    """
    C = contingency_matrix(labels_true, labels_pred)

    # TP = sum of C(i,j) choose 2 for all cells
    tp = (C * (C - 1) // 2).sum()

    # Total pairs in each predicted cluster
    col_sums = C.sum(axis=0)
    tp_fp = (col_sums * (col_sums - 1) // 2).sum()

    # Total pairs in each true class
    row_sums = C.sum(axis=1)
    tp_fn = (row_sums * (row_sums - 1) // 2).sum()

    fp = tp_fp - tp
    fn = tp_fn - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def compute_accuracy_hungarian(labels_true, labels_pred):
    """
    Compute clustering accuracy using Hungarian algorithm for optimal matching.
    
    This finds the best one-to-one mapping between predicted and true clusters.
    """
    # Build contingency matrix
    contingency = contingency_matrix(labels_true, labels_pred)
    
    # Use Hungarian algorithm to find optimal assignment
    # We want to maximize, so negate the matrix
    row_ind, col_ind = linear_sum_assignment(-contingency)
    
    # Compute accuracy
    accuracy = contingency[row_ind, col_ind].sum() / contingency.sum()
    return accuracy


def compute_per_class_f1(reference_labels, predicted_labels, exclude_label=None):
    """Per-class precision, recall, F1 via Hungarian matching.

    Returns a DataFrame with columns: class, count, precision, recall, f1.
    """
    ref = np.array(reference_labels)
    pred = np.array(predicted_labels)
    if exclude_label is not None:
        mask = ref != exclude_label
        ref, pred = ref[mask], pred[mask]

    C = contingency_matrix(ref, pred)
    true_classes = np.unique(ref)
    row_idx, col_idx = linear_sum_assignment(-C)

    rows = []
    for i, tc in enumerate(true_classes):
        matched_col = col_idx[np.where(row_idx == i)[0]] if i < len(row_idx) else np.array([])
        if len(matched_col):
            j = matched_col[0]
            tp = int(C[i, j])
            fp, fn = int(C[:, j].sum() - tp), int(C[i, :].sum() - tp)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        else:
            prec = rec = f1 = 0.0
        rows.append({'class': tc, 'count': int((ref == tc).sum()),
                     'precision': prec, 'recall': rec, 'f1': f1})
    return pd.DataFrame(rows).sort_values('count', ascending=False).reset_index(drop=True)


def compute_noise_fraction(labels):
    """Fraction of points classified as noise (label = -1)."""
    labels = np.asarray(labels)
    return float((labels == -1).sum()) / len(labels) if len(labels) else 0.0


def compute_metrics_safe(reference_labels, predicted_labels):
    """Safe wrapper that handles errors gracefully."""
    try:
        return compute_metrics(reference_labels, predicted_labels)
    except Exception as e:
        return {
            "error": str(e),
            "n_samples": len(reference_labels) if hasattr(reference_labels, '__len__') else 0
        }


# ═══════════════════════════════════════════════════════════════════
#  Per-cluster / per-label statistics (formerly cluster_stats.py)
# ═══════════════════════════════════════════════════════════════════

def compute_cluster_purity(dataframe, membership_col, target_lbl):
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