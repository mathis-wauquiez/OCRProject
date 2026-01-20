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
import pandas as pd

def compute_metrics(reference_labels, predicted_labels):
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
    
    # Remove any NaN values (keep only valid indices)
    valid_mask = ~(pd.isna(ref) | pd.isna(pred))
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
    
    Considers each pair of samples:
    - TP: same cluster in both reference and prediction
    - FP: same cluster in prediction but different in reference
    - FN: different cluster in prediction but same in reference
    """
    n = len(labels_true)
    
    # Count pairs
    tp = fp = fn = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            same_true = (labels_true[i] == labels_true[j])
            same_pred = (labels_pred[i] == labels_pred[j])
            
            if same_true and same_pred:
                tp += 1
            elif same_pred and not same_true:
                fp += 1
            elif same_true and not same_pred:
                fn += 1
    
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


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


# Optional: Add this for better error handling
def compute_metrics_safe(reference_labels, predicted_labels):
    """
    Safe wrapper that handles errors gracefully.
    """
    try:
        return compute_metrics(reference_labels, predicted_labels)
    except Exception as e:
        return {
            "error": str(e),
            "n_samples": len(reference_labels) if hasattr(reference_labels, '__len__') else 0
        }