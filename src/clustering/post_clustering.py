"""
Post-clustering reporting utilities:
  compute_representative — pick most representative patch in a sub-dataframe
  build_glossary — produce a per-character inventory sorted by frequency
"""

import numpy as np
import pandas as pd

from .metrics import UNKNOWN_LABEL


def compute_representative(subdf):
    """Return the dataframe index of the most representative patch.

    Uses degree_centrality by default.  Change the column name below to
    switch criterion (betweenness_centrality, closeness_centrality,
    eigenvector_centrality, …).
    """
    return int(subdf['degree_centrality'].idxmax())


# ================================================================
#  Glossary
# ================================================================

def build_glossary(dataframe, cluster_total_nfa=None):
    """Build a per-character glossary from refined clusters.

    For every OCR-predicted character (``char_chat``), find the cluster
    containing the most occurrences of that character and pick the most
    representative patch from that cluster.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must have ``membership``, ``char_chat``, ``degree_centrality``.
    cluster_total_nfa : dict[int, float] | None
        ``{cluster_id: total_nfa}`` — tiebreaker when two clusters have
        equal count of a character (higher is better).

    Returns
    -------
    glossary_df : pd.DataFrame
        One row per known character, sorted by ``n`` descending.
    """
    labels = dataframe['char_chat'].fillna(UNKNOWN_LABEL)
    known_mask = labels != UNKNOWN_LABEL

    if cluster_total_nfa is None:
        cluster_total_nfa = {}

    rows = []
    for char in labels[known_mask].unique():
        char_mask = labels == char
        n = int(char_mask.sum())

        # Per-cluster counts of this character
        memberships = dataframe.loc[char_mask, 'membership']
        cluster_counts = memberships.value_counts()

        # Pick the biggest cluster; tiebreak by total NFA (descending)
        best_count = cluster_counts.iloc[0]
        tied = cluster_counts[cluster_counts == best_count]
        if len(tied) > 1:
            best_cid = int(max(tied.index,
                               key=lambda c: cluster_total_nfa.get(int(c), 0)))
        else:
            best_cid = int(tied.index[0])

        n_c = int((dataframe['membership'] == best_cid).sum())
        n_chars_c = int(best_count)

        rep_idx = compute_representative(dataframe[dataframe['membership'] == best_cid])

        rows.append({
            'character': char,
            'n': n,
            'n_chars_c': n_chars_c,
            'n_c': n_c,
            'cluster_id': best_cid,
            'representative_idx': rep_idx,
        })

    glossary_df = (pd.DataFrame(rows)
                   .sort_values('n', ascending=False)
                   .reset_index(drop=True))
    return glossary_df
