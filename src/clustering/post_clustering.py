"""
Post-clustering reporting utilities:
  build_glossary — produce a character inventory sorted by count
"""

import numpy as np
import pandas as pd

from .metrics import UNKNOWN_LABEL


# ================================================================
#  Glossary
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
