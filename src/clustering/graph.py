"""Graph construction from NLFA / dissimilarity matrices."""

import networkx as nx
import numpy as np
import torch


def build_graph(nlfa, dissimilarities, epsilon, edges_type, keep_reciprocal=True):
    """Threshold an NLFA matrix and return a weighted NetworkX graph.

    Parameters
    ----------
    nlfa : Tensor
        Square (N, N) a-contrario NFA matrix.
    dissimilarities : Tensor
        Square (N, N) raw dissimilarity matrix (same shape as *nlfa*).
    epsilon : float
        Threshold on the linkage matrix.  For ``edges_type='nlfa'`` it is
        transformed into NFA space automatically.
    edges_type : str
        ``'nlfa'``, ``'dissim'``/``'dissimilarities'``, or
        ``'link'``/``'constant'``.  Controls which matrix supplies edge
        weights.
    keep_reciprocal : bool
        If *True*, keep only edges where both directions pass the threshold
        and symmetrise the weight matrix.

    Returns
    -------
    G : nx.Graph
        Weighted graph with nodes ``0 .. N-1``.
    edges : Tensor
        (E, 2) tensor of directed edges that passed the threshold.
    """
    # ── choose weight matrix ────────────────────────────────────────
    if edges_type == 'nlfa':
        weight_matrix = nlfa
    elif edges_type in ('dissim', 'dissimilarities'):
        weight_matrix = dissimilarities
    elif edges_type in ('link', 'constant'):
        weight_matrix = torch.ones_like(nlfa)
    else:
        raise ValueError(f"Unknown edges_type={edges_type!r}")

    # ── transform threshold into NFA space if needed ────────────────
    N = len(nlfa)
    threshold = -(np.log(epsilon) - 2 * np.log(N)) if edges_type == 'nlfa' else epsilon

    # ── apply threshold ─────────────────────────────────────────────
    connected = nlfa >= threshold
    if keep_reciprocal:
        connected &= nlfa.T >= threshold
        weight_matrix = .5 * (weight_matrix + weight_matrix.T)

    edges = torch.nonzero(connected, as_tuple=False)
    edges = edges[edges[:, 0] != edges[:, 1]]

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_weighted_edges_from(
        [(int(i.item()), int(j.item()), weight_matrix[i, j].item())
         for i, j in edges]
    )
    return G, edges
