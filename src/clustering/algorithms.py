import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, AffinityPropagation, estimate_bandwidth, MeanShift
import igraph as ig
import leidenalg as la
import community.community_louvain as community_louvain
from networkx.algorithms import community
from functools import partial

def sklearn_wrap(function):
    def wrapped(subdf, subgraph, *args, min_size=2, metric='euclidean', **kwargs):
        if len(subdf) < min_size:
            return np.zeros(len(subdf), dtype=int)
        X = np.stack(subdf['clustering_feature'].values)
        D = pairwise_distances(X, metric=metric)
        return function(D, *args, **kwargs)   
    return wrapped

def sklearn_wrap_features(function):
    """ Wrapper for algorithms that need feature vectors, not distance matrices """
    def wrapped(subdf, subgraph, *args, min_size=2, **kwargs):
        if len(subdf) < min_size:
            return np.zeros(len(subdf), dtype=int)
        X = np.stack(subdf['clustering_feature'].values)
        return function(X, *args, **kwargs)   
    return wrapped


def graph_wrap(function):
    """Wrapper for graph-based clustering algorithms that need the graph structure"""
    
    def wrapped(subdf, subgraph, *args, min_size=2, **kwargs):
        
        if len(subdf) < min_size:
            return np.zeros(len(subdf), dtype=int)
        
        if subgraph is None:
            raise ValueError("Graph-based clustering requires a subgraph")
        
        # Get subgraph for only the nodes in subdf
        nodes = subdf.index.tolist()
        G_sub = subgraph.subgraph(nodes).copy()
        
        # Relabel nodes to 0, 1, 2, ... for consistent indexing
        node_mapping = {node: i for i, node in enumerate(nodes)}
        G_relabeled = nx.relabel_nodes(G_sub, node_mapping)
        
        # Run the graph clustering algorithm
        membership = function(G_relabeled, *args, **kwargs)
        
        return np.array(membership)
    
    return wrapped


# ---- Distance-based clustering methods ----

@sklearn_wrap
def dbscan(D, eps):
    """DBSCAN with precomputed distance matrix"""
    # Class documentation:
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    clustering = DBSCAN(eps=eps, min_samples=2, metric='precomputed').fit(D)
    return clustering.labels_


@sklearn_wrap
def hdbscan(D):
    """HDBSCAN with precomputed distance matrix"""
    # Class documentation:
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
    clustering = HDBSCAN(min_cluster_size=2, min_samples=2, metric='precomputed').fit(D)
    return clustering.labels_


@sklearn_wrap
def optics(D, min_samples=5):
    """OPTICS with precomputed distance matrix"""
    clustering = OPTICS(min_samples=min_samples, metric='precomputed').fit(D)
    return clustering.labels_


@sklearn_wrap
def affinity_propagation(D, damping=0.5, preference=None):
    """Affinity Propagation with precomputed similarity matrix"""
    # AffinityPropagation expects a similarity matrix, not distance
    # Convert distance to similarity: S = -D or S = exp(-D^2)
    S = -D
    
    clustering = AffinityPropagation(
        damping=damping,
        preference=preference,
        affinity='precomputed',
        random_state=42
    ).fit(S)
    return clustering.labels_


# ---- Feature-based methods (cannot use distance matrices) ----

@sklearn_wrap_features
def mean_shift(X, q):
    """Mean Shift - requires feature vectors for bandwidth estimation and shifting"""
    # For a review, look at:
    # https://arxiv.org/pdf/1503.00687

    # Bandwidth estimation
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.estimate_bandwidth.html
    bandwidth = estimate_bandwidth(
        X=X,
        quantile=q,
        n_jobs=-1
    )

    # Class documentation:
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    clustering = MeanShift(bandwidth=bandwidth, max_iter=1000).fit(X=X)    
    return clustering.labels_


# ---- Graph-based core functions (unwrapped) ----
# These can be reused directly by graph_clustering.py class-based wrappers.

def _communities_to_membership(communities, n_nodes):
    """Convert a community iterator to a membership list."""
    membership = [0] * n_nodes
    for comm_id, comm in enumerate(communities):
        for node in comm:
            membership[node] = comm_id
    return membership


def leiden_core(G, gamma):
    """Leiden community detection on a plain NetworkX graph."""
    G_ig = ig.Graph.from_networkx(G)
    partition = la.find_partition(
        G_ig,
        la.RBConfigurationVertexPartition,
        resolution_parameter=gamma,
        n_iterations=-1,
        seed=42
    )
    return partition.membership


def louvain_core(G):
    """Louvain community detection on a plain NetworkX graph."""
    partition = community_louvain.best_partition(G, random_state=42)
    return [partition[i] for i in range(len(G.nodes()))]


def greedy_modularity_core(G):
    """Greedy modularity optimization on a plain NetworkX graph."""
    communities = community.greedy_modularity_communities(G)
    return _communities_to_membership(communities, len(G.nodes()))


def label_propagation_core(G):
    """Label propagation on a plain NetworkX graph."""
    communities = community.label_propagation_communities(G)
    return _communities_to_membership(communities, len(G.nodes()))


def infomap_core(G):
    """Infomap community detection on a plain NetworkX graph."""
    G_ig = ig.Graph.from_networkx(G)
    partition = G_ig.community_infomap()
    return partition.membership


def walktrap_core(G, steps=4):
    """Walktrap community detection on a plain NetworkX graph."""
    G_ig = ig.Graph.from_networkx(G)
    partition = G_ig.community_walktrap(steps=steps).as_clustering()
    return partition.membership


# ---- Graph-based clustering methods (wrapped for sweep API) ----

@graph_wrap
def leiden(G, gamma):
    return leiden_core(G, gamma)

@graph_wrap
def louvain(G):
    return louvain_core(G)

@graph_wrap
def greedy_modularity(G):
    return greedy_modularity_core(G)

@graph_wrap
def label_propagation(G):
    return label_propagation_core(G)

@graph_wrap
def infomap(G):
    return infomap_core(G)

@graph_wrap
def walktrap(G, steps=4):
    return walktrap_core(G, steps=steps)



def get_algorithms(quantiles, db_epsilons, leiden_gammas, optics_min_samples, metric='euclidean', min_size=2):
    
    algorithms = {}
    
    # Feature-based method (cannot use distance matrix)
    algorithms.update({
        f'mean_shift_q={q:.2f}': partial(mean_shift, q=q, min_size=min_size)
        for q in quantiles
    })

    # Distance-based methods (work with general metric spaces)
    algorithms.update({'HDBSCAN': partial(hdbscan, metric=metric, min_size=min_size)})
    
    algorithms.update({
        f'dbscan_eps={eps:.2f}': partial(dbscan, eps=eps, metric=metric, min_size=min_size)
        for eps in db_epsilons
    })
    
    algorithms.update({
        f'optics_min_samples={ms}': partial(optics, min_samples=ms, metric=metric, min_size=min_size)
        for ms in optics_min_samples
    })
    
    algorithms.update({
        'affinity_propagation': partial(affinity_propagation, metric=metric, min_size=min_size)
    })

    # Graph-based algorithms (don't need metric)
    algorithms.update({
        f'leiden_gamma={gamma:.2f}': partial(leiden, gamma=gamma, min_size=min_size)
        for gamma in leiden_gammas
    })

    algorithms.update({
        'louvain': partial(louvain, min_size=min_size),
        'greedy_modularity': partial(greedy_modularity, min_size=min_size),
        'label_propagation': partial(label_propagation, min_size=min_size),
        'infomap': partial(infomap, min_size=min_size),
        'walktrap': partial(walktrap, min_size=min_size)
    })

    return algorithms