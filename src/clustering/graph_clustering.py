
from .feature_matching import featureMatching

import numpy as np
import torch

from abc import ABC, abstractmethod
import networkx as nx

from typing import List

import igraph as ig
import leidenalg as la

Membership = List[int]

class communityDetectionBase(ABC):
    name: str = "abstract"

    @abstractmethod
    def __call__(
        self, graph: nx.Graph
    ) -> Membership:
        ...



class leindenCommunityDetection(communityDetectionBase):

    name = "leiden"

    def __init__(self, gamma: float):
        self.gamma = gamma
    
    def __call__(self, G: nx.Graph):
        G_ig = ig.Graph.from_networkx(G)
        partition = la.find_partition(
            G_ig,
            la.RBConfigurationVertexPartition,
            resolution_parameter=self.gamma,
            n_iterations=-1,
            seed=42
        )

        return partition.membership


class louvainCommunityDetection(communityDetectionBase):
    """Fast, widely-used modularity optimization"""
    name = "louvain"
    
    def __call__(self, G: nx.Graph) -> Membership:
        import community.community_louvain as community_louvain
        partition = community_louvain.best_partition(G, random_state=42)
        return [partition[i] for i in range(len(G.nodes()))]


class greedyModularityCommunityDetection(communityDetectionBase):
    """NetworkX built-in greedy modularity"""
    name = "greedy_modularity"
    
    def __call__(self, G: nx.Graph) -> Membership:
        from networkx.algorithms import community
        communities = community.greedy_modularity_communities(G)
        membership = [0] * len(G.nodes())
        for comm_id, comm in enumerate(communities):
            for node in comm:
                membership[node] = comm_id
        return membership

class labelPropagationCommunityDetection(communityDetectionBase):
    """Fast, near-linear time algorithm"""
    name = "label_propagation"
    
    def __call__(self, G: nx.Graph) -> Membership:
        from networkx.algorithms import community
        communities = community.label_propagation_communities(G)
        membership = [0] * len(G.nodes())
        for comm_id, comm in enumerate(communities):
            for node in comm:
                membership[node] = comm_id
        return membership


class graphClustering:
    def __init__(
            self,
            feature: str,
            featureMatcher: featureMatching,
            edges_type: str,
            communityDetection:  communityDetectionBase,
            device: str = "cuda"
    ):
        self.feature = feature
        self.featureMatcher = featureMatcher
        self.edges_type = edges_type
        self.communityDetection = communityDetection
        self.device = device

    def __call__(
            self, 
            dataframe
    ) -> Membership:
        
        # -- Compute the matches thanks to the featureMatching class --

        features = dataframe['feature']
        features = torch.tensor(features, device=self.device)

        matches, nlfa, dissimilarities = self.featureMatcher.match(features, features)

        if self.edges_type == 'nlfa':
            edge_weights = nlfa
            del dissimilarities      # free GPU mem usage
        elif self.edges_type in ['dissim', 'dissimilarities']:
            edge_weights = dissimilarities
            del nlfa                # free GPU mem usage
        elif self.edges_type in ['link', 'constant']:
            edge_weights = torch.ones_like(nlfa)
            del nlfa, dissimilarities # free GPU mem usage

        # -- Build the networkx Graph --

        N = nlfa.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(N))          # One node for every image
        edges = [(int(i.item()), int(j.item()), edge_weights[i, j].item()) for i, j in matches if i != j]  
        G.add_weighted_edges_from(edges)    # Add the weights

        # -- Run the community detection/graph clustering --

        membership = self.communityDetection(G)

        return membership