
from .feature_matching import featureMatching
from .algorithms import (
    leiden_core, louvain_core, greedy_modularity_core, label_propagation_core
)

import numpy as np
import torch

from abc import ABC, abstractmethod
import networkx as nx

from typing import List

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
        return leiden_core(G, self.gamma)


class louvainCommunityDetection(communityDetectionBase):
    """Fast, widely-used modularity optimization"""
    name = "louvain"

    def __call__(self, G: nx.Graph) -> Membership:
        return louvain_core(G)


class greedyModularityCommunityDetection(communityDetectionBase):
    """NetworkX built-in greedy modularity"""
    name = "greedy_modularity"

    def __call__(self, G: nx.Graph) -> Membership:
        return greedy_modularity_core(G)

class labelPropagationCommunityDetection(communityDetectionBase):
    """Fast, near-linear time algorithm"""
    name = "label_propagation"

    def __call__(self, G: nx.Graph) -> Membership:
        return label_propagation_core(G)


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