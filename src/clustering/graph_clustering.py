"""Backward-compatibility shim — all definitions now live in algorithms.py."""

from .algorithms import (
    Membership,
    communityDetectionBase,
    leidenCommunityDetection as leindenCommunityDetection,
    louvainCommunityDetection,
    greedyModularityCommunityDetection,
    labelPropagationCommunityDetection,
    graphClustering,
)
