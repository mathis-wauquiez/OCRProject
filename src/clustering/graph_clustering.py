"""Backward-compatibility shim — all definitions now live in algorithms.py."""

from .algorithms import (
    Membership,
    communityDetectionBase,
    leidenCommunityDetection,
    leidenCommunityDetection as leindenCommunityDetection,  # legacy typo alias
    louvainCommunityDetection,
    greedyModularityCommunityDetection,
    labelPropagationCommunityDetection,
    graphClustering,
)

from .nfa_clustering import (
    HDBSCANClustering,
    AffinityPropagationClustering,
    HDBSCANNFACommunityDetection,
    AffinityPropagationNFACommunityDetection,
)

from .mrf_refinement import MRFRefinementStep
from .kmedoids_refinement import KMedoidsSplitMergeStep
