"""
MRF with Loopy Belief Propagation — NFA + OCR fusion for cluster refinement.

Unary potentials encode OCR predictions, pairwise potentials encode
NFA visual similarity.  Inference via max-sum loopy BP on the sparse graph.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional, Dict, Any, List
from collections import defaultdict

from .refinement import ClusterRefinementStep, RefinementResult

log = logging.getLogger(__name__)


class MRFBeliefPropagation:
    """Loopy BP on an NFA-weighted graph with OCR unaries.

    Parameters
    ----------
    beta : float or None
        Coupling strength.  ``None`` → ``1 / median(NLFA)``.
    max_iter, damping, convergence_tol : BP parameters.
    """

    def __init__(self, beta=None, max_iter=100, damping=0.5, convergence_tol=5):
        self.beta = beta
        self.max_iter = max_iter
        self.damping = damping
        self.convergence_tol = convergence_tol

    def fit(self, nlfa, edges, ocr_probs=None, init_labels=None):
        N = nlfa.shape[0]

        # Label space
        if init_labels is not None:
            unique_labels = np.unique(init_labels[init_labels >= 0])
            K = len(unique_labels)
            label_map = {l: i for i, l in enumerate(unique_labels)}
        elif ocr_probs is not None:
            K = ocr_probs.shape[1]
            label_map = None
        else:
            raise ValueError("Either init_labels or ocr_probs must be given.")

        # Sparse adjacency
        neighbors = defaultdict(list)
        edge_nlfa = {}
        for i, j in edges:
            i, j = int(i), int(j)
            if i != j:
                neighbors[i].append(j)
                edge_nlfa[(i, j)] = float(nlfa[i, j])

        # Beta
        if self.beta is None:
            vals = np.array(list(edge_nlfa.values()))
            beta = 1.0 / max(np.median(vals) if len(vals) else 1.0, 1e-8)
        else:
            beta = self.beta

        # Unary potentials
        unary = np.zeros((N, K))
        if ocr_probs is not None:
            unary = np.log(np.clip(ocr_probs, 1e-10, 1.0))
        elif init_labels is not None:
            for i in range(N):
                if init_labels[i] >= 0 and init_labels[i] in label_map:
                    unary[i, label_map[init_labels[i]]] = 1.0

        # Messages
        messages = {(int(i), int(j)): np.zeros(K) for i, j in edges if int(i) != int(j)}

        prev_assignment = np.full(N, -1, dtype=int)
        stable_count = 0

        for iteration in range(self.max_iter):
            new_messages = {}
            for i, j in edges:
                i, j = int(i), int(j)
                if i == j:
                    continue
                incoming = unary[i].copy()
                for k in neighbors.get(i, []):
                    if k != j and (k, i) in messages:
                        incoming += messages[(k, i)]

                pairwise_same = beta * edge_nlfa.get((i, j), 0.0)
                max_incoming = np.max(incoming)
                msg = np.array([max(incoming[zj] + pairwise_same, max_incoming) for zj in range(K)])
                msg -= msg.max()

                if (i, j) in messages:
                    msg = self.damping * messages[(i, j)] + (1 - self.damping) * msg
                new_messages[(i, j)] = msg

            messages = new_messages

            beliefs = unary.copy()
            for i in range(N):
                for k in neighbors.get(i, []):
                    if (k, i) in messages:
                        beliefs[i] += messages[(k, i)]

            assignment = beliefs.argmax(axis=1)
            if np.array_equal(assignment, prev_assignment):
                stable_count += 1
                if stable_count >= self.convergence_tol:
                    break
            else:
                stable_count = 0
            prev_assignment = assignment.copy()

        if init_labels is not None:
            reverse_map = {i: l for l, i in label_map.items()}
            final_labels = np.array([reverse_map.get(a, -1) for a in assignment])
        else:
            final_labels = assignment

        return {'labels': final_labels, 'beliefs': beliefs,
                'n_iter': iteration + 1, 'beta': beta}


class MRFRefinementStep(ClusterRefinementStep):
    """Refinement step that runs MRF-BP on NFA pairwise + OCR unary potentials.

    Requires ``nlfa`` (and optionally ``graph``) to be passed through ``ctx``.
    """

    name = "mrf_bp"

    def __init__(self, beta=None, max_iter=100, damping=0.5,
                 ocr_label_col='char_chat', ocr_conf_col='conf_chat'):
        self.mrf = MRFBeliefPropagation(beta=beta, max_iter=max_iter, damping=damping)
        self.ocr_label_col = ocr_label_col
        self.ocr_conf_col = ocr_conf_col

    def run(self, dataframe, membership, renderer, *, target_lbl, **ctx):
        import torch

        nlfa = ctx.get('nlfa')
        graph = ctx.get('graph')
        if nlfa is None:
            log.warning("MRFRefinementStep: no NLFA in ctx, skipping.")
            return RefinementResult(membership=membership, log=[], metadata={'skipped': True})

        nlfa_np = nlfa.cpu().numpy() if isinstance(nlfa, torch.Tensor) else np.asarray(nlfa)

        # Edges
        if graph is not None:
            edges = np.array(list(graph.edges()), dtype=int)
        else:
            row, col = np.where(nlfa_np > 0)
            edges = np.column_stack([row, col])

        # OCR probabilities
        ocr_probs = None
        if self.ocr_label_col in dataframe.columns:
            labels = dataframe[self.ocr_label_col].fillna('').values
            unique = sorted(set(l for l in labels if l and l != '\u25af'))
            if unique:
                K = len(unique)
                lbl2idx = {l: i for i, l in enumerate(unique)}
                N = len(dataframe)
                ocr_probs = np.full((N, K), 1.0 / K)
                confs = dataframe[self.ocr_conf_col].fillna(0.5).values if self.ocr_conf_col in dataframe.columns else np.ones(N)
                for i in range(N):
                    if labels[i] in lbl2idx:
                        c = float(confs[i])
                        ocr_probs[i, :] = (1 - c) / K
                        ocr_probs[i, lbl2idx[labels[i]]] = c

        result = self.mrf.fit(nlfa_np, edges, ocr_probs=ocr_probs,
                              init_labels=membership if ocr_probs is None else None)

        return RefinementResult(membership=result['labels'],
                                log=[{'n_iter': result['n_iter'], 'beta': result['beta']}],
                                metadata=result)
