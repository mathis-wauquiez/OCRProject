"""
MRF with Loopy Belief Propagation — NFA + OCR fusion for cluster refinement.

Defines a Markov Random Field over the NFA graph where:
  - Unary potentials encode OCR predictions (linguistic signal)
  - Pairwise potentials encode visual similarity (NFA signal)

Inference is done via max-sum loopy BP on the sparse NFA graph.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List
from collections import defaultdict

from .refinement import ClusterRefinementStep, RefinementResult


class MRFBeliefPropagation:
    """
    Loopy Belief Propagation on an NFA-weighted graph with OCR unaries.

    Energy:
        E(z) = sum_i psi_i(z_i) + sum_{(i,j) in E} psi_ij(z_i, z_j)

    Unary potential (OCR):
        psi_i(z_i = c) = -log P_OCR(I_i = c)

    Pairwise potential (NFA — generalized Potts):
        psi_ij(z_i, z_j) = -beta * NLFA(i,j)  if z_i == z_j
                          = 0                    otherwise

    Parameters
    ----------
    beta : float or None
        Coupling strength.  ``None`` → ``1 / median(NLFA)``.
    max_iter : int
        Maximum BP iterations.
    damping : float
        Message damping in [0, 1).  0 = no damping, 0.5 recommended.
    convergence_tol : int
        Stop if assignments unchanged for this many iterations.
    """

    def __init__(
        self,
        beta: Optional[float] = None,
        max_iter: int = 100,
        damping: float = 0.5,
        convergence_tol: int = 5,
    ):
        self.beta = beta
        self.max_iter = max_iter
        self.damping = damping
        self.convergence_tol = convergence_tol

    def fit(
        self,
        nlfa: np.ndarray,
        edges: np.ndarray,
        ocr_probs: Optional[np.ndarray] = None,
        label_space: Optional[List[str]] = None,
        init_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run loopy BP.

        Parameters
        ----------
        nlfa : ndarray (N, N)
            NLFA matrix (sparse; only edge entries are used).
        edges : ndarray (E, 2)
            Edge list.
        ocr_probs : ndarray (N, K) or None
            Per-character OCR probability over K labels.
            If None, uniform unaries are used.
        label_space : list of str, length K
            Names corresponding to columns of ocr_probs.
        init_labels : ndarray (N,) of int or None
            Initial cluster assignments (e.g., from HDBSCAN).
            If provided, the label space is derived from these.

        Returns
        -------
        dict with:
            ``labels`` : ndarray (N,) — MAP assignment
            ``beliefs`` : ndarray (N, K) — final beliefs
            ``n_iter`` : int — iterations run
        """
        N = nlfa.shape[0]

        # Determine label space
        if init_labels is not None:
            unique_labels = np.unique(init_labels[init_labels >= 0])
            K = len(unique_labels)
            label_map = {l: i for i, l in enumerate(unique_labels)}
        elif ocr_probs is not None:
            K = ocr_probs.shape[1]
        else:
            raise ValueError("Either init_labels or ocr_probs must be given.")

        # Build adjacency (sparse)
        neighbors: Dict[int, List[int]] = defaultdict(list)
        edge_nlfa: Dict[tuple, float] = {}
        for i, j in edges:
            i, j = int(i), int(j)
            if i == j:
                continue
            neighbors[i].append(j)
            edge_nlfa[(i, j)] = float(nlfa[i, j])

        # Set beta
        if self.beta is None:
            nlfa_vals = np.array(list(edge_nlfa.values()))
            median_nlfa = np.median(nlfa_vals) if len(nlfa_vals) > 0 else 1.0
            beta = 1.0 / max(median_nlfa, 1e-8)
        else:
            beta = self.beta

        # Unary potentials (log domain: higher = more preferred)
        unary = np.zeros((N, K))
        if ocr_probs is not None:
            # -log(prob) as cost → convert to log-belief (negate)
            unary = np.log(np.clip(ocr_probs, 1e-10, 1.0))
        elif init_labels is not None:
            # Soft assignment from initial labels
            for i in range(N):
                if init_labels[i] >= 0 and init_labels[i] in label_map:
                    unary[i, label_map[init_labels[i]]] = 1.0

        # Initialize messages: m[i->j] is shape (K,), stored as dict
        # Use tuple (i,j) as key
        messages = {}
        for i, j in edges:
            i, j = int(i), int(j)
            if i != j:
                messages[(i, j)] = np.zeros(K)

        # BP iterations
        prev_assignment = np.full(N, -1, dtype=int)
        stable_count = 0

        for iteration in range(self.max_iter):
            new_messages = {}

            for i, j in edges:
                i, j = int(i), int(j)
                if i == j:
                    continue

                # Compute m_{i->j}(z_j) = max_{z_i} [psi_i(z_i) + psi_ij(z_i,z_j) + sum_{k!=j} m_{k->i}(z_i)]
                # Collect incoming messages to i excluding j
                incoming = unary[i].copy()
                for k in neighbors.get(i, []):
                    if k != j and (k, i) in messages:
                        incoming += messages[(k, i)]

                # Pairwise: -beta * NLFA when same label, 0 otherwise
                nlfa_val = edge_nlfa.get((i, j), 0.0)
                pairwise_same = beta * nlfa_val  # reward for same label

                # m_{i->j}(z_j) for each z_j:
                #   = max(max_{z_i != z_j} incoming[z_i],
                #         incoming[z_j] + pairwise_same)
                msg = np.zeros(K)
                max_incoming = np.max(incoming)  # max over all z_i
                for zj in range(K):
                    # Option 1: z_i = z_j (same label) → incoming[z_j] + pairwise_same
                    same_val = incoming[zj] + pairwise_same
                    # Option 2: z_i != z_j → max of incoming (approx)
                    # For efficiency, use max of all incoming
                    diff_val = max_incoming
                    msg[zj] = max(same_val, diff_val)

                # Normalize messages for numerical stability
                msg -= msg.max()

                # Damping
                old_key = (i, j)
                if old_key in messages:
                    msg = self.damping * messages[old_key] + (1 - self.damping) * msg

                new_messages[(i, j)] = msg

            messages = new_messages

            # Compute beliefs and check convergence
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

        # Map back to original labels if init_labels was used
        if init_labels is not None:
            reverse_map = {i: l for l, i in label_map.items()}
            final_labels = np.array([reverse_map.get(a, -1) for a in assignment])
        else:
            final_labels = assignment

        return {
            'labels': final_labels,
            'beliefs': beliefs,
            'n_iter': iteration + 1,
            'beta': beta,
        }


# ---------------------------------------------------------------------------
#  Refinement step wrapper
# ---------------------------------------------------------------------------


class MRFRefinementStep(ClusterRefinementStep):
    """
    Refinement step that runs MRF-BP to jointly optimize cluster
    assignments using NFA pairwise potentials and OCR unary potentials.

    Subsumes the 3 ad hoc stages (Hausdorff split, OCR rematch, PCA z-score)
    into a single principled energy minimization.
    """

    name = "mrf_bp"

    def __init__(
        self,
        beta: Optional[float] = None,
        max_iter: int = 100,
        damping: float = 0.5,
        ocr_label_col: str = 'char_chat',
        ocr_conf_col: str = 'conf_chat',
    ):
        self.mrf = MRFBeliefPropagation(
            beta=beta, max_iter=max_iter, damping=damping,
        )
        self.ocr_label_col = ocr_label_col
        self.ocr_conf_col = ocr_conf_col

    def run(self, dataframe, membership, renderer, *,
            target_lbl, graph=None, nlfa=None, **ctx) -> RefinementResult:
        """
        Run MRF-BP refinement.

        Requires ``nlfa`` and ``graph`` in ctx or as keyword args.
        """
        import torch

        if nlfa is None:
            nlfa = ctx.get('nlfa')
        if nlfa is None:
            # Fallback: no NLFA available, return unchanged
            return RefinementResult(
                membership=membership, log=[],
                metadata={'error': 'no NLFA matrix provided'},
            )

        if isinstance(nlfa, torch.Tensor):
            nlfa_np = nlfa.cpu().numpy()
        else:
            nlfa_np = np.asarray(nlfa)

        # Build edge list from graph or NLFA
        if graph is not None:
            edges = np.array(list(graph.edges()), dtype=int)
        else:
            # Use NLFA > 0 as edges
            row, col = np.where(nlfa_np > 0)
            edges = np.column_stack([row, col])

        # Build OCR probability matrix
        ocr_probs = None
        label_space = None
        if self.ocr_label_col in dataframe.columns:
            labels = dataframe[self.ocr_label_col].fillna('').values
            unique_labels = sorted(set(l for l in labels if l and l != '\u25af'))
            if unique_labels:
                label_space = unique_labels
                K = len(unique_labels)
                label_to_idx = {l: i for i, l in enumerate(unique_labels)}
                N = len(dataframe)
                ocr_probs = np.full((N, K), 1.0 / K)  # uniform prior

                confs = np.ones(N)
                if self.ocr_conf_col in dataframe.columns:
                    confs = dataframe[self.ocr_conf_col].fillna(0.5).values

                for i in range(N):
                    lbl = labels[i]
                    if lbl in label_to_idx:
                        c = float(confs[i])
                        ocr_probs[i, :] = (1 - c) / K
                        ocr_probs[i, label_to_idx[lbl]] = c

        result = self.mrf.fit(
            nlfa=nlfa_np,
            edges=edges,
            ocr_probs=ocr_probs,
            label_space=label_space,
            init_labels=membership if ocr_probs is None else None,
        )

        return RefinementResult(
            membership=result['labels'],
            log=[{
                'n_iter': result['n_iter'],
                'beta': result['beta'],
            }],
            metadata=result,
        )
