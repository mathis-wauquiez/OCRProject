# Implementation Plan

## Overview

Four interconnected workstreams (A–D) that share an abstraction layer (B).

---

## A. Fix UNKNOWN_LABEL handling in cluster splitting

**Problem:** `_compute_cluster_linkages` computes pairwise Hausdorff distances for every patch in a Leiden cluster, including `UNKNOWN_LABEL ('▯')` patches. These unknowns pollute the distance matrix and linkage tree — they should be excluded from the splitting decision, then reassigned to whatever sub-cluster their original cluster maps to.

**Changes:**

1. **`clustering_sweep.py: _compute_cluster_linkages`** — before calling `compute_distance_matrices_batched`, partition `idx` into `known_idx` (patches with a known label or no label column issue) and `unknown_idx`. Build the distance matrix and linkage only on `known_idx`. Store both in the result dict.

2. **`clustering_sweep.py: _apply_split_threshold`** — after `fcluster` on the known-only linkage, assign each unknown patch to the same sub-cluster as the plurality of known patches from its original Leiden cluster (or to the largest sub-cluster if no known patches exist).

**Files:** `src/clustering/clustering_sweep.py`

---

## B. Abstraction layer: `ClusterRefinementStep` protocol

**Goal:** A uniform interface for any post-Leiden refinement operation (splitting, merging, rematching) so they can be composed, configured via YAML, and reported uniformly.

**Design:**

```python
# src/clustering/refinement.py  (NEW file)

class ClusterRefinementStep(ABC):
    """One step in a post-partition refinement pipeline."""
    name: str

    @abstractmethod
    def run(self, dataframe, membership, renderer, **ctx) -> RefinementResult:
        """
        Args:
            dataframe: full dataframe with 'svg', target_lbl, 'histogram', etc.
            membership: np.ndarray of current cluster IDs
            renderer: Renderer instance
            **ctx: extra context (graph, features, reg_metric, …)
        Returns:
            RefinementResult with new membership + diagnostics
        """
        ...

@dataclass
class RefinementResult:
    membership: np.ndarray
    log: list[dict]            # per-cluster action log (for reporting)
    metadata: dict             # anything the reporter might need
```

**Concrete implementations (each its own class):**

| Class | File | What it does |
|---|---|---|
| `HausdorffSplitStep` | `refinement.py` | Current Hausdorff splitting logic, extracted from `_compute_cluster_linkages` + `_apply_split_threshold` |
| `OCRRematchStep` | `refinement.py` | Stage-1 rematching: use recognized character labels to merge small/singleton clusters with compatible larger ones |
| `PCAZScoreRematchStep` | `refinement.py` | Stage-2 rematching: register against most-central patch, PCA on registered images, z-score test on first k components |
| `HausdorffMergeStep` | `refinement.py` | Alternative stage-2: Hausdorff distance from singleton to cluster centroids, merge if below threshold |

**Integration into `graphClusteringSweep`:**

- New constructor param: `refinement_steps: list[ClusterRefinementStep]` (default: `[HausdorffSplitStep(...)]` for backward compat).
- `report_graph` runs each step sequentially, passing the output membership of step N as input to step N+1.
- Each step's `RefinementResult.log` feeds into a new reporter method `report_refinement_step()`.

**Config:** `confs/clustering.yaml` gains a `refinement_steps` list that Hydra instantiates.

**Files:** `src/clustering/refinement.py` (new), `src/clustering/clustering_sweep.py` (refactor), `confs/clustering.yaml`

---

## C. Rematching methodology

### C.1 Reporting fragmented characters

**Problem:** A character type may be split across multiple clusters. We already compute `_compute_label_dataframe` (completeness stats per label), but we need a dedicated, clear report section showing:
- Which labels are fragmented (spread across >1 cluster)
- For each fragmented label: which clusters contain it, how many patches in each, the dominant cluster's share
- A visual: for each fragmented label, show the representative from each cluster side by side

**Changes:**
- New reporter method `report_fragmentation(dataframe, label_dataframe)` in `clustering_sweep_report.py`.
- Called from `report_graph_results` as a new section "Label Fragmentation".

### C.2 Stage-1 rematching: OCR-based (`OCRRematchStep`)

For each small cluster (size ≤ `rematch_max_size`) or singleton:
1. Look at the recognized character label(s) in the cluster.
2. Find the largest cluster whose dominant label matches.
3. Merge the small cluster into it (reassign membership).

This is fast (no GPU), purely label-driven, and handles the easy cases.

### C.3 Stage-2 rematching: PCA z-score (`PCAZScoreRematchStep`)

For each remaining small cluster / singleton after stage 1:
1. Find candidate target clusters (e.g., top-k by HOG feature similarity to the small cluster's centroid).
2. For each candidate:
   a. Get the most-central patch of the candidate cluster (the "anchor").
   b. Register (IC) the query patch against the anchor.
   c. Collect the registered images of all patches in the candidate cluster (already computed during splitting, or compute lazily).
   d. Run PCA on the candidate cluster's registered patches (first `k` components, `k` from config).
   e. Project the query patch into this PCA space.
   f. Compute the z-score of the query's projection on each of the first `k` components.
   g. If all z-scores are below a threshold `z_max`, the query is compatible → merge.
3. Among compatible candidates, pick the one with lowest aggregate z-score.

**Config params:** `rematch_max_size`, `rematch_pca_k`, `rematch_z_max`, `rematch_n_candidates`

### C.4 Alternative stage-2 ideas (implement reporting hooks, leave as future)

- **Hausdorff-to-centroid:** compute Hausdorff distance from query to each candidate cluster's most-central patch; merge if below threshold.
- **Feature-space kNN:** use HOG feature L2/CEMD distance to the k nearest neighbors; if majority are in one cluster, merge.
- **A-contrario rematch:** run `featureMatching.match_subset()` between the query and candidate cluster members; merge if significant matches exist.

These can be implemented as additional `ClusterRefinementStep` subclasses later; the abstraction supports them.

### C.5 Integration with splitting

The same `ClusterRefinementStep` protocol supports both splitting and merging. The pipeline in `report_graph` becomes:

```
Leiden partition
  → HausdorffSplitStep        (split large impure clusters)
  → OCRRematchStep             (merge small clusters by label)
  → PCAZScoreRematchStep       (merge remaining singletons by image similarity)
```

Each step produces a `RefinementResult` and the reporter generates a section for each.

**Files:**
- `src/clustering/refinement.py` (new — all step implementations)
- `src/clustering/clustering_sweep.py` (refactor `report_graph` to use pipeline)
- `src/clustering/clustering_sweep_report.py` (new `report_refinement_step`, `report_fragmentation`)
- `confs/clustering.yaml` (new params)

---

## D. Methodology report

A new dedicated report section "Methodology" generated at the top of the HTML report, covering:

1. **Feature extraction:** HOG parameters, gradient method, cell size, normalization
2. **A-contrario matching:** CEMD/L2, epsilon, NLFA threshold, reciprocal matching
3. **Graph construction:** edge type, reciprocal filtering
4. **Community detection:** Leiden, resolution parameter gamma
5. **Cluster refinement pipeline:** list of steps with their parameters
6. **Evaluation metrics:** which metrics are computed, how UNKNOWN_LABEL is handled

This is a static summary section (no figures), assembled from the config parameters stored in `self`. Placed as Section 0 in the report.

**Files:** `src/clustering/clustering_sweep_report.py` (new method `report_methodology`)

---

## Execution Order

1. **B first** — create `refinement.py` with the abstract base + `RefinementResult`
2. **A** — implement `HausdorffSplitStep` (extract from current code), fix UNKNOWN handling
3. **C.1** — fragmentation reporting
4. **C.2** — `OCRRematchStep`
5. **C.3** — `PCAZScoreRematchStep`
6. **Wire up** — refactor `clustering_sweep.py` to use the refinement pipeline
7. **D** — methodology report section
8. **Config** — update `confs/clustering.yaml`

---

## Files Changed / Created

| File | Action |
|---|---|
| `src/clustering/refinement.py` | **CREATE** — ABC + all refinement steps |
| `src/clustering/clustering_sweep.py` | **EDIT** — refactor `report_graph` to use refinement pipeline, remove inlined splitting logic |
| `src/clustering/clustering_sweep_report.py` | **EDIT** — add `report_refinement_step`, `report_fragmentation`, `report_methodology` |
| `confs/clustering.yaml` | **EDIT** — add refinement pipeline config |
