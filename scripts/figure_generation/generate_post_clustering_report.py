"""
Generate an HTML methodology report for the post-clustering refinement pipeline:
  1. CHAT-based cluster splitting
  2. Hapax-to-cluster association
  3. Glossary generation

Produces a self-contained HTML file with KaTeX math that can be opened in a
browser and printed to PDF.

Usage:
    python scripts/figure_generation/generate_post_clustering_report.py [--output-dir ./reports]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.auto_report import AutoReport, ReportConfig, Theme


def build_report(output_dir: str = "./reports/post_clustering_methodology"):
    report = AutoReport(
        title="Post-Clustering Refinement — Methodology Report",
        author="OCR Project",
        output_dir=output_dir,
        config=ReportConfig(
            include_katex=True,
            theme=Theme.DEFAULT,
            dpi=150,
            output_format="png",
        ),
    )

    # ==================================================================
    # Section 1: Introduction & Motivation
    # ==================================================================
    with report.section("1. Introduction"):
        report.report_text(
            "## Context\n\n"
            "After the Leiden community detection and Hausdorff-based hierarchical "
            "splitting, the pipeline produces a set of character clusters. Each "
            "cluster ideally groups all occurrences of a single character type. "
            "In practice, two systematic problems remain:\n\n"
            "1. **Mixed clusters** — A single cluster may contain two or more "
            "distinct characters that are visually similar (e.g. 大 and 太, or "
            "characters with similar stroke structure). CHAT OCR assigns different "
            "labels to these, revealing the mixture.\n"
            "2. **Orphaned singletons (hapax)** — Characters that appear only once "
            "in the corpus end up in singleton clusters. Many of these actually "
            "belong to an existing multi-element cluster but were isolated because "
            "their HOG feature did not match strongly enough.\n\n"
            "This report describes three post-clustering operations that exploit "
            "the CHAT OCR predictions (`char_chat`) to address these issues and "
            "produce a character glossary.\n\n"
            "## Pipeline Position\n\n"
            "These operations run *after* the existing clustering sweep and "
            "*before* any downstream analysis:\n\n"
            "```\n"
            "Patches + HOG features\n"
            "  → A-contrario matching → Leiden clustering → Hausdorff splitting\n"
            "  → [NEW] CHAT-based cluster splitting          ← Operation 1\n"
            "  → [NEW] Hapax-to-cluster association           ← Operation 2\n"
            "  → [NEW] Glossary generation                   ← Operation 3\n"
            "```",
            title="Motivation & Pipeline Position",
        )

    # ==================================================================
    # Section 2: CHAT-Based Cluster Splitting
    # ==================================================================
    with report.section("2. CHAT-Based Cluster Splitting"):
        report.report_text(
            "## Problem Statement\n\n"
            "After Leiden + Hausdorff splitting, some clusters still contain "
            "patches labelled by CHAT as two or more distinct characters. For "
            "example, a cluster of 30 patches might have 18 labelled '人' and "
            "12 labelled '入'. The Hausdorff splitting could not separate them "
            "because their binary shapes are metrically close.\n\n"
            "## Approach\n\n"
            "We use the CHAT OCR predictions as a *secondary signal* to sub-divide "
            "clusters that the shape-based pipeline could not resolve.\n\n"
            "### Algorithm\n\n"
            "For each cluster \\(C_k\\) with membership indices \\(I_k\\):\n\n"
            "1. **Collect CHAT labels**: Let \\(L_k = \\{\\ell_i : i \\in I_k\\}\\) "
            "be the multiset of CHAT predictions, excluding unknown characters "
            "(\\(\\ell = \\texttt{▯}\\)).\n\n"
            "2. **Count distinct labels**: Let \\(U_k = \\text{unique}(L_k)\\). "
            "If \\(|U_k| \\leq 1\\), the cluster is *pure* w.r.t. CHAT — skip it.\n\n"
            "3. **Check dominance**: Compute the label frequency distribution. "
            "If the most frequent label accounts for \\(\\geq p_{\\text{pure}}\\) "
            "(default 0.90) of known labels, treat the cluster as effectively "
            "pure — the minority labels are likely CHAT errors. Skip.\n\n"
            "4. **Split by CHAT label**: Create one sub-cluster per CHAT label "
            "\\(u \\in U_k\\):\n"
            "$$S_{k,u} = \\{i \\in I_k : \\ell_i = u\\}$$\n\n"
            "5. **Assign unknowns**: Patches with \\(\\ell_i = \\texttt{▯}\\) "
            "are not immediately placed. For each unknown patch \\(i\\):\n"
            "   - Compute the mean HOG dissimilarity "
            "\\(\\bar{d}(i, S_{k,u})\\) to each sub-cluster \\(S_{k,u}\\)\n"
            "   - Assign \\(i\\) to \\(\\arg\\min_u \\bar{d}(i, S_{k,u})\\)\n\n"
            "6. **Reassign IDs**: Replace the original cluster ID \\(k\\) with "
            "new contiguous IDs for each sub-cluster.\n\n"
            "### Parameters\n\n"
            "| Parameter | Default | Description |\n"
            "|-----------|---------|-------------|\n"
            "| `purity_threshold` | 0.90 | Minimum known-label purity to skip splitting |\n"
            "| `min_split_size` | 3 | Minimum cluster size to consider for splitting |\n"
            "| `min_label_count` | 2 | A CHAT label must appear ≥ this many times to form its own sub-cluster (otherwise treated as noise) |\n",
            title="Algorithm",
            is_katex=True,
        )

        report.report_text(
            "## Handling CHAT Errors\n\n"
            "CHAT OCR is not perfect. Several safeguards prevent "
            "erroneous splits:\n\n"
            "1. **Purity threshold** (\\(p_{\\text{pure}} = 0.90\\)): If 90% of "
            "known labels agree, a single outlier prediction is treated as OCR "
            "noise, not a genuine mixture. The cluster is left intact.\n\n"
            "2. **Minimum label count**: A CHAT label that appears only once in "
            "a cluster is almost certainly an error. Such singleton labels are "
            "folded into the dominant group rather than spawning their own "
            "sub-cluster.\n\n"
            "3. **Unknown assignment**: Patches with \\(\\ell = \\texttt{▯}\\) "
            "(CHAT could not recognise the character) are assigned by visual "
            "similarity, not by CHAT label. This prevents information loss.\n\n"
            "4. **Confidence weighting** (optional): When multiple CHAT labels "
            "are close in frequency, the mean confidence score "
            "\\(\\bar{c}_u = \\frac{1}{|S_{k,u}|} \\sum_{i \\in S_{k,u}} "
            "c_i\\) can serve as a tie-breaker — groups with higher average "
            "CHAT confidence are more trustworthy.\n\n"
            "## Complexity\n\n"
            "The splitting is \\(O(K \\cdot \\max|I_k|)\\) — linear in the "
            "total number of patches.  No pairwise distance computation is "
            "needed (the dissimilarity matrix from the matching step is reused "
            "only for unknown assignment).  This is much cheaper than the "
            "Hausdorff splitting step.",
            title="Robustness & Complexity",
            is_katex=True,
        )

    # ==================================================================
    # Section 3: Hapax-to-Cluster Association
    # ==================================================================
    with report.section("3. Hapax-to-Cluster Association"):
        report.report_text(
            "## Problem Statement\n\n"
            "A *hapax* (plural: *hapax legomena*) is a character that appears "
            "exactly once in the corpus.  After clustering, these end up as "
            "singleton clusters — clusters of size 1.  Some of these are "
            "genuinely unique characters, but many are duplicates of existing "
            "multi-element clusters that the matching step missed.\n\n"
            "The goal is to merge hapax singletons into existing clusters where "
            "there is strong evidence (from CHAT) that they represent the same "
            "character, confirmed by visual similarity.\n\n"
            "## Algorithm\n\n"
            "**Input**: DataFrame with `membership`, `char_chat`, `conf_chat`, "
            "`histogram` columns.  The dissimilarity matrix from the feature "
            "matching step.\n\n"
            "**Step 1 — Build cluster label index:**\n"
            "For each non-singleton cluster \\(C_k\\), compute its *dominant "
            "CHAT label*:\n"
            "$$\\hat{\\ell}_k = \\arg\\max_{\\ell} \\sum_{i \\in C_k} "
            "\\mathbb{1}[\\ell_i = \\ell]$$\n"
            "Build a dictionary: \\(\\text{label\\_to\\_clusters}[\\ell] "
            "\\to \\{k : \\hat{\\ell}_k = \\ell\\}\\).\n\n"
            "**Step 2 — For each hapax \\(h\\) with CHAT label \\(\\ell_h\\):**\n\n"
            "1. If \\(\\ell_h = \\texttt{▯}\\), this hapax has no CHAT label — "
            "it cannot be matched by OCR.  **Skip** (it remains a singleton).\n\n"
            "2. Look up candidate clusters: "
            "\\(\\mathcal{C}_h = \\text{label\\_to\\_clusters}[\\ell_h]\\).\n\n"
            "3. If \\(|\\mathcal{C}_h| = 0\\), no existing cluster has this CHAT "
            "label.  The hapax is genuinely unique.  **Skip**.\n\n"
            "4. **Visual verification**: For each candidate cluster \\(C_k\\), "
            "compute the mean dissimilarity between the hapax and the cluster "
            "members:\n"
            "$$\\bar{d}(h, C_k) = \\frac{1}{|C_k|} \\sum_{i \\in C_k} "
            "d(h, i)$$\n"
            "where \\(d(h, i)\\) is the precomputed HOG dissimilarity.\n\n"
            "5. Select the best candidate:\n"
            "$$k^* = \\arg\\min_{k \\in \\mathcal{C}_h} \\bar{d}(h, C_k)$$\n\n"
            "6. **Acceptance test**: Accept the match if "
            "\\(\\bar{d}(h, C_{k^*}) < \\tau_{\\text{hapax}}\\), where "
            "\\(\\tau_{\\text{hapax}}\\) is a dissimilarity threshold.  A natural "
            "choice is the median intra-cluster dissimilarity of \\(C_{k^*}\\).\n\n"
            "7. If accepted, set \\(\\text{membership}[h] = k^*\\).\n\n"
            "**Step 3 — Renumber**: Renumber cluster IDs contiguously.",
            title="Algorithm",
            is_katex=True,
        )

        report.report_text(
            "## Design Decisions\n\n"
            "**Why require a CHAT label match?**  Without CHAT, we would need "
            "to search all \\(K\\) clusters for each hapax — an \\(O(K)\\) scan. "
            "CHAT narrows the search to a handful of candidates (typically 0–2). "
            "More importantly, it prevents false merges between visually similar "
            "but semantically different characters.\n\n"
            "**Why the visual verification step?**  CHAT can produce the same "
            "label for different characters (especially common radicals shared "
            "across characters).  The dissimilarity check acts as a second "
            "opinion, preventing incorrect merges.\n\n"
            "**What about hapax with \\(\\ell_h = \\texttt{▯}\\)?**  These "
            "patches could not be recognised by CHAT at all.  Without a label "
            "to narrow the search, we have no reliable way to match them. They "
            "remain as singletons.  A future improvement could use a k-NN "
            "fallback with a strict dissimilarity threshold.\n\n"
            "## Parameters\n\n"
            "| Parameter | Default | Description |\n"
            "|-----------|---------|-------------|\n"
            "| `hapax_max_dissimilarity` | `None` (use median intra-cluster) | Maximum mean dissimilarity to accept a hapax–cluster match |\n"
            "| `hapax_min_confidence` | 0.3 | Minimum CHAT confidence for the hapax label to be trusted |\n",
            title="Design Decisions & Parameters",
            is_katex=True,
        )

    # ==================================================================
    # Section 4: Glossary Generation
    # ==================================================================
    with report.section("4. Glossary Generation"):
        report.report_text(
            "## Objective\n\n"
            "Produce a sorted inventory of all distinct characters found in the "
            "corpus, with occurrence counts and representative images. This is "
            "the primary deliverable for philological analysis.\n\n"
            "## Algorithm\n\n"
            "After CHAT-based splitting and hapax association, each cluster "
            "\\(C_k\\) represents a single character type (ideally).\n\n"
            "**For each cluster \\(C_k\\):**\n\n"
            "1. **Canonical label**: Take the dominant CHAT label among known "
            "patches:\n"
            "$$\\hat{\\ell}_k = \\arg\\max_{\\ell \\neq \\texttt{▯}} "
            "\\sum_{i \\in C_k} \\mathbb{1}[\\ell_i = \\ell]$$\n"
            "If all patches are unknown, use \\(\\hat{\\ell}_k = \\texttt{▯}\\).\n\n"
            "2. **Occurrence count**: \\(n_k = |C_k|\\).\n\n"
            "3. **Representative patch**: The patch with the highest degree "
            "centrality in the similarity graph:\n"
            "$$r_k = \\arg\\max_{i \\in C_k} \\text{degree\\_centrality}(i)$$\n\n"
            "4. **Confidence score**: Mean CHAT confidence over known patches:\n"
            "$$\\bar{c}_k = \\frac{1}{|K_k|} \\sum_{i \\in K_k} c_i, "
            "\\quad K_k = \\{i \\in C_k : \\ell_i \\neq \\texttt{▯}\\}$$\n\n"
            "5. **Purity**: Fraction of known patches that agree with the "
            "dominant label:\n"
            "$$p_k = \\frac{\\max_\\ell |\\{i \\in K_k : \\ell_i = \\ell\\}|}"
            "{|K_k|}$$\n\n"
            "## Output Format\n\n"
            "The glossary is a pandas DataFrame sorted by occurrence count "
            "(descending):\n\n"
            "| Column | Type | Description |\n"
            "|--------|------|-------------|\n"
            "| `character` | str | Dominant CHAT label |\n"
            "| `count` | int | Number of occurrences |\n"
            "| `cluster_id` | int | Cluster ID after refinement |\n"
            "| `purity` | float | Label agreement within cluster |\n"
            "| `mean_confidence` | float | Mean CHAT confidence |\n"
            "| `representative_idx` | int | DataFrame index of representative patch |\n"
            "| `pages` | list | Pages on which this character appears |\n\n"
            "The glossary is also rendered as an HTML report section with "
            "thumbnail images of representative patches.",
            title="Algorithm & Output",
            is_katex=True,
        )

    # ==================================================================
    # Section 5: Ordering & Interaction
    # ==================================================================
    with report.section("5. Operation Ordering & Interaction"):
        report.report_text(
            "## Execution Order\n\n"
            "The three operations are applied in a strict sequence:\n\n"
            "1. **CHAT-based splitting** comes first, because it may increase "
            "the number of clusters. This creates new potential targets for "
            "hapax association.\n\n"
            "2. **Hapax association** comes second. It references the "
            "post-split cluster inventory and dominant labels. Running it "
            "after splitting ensures that previously-mixed clusters (which may "
            "have had ambiguous dominant labels) are now resolved.\n\n"
            "3. **Glossary generation** comes last, operating on the final, "
            "refined cluster assignments.\n\n"
            "## Idempotency\n\n"
            "All three operations are deterministic given the same input. "
            "Re-running them produces identical results.\n\n"
            "## Data Flow\n\n"
            "```\n"
            "Input: dataframe['membership']  (from Leiden + Hausdorff split)\n"
            "       dataframe['char_chat']   (CHAT OCR predictions)\n"
            "       dataframe['conf_chat']   (CHAT confidence scores)\n"
            "       dataframe['histogram']   (HOG feature vectors)\n"
            "       dissimilarities          (N×N pairwise HOG dissimilarity matrix)\n"
            "\n"
            "Step 1: chat_split(dataframe, purity_threshold=0.90)\n"
            "        → dataframe['membership'] updated (more clusters)\n"
            "        → dataframe['membership_pre_chat_split'] saved\n"
            "\n"
            "Step 2: associate_hapax(dataframe, dissimilarities)\n"
            "        → dataframe['membership'] updated (fewer singletons)\n"
            "        → dataframe['membership_pre_hapax'] saved\n"
            "\n"
            "Step 3: build_glossary(dataframe)\n"
            "        → glossary_df  (one row per character type)\n"
            "```",
            title="Ordering & Data Flow",
        )

    # ==================================================================
    # Section 6: Evaluation & Metrics
    # ==================================================================
    with report.section("6. Evaluation"):
        report.report_text(
            "## Metrics\n\n"
            "Each operation is evaluated against the CHAT labels (as reference) "
            "using the same metrics already computed by the clustering sweep:\n\n"
            "- **Adjusted Rand Index (ARI)**: Measures agreement between "
            "predicted clusters and reference labels, adjusted for chance.\n"
            "- **Purity**: Fraction of patches in a cluster that share the "
            "dominant label.\n"
            "- **Completeness**: Fraction of a character's occurrences that are "
            "in the same cluster.\n\n"
            "Additionally, we track operation-specific metrics:\n\n"
            "| Metric | Operation | Description |\n"
            "|--------|-----------|-------------|\n"
            "| Clusters split | Splitting | Number of clusters that were divided |\n"
            "| Sub-clusters created | Splitting | Total new clusters from splitting |\n"
            "| Mean post-split purity | Splitting | Average purity after splitting |\n"
            "| Hapax matched | Association | Number of singletons merged |\n"
            "| Hapax remaining | Association | Number of singletons still isolated |\n"
            "| Match rate | Association | Fraction of hapax successfully merged |\n"
            "| Glossary entries | Glossary | Number of distinct character types |\n"
            "| Coverage | Glossary | Fraction of patches with a known character label |\n",
            title="Evaluation Metrics",
        )

    # ==================================================================
    # Section 7: Summary
    # ==================================================================
    with report.section("7. Summary"):
        report.report_text(
            "The three post-clustering operations form a lightweight refinement "
            "pass that leverages CHAT OCR as a secondary signal:\n\n"
            "1. **CHAT splitting** resolves mixed clusters that shape-based "
            "splitting could not separate.\n"
            "2. **Hapax association** reduces singleton noise by merging "
            "orphaned characters into their correct cluster.\n"
            "3. **Glossary generation** produces the final character inventory.\n\n"
            "All three operations are computationally cheap (no GPU, no new "
            "feature computation) and add at most a few seconds to the pipeline. "
            "The key insight is that CHAT provides a complementary signal to HOG "
            "features: while HOG captures shape similarity, CHAT captures "
            "semantic identity. Combining both gives a more accurate clustering.",
            title="Summary",
        )

    # Generate
    html_path = report.generate_html()
    print(f"\nReport generated: file://{html_path.absolute()}")
    return html_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate post-clustering methodology report")
    parser.add_argument("--output-dir",
                        default="./reports/post_clustering_methodology",
                        help="Output directory for the report")
    args = parser.parse_args()
    build_report(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
