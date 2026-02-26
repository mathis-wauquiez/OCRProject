"""
Generate an HTML methodology report for the character extraction pipeline.

Produces a self-contained HTML file (with KaTeX math rendering) that can be
opened in a browser and printed to PDF.  When ``--viz-dir`` is supplied the
report embeds real pipeline visualisations instead of figure placeholders.

Usage:
    python scripts/figure_generation/generate_extraction_report.py [--output-dir ./reports]
    python scripts/figure_generation/generate_extraction_report.py --viz-dir results/extraction/book1/visualizations
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.auto_report import AutoReport, ReportConfig, Theme


# ======================================================================
# Figure discovery helpers
# ======================================================================

def find_figures(viz_dir: Optional[str]) -> Dict[str, Path]:
    """Scan a visualisations directory and return a mapping of figure keys
    to file paths.  Returns an empty dict when *viz_dir* is ``None`` or
    does not exist.

    The directory layout produced by ``visualize_extraction_result()``::

        viz_dir/
        ├── inputs/<prefix>.png
        ├── binary/<prefix>.png
        ├── craft/<prefix>_{segmentation,deletion,labeled}.png
        ├── craft_merged/<prefix>_groups.png
        ├── image_components/<prefix>_{all,filtered}.png
        ├── similarities/<prefix>_{matrix,matching}.png
        ├── contour_filtering/<prefix>.png
        ├── characters/<prefix>_{segmentation,deletion,labeled}.png
        └── summary/<prefix>_complete.png
    """
    if not viz_dir:
        return {}
    vd = Path(viz_dir)
    if not vd.exists():
        return {}

    # Detect first available prefix from the inputs sub-directory
    inputs = sorted((vd / "inputs").glob("*.png")) if (vd / "inputs").exists() else []
    if not inputs:
        return {}
    prefix = inputs[0].stem

    candidates = {
        "input":                     vd / "inputs"           / f"{prefix}.png",
        "binary":                    vd / "binary"           / f"{prefix}.png",
        "craft_segmentation":        vd / "craft"            / f"{prefix}_segmentation.png",
        "craft_deletion":            vd / "craft"            / f"{prefix}_deletion.png",
        "craft_labeled":             vd / "craft"            / f"{prefix}_labeled.png",
        "craft_merged_groups":       vd / "craft_merged"     / f"{prefix}_groups.png",
        "image_components_all":      vd / "image_components" / f"{prefix}_all.png",
        "image_components_filtered": vd / "image_components" / f"{prefix}_filtered.png",
        "similarity_matrix":         vd / "similarities"     / f"{prefix}_matrix.png",
        "similarity_matching":       vd / "similarities"     / f"{prefix}_matching.png",
        "contour_filtering":         vd / "contour_filtering"/ f"{prefix}.png",
        "characters_segmentation":   vd / "characters"       / f"{prefix}_segmentation.png",
        "characters_deletion":       vd / "characters"       / f"{prefix}_deletion.png",
        "characters_labeled":        vd / "characters"       / f"{prefix}_labeled.png",
        "summary":                   vd / "summary"          / f"{prefix}_complete.png",
    }
    return {k: p for k, p in candidates.items() if p.exists()}


def _fig(report, figures: Dict[str, Path], key: str, title: str,
         placeholder: str):
    """Embed an image figure if available, else add the placeholder text."""
    if key in figures:
        report.report_img(str(figures[key]), title=title)
    else:
        report.report_text(placeholder, title=f"Figure: {title}")


# ======================================================================
# Report builder
# ======================================================================

def build_report(output_dir: str = "./reports/extraction_methodology",
                 viz_dir: Optional[str] = None):
    figures = find_figures(viz_dir)
    if figures:
        print(f"Found {len(figures)} pipeline figures in {viz_dir}")
    else:
        print("No visualisation directory provided — using figure placeholders.")

    report = AutoReport(
        title="Character Extraction Pipeline — Methodology Report",
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
    # Section 1: Introduction
    # ==================================================================
    with report.section("1. Introduction"):
        report.report_text(
            "This report documents the character extraction methodology used in "
            "the OCR pipeline for historical Chinese books. The pipeline takes "
            "scanned page images as input and produces per-character segmentation "
            "masks, bounding boxes, and downstream OCR predictions.\n\n"
            "The document covers:\n"
            "- The complete extraction pipeline (CRAFT detection → watershed "
            "segmentation → binarization → component assignment → filtering)\n"
            "- Configuration reference",
            title="Scope",
        )

    # ==================================================================
    # Section 2: Current Methodology
    # ==================================================================
    with report.section("2. Current Extraction Methodology"):
        report.report_text(
            "The extraction pipeline has two main stages: (1) character detection "
            "and segmentation (GPU), and (2) patch processing and OCR recognition "
            "(CPU + GPU). This report focuses on stage 1, which is where the "
            "character boundaries are determined.",
            title="Overview",
        )

        report.report_text(
            "**Stage 1 — Character Detection Pipeline**\n\n"
            "```\n"
            "PIL Image\n"
            "  → CRAFT neural network inference\n"
            "      → score_text (H/2, W/2)   [character region confidence]\n"
            "      → score_link (H/2, W/2)   [affinity between regions]\n"
            "  → Watershed segmentation on score_text\n"
            "      → CRAFT components (one blob per detected character)\n"
            "  → Filter CRAFT components (area, aspect ratio)\n"
            "  → Merge nearby CRAFT components (centroid distance < min_dist)\n"
            "  → Binarize original image (Otsu threshold)\n"
            "      → Image connected components (ink blobs)\n"
            "  → Filter image components (remove lines)\n"
            "  → Compute similarity: image CCs ↔ CRAFT components\n"
            "  → Assign each image CC to nearest CRAFT component\n"
            "  → Filter by contour proximity, size, area\n"
            "  → Final character components\n"
            "```",
            title="Pipeline Architecture",
        )

        _fig(report, figures, "summary",
             "Pipeline Summary",
             "*[Figure placeholder: comprehensive 3×4 pipeline summary]*")

    with report.section("2.1 CRAFT Detection"):
        report.report_text(
            "CRAFT (Character Region Awareness For Text detection) is a "
            "convolutional neural network that produces two score maps at "
            "half the input resolution:\n\n"
            "- **score_text**: per-pixel confidence that a pixel belongs to a "
            "character region (values in [0, 1])\n"
            "- **score_link**: per-pixel affinity confidence between adjacent "
            "character sub-components (values in [0, 1])\n\n"
            "The input image is preprocessed by resizing to fit within "
            "\\(\\texttt{canvas\\_size} = 1280\\) pixels (with "
            "\\(\\texttt{mag\\_ratio} = 5.0\\)) and normalizing with ImageNet "
            "statistics. The model checkpoint is "
            "``craft_mlt_25k.pth`` (multi-lingual text, 25k iterations).\n\n"
            "**Important**: In the current pipeline, ``score_link`` is computed "
            "but not used. This is relevant to the proposed improvement.",
            title="CRAFT Model Outputs",
            is_katex=True,
        )
        _fig(report, figures, "craft_segmentation",
             "CRAFT Score Maps",
             "*[Figure placeholder: CRAFT score_text and score_link heatmaps "
             "for a sample page with composite characters]*")

    with report.section("2.2 Watershed Segmentation"):
        report.report_text(
            "CRAFT components are extracted from ``score_text`` using watershed "
            "segmentation (``connectedComponent.from_image_watershed()``).\n\n"
            "The algorithm:\n"
            "1. **Binarize**: Create a foreground mask where "
            "\\(\\texttt{score\\_text} > \\tau_t\\) "
            "(``text_threshold``, default 0.6)\n"
            "2. **Find seeds**: Detect local maxima in the score map within "
            "the foreground mask (``peak_local_max``, "
            "``min_distance`` = ``min_dist``)\n"
            "3. **Watershed**: Using elevation surface \\(E = -\\texttt{score\\_text}\\), "
            "expand each seed basin within the foreground mask\n\n"
            "**Critical observation**: The same threshold \\(\\tau_t = 0.6\\) is "
            "used for both seed detection (step 2) and basin expansion mask "
            "(step 3). This means areas with moderate CRAFT scores (0.3–0.5) "
            "are excluded from the watershed entirely — they receive no CRAFT "
            "label.",
            title="Watershed Algorithm",
            is_katex=True,
        )
        _fig(report, figures, "craft_labeled",
             "Watershed Result",
             "*[Figure placeholder: Watershed segmentation result showing CRAFT "
             "component boundaries overlaid on score_text heatmap]*")

    with report.section("2.3 CRAFT Component Post-Processing"):
        report.report_text(
            "After watershed, CRAFT components are refined in two steps:\n\n"
            "**Filtering** — Remove spurious detections:\n"
            "- Components with area < ``min_area`` (18 px)\n"
            "- Components with aspect ratio outside "
            "[``min_aspect_ratio``, ``max_aspect_ratio``] "
            "= [0.2, 5.0]\n\n"
            "**Merging** — Combine nearby components:\n"
            "- Compute pairwise centroid distances in CRAFT space\n"
            "- Build a graph where edges connect components with distance "
            "< ``min_dist`` (8.0 px in CRAFT space ≈ 16 px in image space)\n"
            "- Find connected components → merge groups\n"
            "- All labels in a group are mapped to a single representative",
            title="Filtering and Merging",
        )
        _fig(report, figures, "craft_deletion",
             "CRAFT Component Filtering",
             "*[Figure placeholder: CRAFT components coloured by deletion reason]*")
        _fig(report, figures, "craft_merged_groups",
             "CRAFT Merge Groups",
             "*[Figure placeholder: merged CRAFT components with group colours]*")

    with report.section("2.4 Image Binarization & Connected Components"):
        report.report_text(
            "In parallel with CRAFT analysis, the original image is:\n\n"
            "1. **Binarized** using Otsu's method (foreground = ink, "
            "background = paper)\n"
            "2. **Connected components** extracted via "
            "``cv2.connectedComponentsWithStats(connectivity=4)``\n"
            "3. **Filtered**: elongated components with aspect ratio > 20 "
            "and major axis > 100 px are removed (these are ruling lines, "
            "not characters)",
            title="Image Processing",
        )
        _fig(report, figures, "binary",
             "Binary Image",
             "*[Figure placeholder: Otsu-binarized image]*")
        _fig(report, figures, "image_components_all",
             "All Image Connected Components",
             "*[Figure placeholder: all extracted image connected components]*")
        _fig(report, figures, "image_components_filtered",
             "Filtered Image Components",
             "*[Figure placeholder: image components after line removal]*")

    with report.section("2.5 Similarity Computation & Assignment"):
        report.report_text(
            "Each ink connected component (image CC) is assigned to the nearest "
            "CRAFT component using centroid-based distance.\n\n"
            "For each CRAFT component \\(j\\) with centroid "
            "\\(\\boldsymbol{\\mu}_j\\) and inertia tensor \\(\\Sigma_j\\), "
            "and each image CC \\(i\\) with centroid \\(\\mathbf{c}_i\\), "
            "the Mahalanobis distance is:\n\n"
            "$$d_M(i, j) = (\\mathbf{c}_i - \\boldsymbol{\\mu}_j)^T \\, "
            "\\Sigma_j^{-1} \\, "
            "(\\mathbf{c}_i - \\boldsymbol{\\mu}_j)$$\n\n"
            "The similarity is \\(s(i, j) = -d_M(i, j)\\). Each image CC is "
            "assigned to the CRAFT component with the highest similarity "
            "(lowest Mahalanobis distance). CCs with "
            "\\(\\max_j s(i,j) < \\tau_s\\) (``similarity_threshold`` = −15) "
            "are discarded as background.\n\n"
            "The inertia tensor \\(\\Sigma_j\\) encodes the shape of the CRAFT "
            "blob: for a vertically-elongated character, \\(\\Sigma_j\\) has a "
            "large eigenvalue in the vertical direction, making the Mahalanobis "
            "distance smaller vertically. This is an elliptical distance metric "
            "that naturally adapts to character shape.",
            title="Mahalanobis Distance Assignment",
            is_katex=True,
        )
        _fig(report, figures, "similarity_matrix",
             "Similarity Matrix",
             "*[Figure placeholder: similarity matrix heatmap]*")
        _fig(report, figures, "similarity_matching",
             "Similarity Matching",
             "*[Figure placeholder: Similarity matching visualization showing "
             "lines connecting image CCs to their assigned CRAFT components, "
             "color-coded by match quality]*")

    with report.section("2.6 Post-Filtering"):
        report.report_text(
            "After assignment, merged character components are filtered:\n\n"
            "1. **Contour proximity filter**: Remove small components that are "
            "too close (< 50 px) to the contour of large components "
            "(> 4000 px area). This removes artifacts near borders and seals.\n"
            "2. **Size filter**: Remove characters with bounding box outside "
            "[30, 30] — [150, 150] pixels\n"
            "3. **Minimum area filter**: Remove characters with filled area "
            "< 200 px",
            title="Character Filtering",
        )
        _fig(report, figures, "contour_filtering",
             "Contour Proximity Filtering",
             "*[Figure placeholder: contour proximity filter result]*")
        _fig(report, figures, "characters_segmentation",
             "Final Character Segmentation",
             "*[Figure placeholder: final extracted characters]*")
        _fig(report, figures, "characters_deletion",
             "Character Deletion Reasons",
             "*[Figure placeholder: characters coloured by deletion reason]*")

    # ==================================================================
    # Section 3: Configuration Reference
    # ==================================================================
    with report.section("3. Configuration Reference"):
        report.report_text(
            "All parameters are set in ``confs/extraction_pipeline.yaml`` and "
            "can be overridden via Hydra command line.\n\n"
            "**CRAFT Detection** (``craftDetectorParams``):\n\n"
            "| Parameter | Default | Description |\n"
            "|---|---|---|\n"
            "| ``mag_ratio`` | 5.0 | Image upscaling factor for CRAFT |\n"
            "| ``canvas_size`` | 1280 | Max image dimension after upscaling |\n"
            "| ``chckpt`` | craft_mlt_25k.pth | Model checkpoint |\n\n"
            "**CRAFT Component Analysis** (``craftComponentAnalysisParams``):\n\n"
            "| Parameter | Default | Description |\n"
            "|---|---|---|\n"
            "| ``text_threshold`` | 0.6 | Threshold for seed detection (high) |\n"
            "| ``mask_threshold`` | 0.3 | Threshold for watershed mask expansion (lower). "
            "Set to ``null`` for original behavior. |\n"
            "| ``link_threshold`` | null | Link score combination threshold. "
            "Set to ``null`` to disable. |\n"
            "| ``min_dist`` | 8.0 | Merge distance for nearby CRAFT components (px) |\n"
            "| ``min_area`` | 18 | Minimum CRAFT component area |\n"
            "| ``min_aspect_ratio`` | 0.2 | Minimum CRAFT component aspect ratio |\n"
            "| ``max_aspect_ratio`` | 5.0 | Maximum CRAFT component aspect ratio |\n\n"
            "**Image Component Analysis** (``imageComponentsPipelineParams``):\n\n"
            "| Parameter | Default | Description |\n"
            "|---|---|---|\n"
            "| ``threshold`` | otsu | Binarization method |\n"
            "| ``similarity_threshold`` | −10.0 | Mahalanobis distance cutoff |\n"
            "| ``similarity_metric`` | mahalanobis | Distance metric |\n"
            "| ``min_box_size`` | [30, 30] | Min character bbox [w, h] |\n"
            "| ``max_box_size`` | [150, 150] | Max character bbox [w, h] |\n"
            "| ``min_area`` | 200 | Min character filled area |\n"
            "| ``cc_filtering`` | true | Enable contour proximity filter |\n"
            "| ``cc_distance_threshold`` | 50 | Contour proximity distance |\n"
            "| ``cc_min_comp_size`` | 4000 | Min size for reference contours |",
            title="Full Parameter Table",
        )

    # ==================================================================
    # Generate
    # ==================================================================
    html_path = report.generate_html(filename="extraction_methodology.html")
    print(f"HTML report generated: {html_path}")
    print(f"Open in browser and use Ctrl+P / Cmd+P to print to PDF.")
    return html_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate extraction methodology report"
    )
    parser.add_argument(
        "--output-dir",
        default="./reports/extraction_methodology",
        help="Output directory for the report",
    )
    parser.add_argument(
        "--viz-dir",
        default=None,
        help="Path to pipeline visualisations directory "
             "(e.g. results/extraction/book1/visualizations).  When provided, "
             "real figures are embedded instead of placeholders.",
    )
    args = parser.parse_args()
    build_report(args.output_dir, args.viz_dir)
