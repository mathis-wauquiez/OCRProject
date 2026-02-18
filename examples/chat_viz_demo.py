#!/usr/bin/env python3
"""
Demo: CHAT OCR Pipeline Visualization

This script demonstrates how to enable visualizations for the CHAT character
extraction pipeline using AutoReport.

The visualizations show:
  1. Grayscale crop (raw subcolumn extraction)
  2. Binarized image (PIL mode "1" after threshold 128)
  3. Predictions overlaid with synthetic baseline + confidence colors

Usage:
    python examples/chat_viz_demo.py

The report will be saved to ./reports/chat_pipeline_viz.html
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.auto_report import AutoReport, ReportConfig, Theme
from src.patch_processing.processor import PatchPreprocessing

# Initialize AutoReport for visualization
config = ReportConfig(
    theme=Theme.DEFAULT,
    show_progress=True,
    auto_close_figures=True,
)

report = AutoReport(
    title="CHAT OCR Pipeline Visualization",
    author="OCR Processing System",
    output_dir="./reports",
    config=config,
)

# Add introduction section
with report.section("Overview"):
    report.report_text("""
# CHAT Character Extraction Pipeline

This report visualizes the preprocessing steps for CHAT OCR:

1. **Grayscale Crop**: Tight bounding box around CRAFT-detected characters
2. **Binarization**: Threshold at 128 to convert to PIL mode "1" (black & white)
3. **Baseline + Predictions**:
   - Blue line: Synthetic baseline (vertical at median x-coord)
   - Green dashed: Bounding polygon
   - Red X marks: CRAFT centroids
   - Colored text: Predictions with confidence-based colors
     - 🟢 Green: High confidence (≥0.8)
     - 🟠 Orange: Medium confidence (0.5-0.8)
     - 🔴 Red: Low confidence (<0.5)

---

**Note**: Only the first 3 subcolumns per page are visualized to keep the report size manageable.
    """, title="Introduction")

# TODO: Initialize PatchPreprocessing with viz_report parameter
#
# Example:
# processor = PatchPreprocessing(
#     reading_order=reading_order,
#     ink_filter=ink_filter,
#     vectorizer=vectorizer,
#     chat_model=chat_model,
#     hog_renderer=hog_renderer,
#     hog_params=hog_params,
#     viz_report=report,  # <-- Enable visualizations
# )
#
# # Run processing (visualizations will be automatically added to the report)
# result_df = processor(image_folder, comps_folder)

# Generate HTML report
print("\n" + "=" * 60)
print("Generating visualization report...")
print("=" * 60)

html_path = report.generate_html(filename="chat_pipeline_viz.html")

print(f"\n✓ Report saved to: {html_path}")
print(f"\nOpen in browser:")
print(f"  file://{html_path.absolute()}")
print("\nKeyboard shortcuts:")
print("  - Ctrl+E: Expand all sections")
print("  - Ctrl+W: Collapse all sections")
print()

report.summary()
