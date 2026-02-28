#!/usr/bin/env python3
"""Aggregate per-page extraction statistics into a document-level summary.

Reads the per-page ``*_stats.json`` files produced by
``visualize_extraction_result()`` and sums them into a single
document-level JSON written next to the page-level metadata.

Usage:
    python scripts/aggregate_extraction_stats.py
    python scripts/aggregate_extraction_stats.py --save-folder results/extraction/book1
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def aggregate(metadata_dir: Path) -> dict:
    """Sum all ``*_stats.json`` files in *metadata_dir* into one dict."""
    files = sorted(metadata_dir.glob("*_stats.json"))
    if not files:
        print(f"No *_stats.json files found in {metadata_dir}", file=sys.stderr)
        sys.exit(1)

    craft_initial = 0
    craft_filtered = 0
    craft_final = 0
    craft_merge_groups = 0
    craft_reasons: Counter = Counter()

    img_extracted = 0
    img_filtered = 0

    char_after_sim = 0
    char_before_contour = 0
    char_final = 0
    char_filtered = 0
    char_reasons: Counter = Counter()

    for f in files:
        with open(f) as fh:
            d = json.load(fh)

        c = d["craft"]
        craft_initial += c["initial"]
        craft_filtered += c["filtered"]
        craft_final += c["final"]
        craft_merge_groups += c.get("merge_groups", 0)
        craft_reasons.update(c.get("deletion_reasons", {}))

        ic = d["image_components"]
        img_extracted += ic["extracted"]
        img_filtered += ic["filtered"]

        ch = d["characters"]
        char_after_sim += ch["after_similarity"]
        char_final += ch["final"]
        char_filtered += ch["filtered"]
        char_reasons.update(ch.get("deletion_reasons", {}))
        if "before_contour_filter" in ch:
            char_before_contour += ch["before_contour_filter"]

    summary = {
        "pages": len(files),
        "craft_detections": {
            "initial": craft_initial,
            "filtered": craft_filtered,
            "merged": craft_merge_groups,
            "final": craft_final,
            "deletion_reasons": dict(craft_reasons),
        },
        "image_components": {
            "extracted": img_extracted,
            "filtered": img_filtered,
        },
        "characters": {
            "after_similarity": char_after_sim,
            "before_contour_filter": char_before_contour,
            "final": char_final,
            "filtered": char_filtered,
            "deletion_reasons": dict(char_reasons),
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-page extraction stats into document summary")
    parser.add_argument(
        "--save-folder", type=Path,
        default="results/extraction/book1",
        help="Extraction output folder (default: results/extraction/book1)")
    args = parser.parse_args()

    save_folder = (args.save_folder if args.save_folder.is_absolute()
                   else PROJECT_ROOT / args.save_folder)
    metadata_dir = save_folder / "visualizations" / "metadata"

    if not metadata_dir.exists():
        parser.error(f"Metadata directory not found: {metadata_dir}")

    summary = aggregate(metadata_dir)

    out_path = save_folder / "document_stats.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Pretty-print to stdout
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
