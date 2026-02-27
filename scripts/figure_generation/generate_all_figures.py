#!/usr/bin/env python3
"""
Master script — generate ALL paper figures from pipeline results.

Links together every figure-generation script so you only run one command.

Usage:
    python scripts/figure_generation/generate_all_figures.py

    Override paths:
    python scripts/figure_generation/generate_all_figures.py \
        --clustering-dir   results/clustering/book1 \
        --preprocessing-dir results/preprocessing/book1 \
        --images-dir       data/datasets/book1 \
        --output-dir       paper/figures/generated

    Generate only specific figures:
    python scripts/figure_generation/generate_all_figures.py --only glossary reverse_manuscript

    Generate glossary with ground-truth labels:
    python scripts/figure_generation/generate_all_figures.py --only glossary --label-col char_consensus
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def run_script(name, args_list, dry_run=False):
    """Run a Python script with arguments."""
    cmd = [sys.executable, str(SCRIPTS_DIR / name)] + [str(a) for a in args_list]
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return 0
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  WARNING: {name} exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Generate all paper figures from pipeline results.",
    )
    parser.add_argument(
        "--clustering-dir", type=Path,
        default=Path("results/clustering/book1"),
    )
    parser.add_argument(
        "--preprocessing-dir", type=Path,
        default=Path("results/preprocessing/book1"),
    )
    parser.add_argument(
        "--images-dir", type=Path,
        default=Path("data/datasets/book1"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("paper/figures/generated"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--label-col", type=str, default="char_consensus",
        help="Label column for glossary and experiment figures: "
             "char_chat (OCR), char_consensus, or char_transcription.",
    )
    parser.add_argument(
        "--alignment-viz-dir", type=Path, default=None,
        help="Path to alignment_viz directory with page_NNN.png images.",
    )
    parser.add_argument(
        "--alignment-page", type=int, default=None,
        help="0-based page index for alignment visualization.",
    )
    parser.add_argument(
        "--only", nargs='+', default=None,
        help="Only generate these figures. Options: "
             "experiments glossary reverse_manuscript main_figure "
             "extraction_report post_clustering_report",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = args.output_dir
    clust_dir = args.clustering_dir
    preproc_dir = args.preprocessing_dir
    images_dir = args.images_dir
    targets = set(args.only) if args.only else None

    def should_run(name):
        return targets is None or name in targets

    # ─────────────────────────────────────────────────────────────
    #  1. Experiment figures (F4-F10, LaTeX macros, ablations,
    #     discrepancy stats, t-SNE selection, alignment viz)
    # ─────────────────────────────────────────────────────────────
    if should_run('experiments'):
        experiment_args = [
            '--clustering-dir', clust_dir,
            '--output-dir', out,
            '--label-col', args.label_col,
            '--dpi', args.dpi,
        ]
        if preproc_dir.exists():
            experiment_args += ['--preprocessing-dir', preproc_dir]
        if args.alignment_viz_dir is not None:
            experiment_args += ['--alignment-viz-dir', args.alignment_viz_dir]
        elif preproc_dir.exists():
            # Auto-detect alignment_viz in preprocessing dir
            candidate = preproc_dir / 'alignment_viz'
            if candidate.exists():
                experiment_args += ['--alignment-viz-dir', candidate]
        if args.alignment_page is not None:
            experiment_args += ['--alignment-page', args.alignment_page]
        run_script('generate_experiment_figures.py', experiment_args, args.dry_run)

    # ─────────────────────────────────────────────────────────────
    #  2. Glossary figures (3 pages: most frequent, least frequent,
    #     intermediate cluster size)
    # ─────────────────────────────────────────────────────────────
    if should_run('glossary'):
        run_script('generate_glossary_figure.py', [
            '--dataframe', clust_dir / 'clustered_patches',
            '--output-dir', out,
            '--label-col', args.label_col,
            '--dpi', args.dpi,
        ], args.dry_run)

    # ─────────────────────────────────────────────────────────────
    #  3. Reverse-printed manuscript
    # ─────────────────────────────────────────────────────────────
    if should_run('reverse_manuscript'):
        run_script('generate_reverse_manuscript.py', [
            '--dataframe', clust_dir / 'clustered_patches',
            '--images-dir', images_dir,
            '--output-dir', out / 'reverse_manuscript',
            '--dpi', args.dpi,
        ], args.dry_run)

    # ─────────────────────────────────────────────────────────────
    #  4. Main pipeline figure
    # ─────────────────────────────────────────────────────────────
    if should_run('main_figure'):
        # Find the first image in the dataset
        example_image = None
        if images_dir.exists():
            for ext in ['*.jpg', '*.png', '*.tif']:
                matches = sorted(images_dir.glob(ext))
                if matches:
                    example_image = matches[0]
                    break

        if example_image is not None:
            main_fig_args = [
                '--image', example_image,
                '--dataframe', preproc_dir,
                '--output', out / 'main_pipeline.pdf',
            ]
            # Try to find the components file
            basename = example_image.name
            comps_dir = Path(f"results/extraction/{images_dir.name}")
            comps_file = comps_dir / "components" / f"{basename}.npz"
            if comps_file.exists():
                main_fig_args += ['--components', comps_file]
            run_script('generate_paper_main_figure.py', main_fig_args, args.dry_run)
        else:
            print("\n  [main_figure] Skipping: no images found")

    # ─────────────────────────────────────────────────────────────
    #  5. Extraction methodology report
    # ─────────────────────────────────────────────────────────────
    if should_run('extraction_report'):
        extraction_dir = Path(f"results/extraction/{images_dir.name}")
        viz_dir = extraction_dir / 'visualizations'
        report_args = ['--output-dir', out / 'extraction_report']
        if viz_dir.exists():
            report_args += ['--viz-dir', viz_dir]
        run_script('generate_extraction_report.py', report_args, args.dry_run)

    # ─────────────────────────────────────────────────────────────
    #  6. Post-clustering methodology report
    # ─────────────────────────────────────────────────────────────
    if should_run('post_clustering_report'):
        run_script('generate_post_clustering_report.py', [
            '--output-dir', out / 'post_clustering_report',
        ], args.dry_run)

    print(f"\n{'='*60}")
    print(f"  All figures generated in {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
