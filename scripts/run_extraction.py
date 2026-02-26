#!/usr/bin/env python3
"""Process images through the OCR extraction pipeline and save results.

Outputs are saved in the format expected by downstream PatchPreprocessing:
  <save_folder>/components/<image_filename>.npz
  <save_folder>/craft_components/<image_filename>.npz
  <save_folder>/visualizations/  (pipeline summary figures)
  <save_folder>/extraction_methodology.html  (auto-generated report)

Usage:
    python scripts/run_extraction.py
    python scripts/run_extraction.py --image-folder data/datasets/book1 --save-folder results/extraction/book1
    python scripts/run_extraction.py --workers 2 --config extraction_pipeline
    python scripts/run_extraction.py --no-report  # skip report generation
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from PIL import Image
from tqdm import tqdm

from src.ocr.pipeline import GlobalPipeline
from src.ocr.visualization import visualize_extraction_result
from scripts.generate_extraction_report import build_report


def setup_pipeline(config_name="extraction_pipeline"):
    """Initialize the extraction pipeline from Hydra config."""
    os.environ["HYDRA_FULL_ERROR"] = "1"

    config_dir = str(PROJECT_ROOT / "confs")
    if not Path(config_dir).exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    initialize_config_dir(config_dir=config_dir, job_name="extraction",
                          version_base=None)
    return instantiate(compose(config_name=config_name))


def process_file(file, image_folder, save_folder, pipeline):
    """Process a single image file."""
    try:
        img = Image.open(image_folder / file)
        results = pipeline.forward(img_pil=img, verbose=False)

        # Save components (format expected by processor.py:688-694)
        (save_folder / 'components').mkdir(parents=True, exist_ok=True)
        (save_folder / 'craft_components').mkdir(parents=True, exist_ok=True)
        (save_folder / 'visualizations').mkdir(parents=True, exist_ok=True)

        results.characters.save(save_folder / 'components' / f'{file}.npz')
        results.craft_components.save(
            save_folder / 'craft_components' / f'{file}.npz')

        # Visualize
        visualize_extraction_result(
            results,
            prefix=Path(file).stem,
            output_dir=save_folder / 'visualizations',
            pipeline=pipeline
        )

        return True, file
    except Exception as e:
        return False, f"{file}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Process images through OCR pipeline")
    parser.add_argument('--image-folder', type=Path,
                        default='data/datasets/book1',
                        help='Input image folder (default: data/datasets/book1)')
    parser.add_argument('--save-folder', type=Path,
                        default='results/extraction/book1',
                        help='Output folder (default: results/extraction/book1)')
    parser.add_argument('--config', default='extraction_pipeline',
                        help='Hydra config name (default: extraction_pipeline)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--no-report', action='store_true',
                        help='Skip automatic report generation')
    args = parser.parse_args()

    # Resolve paths relative to project root
    image_folder = (args.image_folder if args.image_folder.is_absolute()
                    else PROJECT_ROOT / args.image_folder)
    save_folder = (args.save_folder if args.save_folder.is_absolute()
                   else PROJECT_ROOT / args.save_folder)

    # Get files
    if not image_folder.exists():
        parser.error(f"Image folder not found: {image_folder}")

    files = sorted(f.name for f in image_folder.iterdir() if f.is_file())
    if not files:
        parser.error(f"No files found in {image_folder}")

    print(f"Found {len(files)} files")

    # Setup and process
    pipeline = setup_pipeline(args.config)
    save_folder.mkdir(parents=True, exist_ok=True)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_file, f, image_folder,
                                save_folder, pipeline)
                for f in files
            ]
            results = [future.result()
                       for future in tqdm(futures, desc="Processing")]
    else:
        results = [
            process_file(f, image_folder, save_folder, pipeline)
            for f in tqdm(files, desc="Processing")
        ]

    # Report
    successful = sum(success for success, _ in results)
    print(f"\nComplete: {successful}/{len(files)} successful")

    if failed := [info for success, info in results if not success]:
        print("Failed:")
        for info in failed:
            print(f"  {info}")

    # Generate extraction methodology report with embedded figures
    if not args.no_report and successful > 0:
        viz_dir = save_folder / 'visualizations'
        print(f"\nGenerating extraction report in {save_folder} ...")
        try:
            html_path = build_report(
                output_dir=str(save_folder),
                viz_dir=str(viz_dir),
            )
            print(f"Report saved: {html_path}")
        except Exception as e:
            print(f"Warning: report generation failed: {e}")


if __name__ == "__main__":
    main()
