#!/usr/bin/env python3
"""Process images through the OCR extraction pipeline and save results.

Outputs are saved in the format expected by downstream PatchPreprocessing:
  <save_folder>/components/<image_filename>.npz
  <save_folder>/craft_components/<image_filename>.npz
  <save_folder>/visualizations/  (pipeline summary figures)
  <save_folder>/extraction_methodology.html  (auto-generated report)

Uses a producer-consumer scheme: the main thread runs the GPU/CPU pipeline
(producer) while --workers threads handle saving and visualization in
parallel (consumers).  Reporting is the current bottleneck, so this
overlap hides most of the I/O and matplotlib cost.

Usage:
    python scripts/run_extraction.py
    python scripts/run_extraction.py --image-folder data/datasets/book1 --save-folder results/extraction/book1
    python scripts/run_extraction.py --workers 4 --config extraction_pipeline
    python scripts/run_extraction.py --no-report  # skip report generation
"""

import argparse
import os
import sys
import threading
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
from scripts.figure_generation.generate_extraction_report import build_report


def setup_pipeline(config_name="extraction_pipeline"):
    """Initialize the extraction pipeline from Hydra config."""
    os.environ["HYDRA_FULL_ERROR"] = "1"

    config_dir = str(PROJECT_ROOT / "confs")
    if not Path(config_dir).exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    initialize_config_dir(config_dir=config_dir, job_name="extraction",
                          version_base=None)
    return instantiate(compose(config_name=config_name))


def produce(file, image_folder, pipeline):
    """Run the extraction pipeline on a single image (GPU/CPU-bound)."""
    img = Image.open(image_folder / file)
    return pipeline.forward(img_pil=img, verbose=False)


def consume(file, result, save_folder, pipeline):
    """Save extraction results and generate visualizations (I/O-bound)."""
    result.characters.save(save_folder / 'components' / f'{file}.npz')
    result.craft_components.save(
        save_folder / 'craft_components' / f'{file}.npz')

    visualize_extraction_result(
        result,
        prefix=Path(file).stem,
        output_dir=save_folder / 'visualizations',
        pipeline=pipeline
    )


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
                        help='Number of consumer threads for reporting (default: 1)')
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

    # Setup pipeline and output directories
    pipeline = setup_pipeline(args.config)
    for sub in ('components', 'craft_components', 'visualizations'):
        (save_folder / sub).mkdir(parents=True, exist_ok=True)

    # ── Producer-consumer ─────────────────────────────────────────
    # Main thread  = producer  (runs pipeline.forward, may use GPU)
    # Thread pool  = consumers (save .npz + visualize, I/O-bound)
    # A semaphore caps how far the producer can get ahead of
    # consumers so that pipeline results don't pile up in RAM.
    pending = threading.Semaphore(args.workers)
    successful = 0
    failed = []

    def _consume(file, result):
        """Wrapper that releases the semaphore when done."""
        try:
            consume(file, result, save_folder, pipeline)
            return True, file
        except Exception as e:
            return False, f"{file}: {e}"
        finally:
            pending.release()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for file in tqdm(files, desc="Extracting"):
            # Produce: run pipeline on main thread
            try:
                result = produce(file, image_folder, pipeline)
            except Exception as e:
                failed.append(f"{file}: {e}")
                continue

            # Block if all consumer slots are busy (backpressure)
            pending.acquire()
            futures.append(pool.submit(_consume, file, result))

        # Drain consumer results
        for future in tqdm(futures, desc="Reporting"):
            ok, info = future.result()
            if ok:
                successful += 1
            else:
                failed.append(info)

    # Summary
    print(f"\nComplete: {successful}/{len(files)} successful")

    if failed:
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
