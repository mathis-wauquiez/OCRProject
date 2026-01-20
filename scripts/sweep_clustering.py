#!/usr/bin/env python3
"""
Run graph clustering sweep analysis.

Usage:
    python run_clustering_sweep.py
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import logging
import sys
import os

os.environ["HYDRA_FULL_ERROR"] = "1"
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from notebook_utils.parquet_utils import save_dataframe, load_dataframe

import pandas as pd

from src.clustering.clustering_sweep import graphClusteringSweep

logger = logging.getLogger(__name__)


@hydra.main(config_path="../confs", config_name="clustering", version_base=None)
def main(cfg: DictConfig):
    """Run clustering sweep."""
    
    # Load data
    logger.info(f"Loading data from {cfg.data.input_path}")
    input_path = Path(cfg.data.input_path)
    
    dataframe = load_dataframe(input_path)

    logger.info(f"Loaded {len(dataframe)} patches")
    
    # Create clustering sweep
    sweep = graphClusteringSweep(
        feature=cfg.method.feature,
        epsilons=cfg.method.epsilons,
        gammas=cfg.method.gammas,
        target_lbl=cfg.data.target_lbl,
        edges_type=cfg.method.edges_type,
        metric=cfg.method.metric,
        keep_reciprocal=cfg.method.keep_reciprocal,
        device=cfg.method.device,
        output_dir=cfg.data.output_path
    )
    
    # Run sweep
    logger.info("Starting clustering sweep...")
    dataframe, filtered_dataframe, label_representatives_dataframe = sweep(dataframe)
    
    # Generate HTML report
    logger.info("Generating HTML report...")
    html_path = sweep.generate_html()
    logger.info(f"HTML report saved to: {html_path}")
    
    # Optionally generate PDF (can be slow)
    # pdf_path = sweep.generate_pdf()
    # logger.info(f"PDF report saved to: {pdf_path}")
    
    logger.info("Clustering sweep complete!")
    logger.info(f"Report: file://{html_path.absolute()}")
    
    # save the three dataframes
    output_path = Path(cfg.data.output_path)
    save_dataframe(dataframe, output_path / "clustered_patches")
    save_dataframe(filtered_dataframe, output_path / "filtered_patches")
    save_dataframe(label_representatives_dataframe, output_path / "label_representatives")


if __name__ == "__main__":
    main()