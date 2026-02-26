#!/usr/bin/env python3
"""
Run graph clustering sweep analysis.

Usage:
    python run_clustering_sweep.py
"""

import hydra
from hydra.utils import instantiate
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
from src.ocr.wrappers import is_chinese_character, simplified_to_traditional
from tqdm import tqdm

logger = logging.getLogger(__name__)


@hydra.main(config_path="../confs", config_name="clustering", version_base=None)
def main(cfg: DictConfig):
    """Run clustering sweep."""
    
    # Load data
    logger.info(f"Loading data from {cfg.data.input_path}")
    input_path = Path(cfg.data.input_path)
    
    dataframe = load_dataframe(input_path)

    logger.info(f"Loaded {len(dataframe)} patches")
    
    # Instantiate reporter and refinement steps from config
    reporter = instantiate(cfg.reporter)
    refinement_steps = [instantiate(s) for s in cfg.refinement_steps]

    # Create clustering sweep
    sweep = graphClusteringSweep(
        feature=cfg.method.feature,
        epsilons=cfg.method.epsilons,
        gammas=cfg.method.gammas,
        cell_sizes=cfg.method.cell_sizes,
        normalization_methods=cfg.method.normalization_methods,
        grdt_sigmas=cfg.method.grdt_sigmas,
        nums_bins=cfg.method.nums_bins,
        target_lbl=cfg.data.target_lbl,
        edges_type=cfg.method.edges_type,
        metric=cfg.method.metric,
        keep_reciprocal=cfg.method.keep_reciprocal,
        device=cfg.method.device,
        reporter=reporter,
        refinement_steps=refinement_steps,
        # ── Post-clustering refinement ──
        enable_chat_split=cfg.method.get('enable_chat_split', False),
        chat_split_purity_threshold=cfg.method.get('chat_split_purity_threshold', 0.90),
        chat_split_min_size=cfg.method.get('chat_split_min_size', 3),
        chat_split_min_label_count=cfg.method.get('chat_split_min_label_count', 2),
        enable_hapax_association=cfg.method.get('enable_hapax_association', False),
        hapax_min_confidence=cfg.method.get('hapax_min_confidence', 0.3),
        hapax_max_dissimilarity=cfg.method.get('hapax_max_dissimilarity', None),
        enable_glossary=cfg.method.get('enable_glossary', False),
    )
    
    # Run sweep
    logger.info("Starting clustering sweep...")
    dataframe, filtered_dataframe, label_representatives_dataframe, graph, partition = sweep(dataframe)
    
    logger.info("Clustering sweep complete!")
    
    # save the three dataframes
    output_path = Path(cfg.data.output_path)
    save_dataframe(dataframe, output_path / "clustered_patches")
    save_dataframe(filtered_dataframe, output_path / "filtered_patches")
    save_dataframe(label_representatives_dataframe, output_path / "label_representatives")

    import pickle
    pickle.dump(graph, open(output_path / 'graph.gpickle', 'wb'))

    # Save glossary if it was generated
    if cfg.method.get('enable_glossary', False):
        from src.clustering.post_clustering import build_glossary
        glossary_df = build_glossary(dataframe)
        glossary_df.to_csv(output_path / "glossary.csv", index=False)
        save_dataframe(glossary_df, output_path / "glossary")
        logger.info(f"Glossary saved: {len(glossary_df)} entries")

    # Generate HTML report (single call, includes all sections)
    html_path = sweep.reporter.generate_html()
    logger.info(f"Report: file://{html_path.absolute()}")


if __name__ == "__main__":
    main()