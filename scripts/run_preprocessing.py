# scripts/run_preprocessing.py

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import logging
import sys
import os

# disable warnings
import warnings
warnings.filterwarnings("ignore")

os.environ["HYDRA_FULL_ERROR"] = "1"
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from notebook_utils.parquet_utils import save_dataframe
from src.auto_report import AutoReport, ReportConfig

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


def _add_summary_sections(report, cfg, result_df):
    """Add post-processing summary sections to the report."""

    # 1. Configuration
    with report.section("Configuration"):
        report.report_text(
            f"```yaml\n{OmegaConf.to_yaml(cfg)}\n```",
            title="Hydra Configuration",
        )

    # 2. Per-Page Statistics
    if "page" in result_df.columns:
        with report.section("Per-Page Statistics"):
            page_stats = []
            for page, grp in result_df.groupby("page"):
                row = {"page": page, "n_patches": len(grp)}
                if "conf_chat" in grp.columns:
                    confs = pd.to_numeric(grp["conf_chat"], errors="coerce")
                    row["avg_conf_chat"] = round(confs.mean(), 3)
                if "char_chat" in grp.columns:
                    n_unknown = (grp["char_chat"] == "\u25af").sum()
                    row["unknown_frac"] = round(n_unknown / len(grp), 3)
                page_stats.append(row)
            report.report_table(
                pd.DataFrame(page_stats),
                title="Per-Page Patch Counts & OCR Quality",
            )

    # 3. OCR Confidence Distribution
    if "conf_chat" in result_df.columns:
        confs = pd.to_numeric(result_df["conf_chat"], errors="coerce").dropna()
        if len(confs) > 0:
            with report.section("OCR Confidence Distribution"):
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(confs, bins=50, color="#667eea", edgecolor="white", alpha=0.8)
                ax.set_xlabel("CHAT Confidence")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of CHAT OCR Confidence Scores")
                ax.axvline(confs.median(), color="red", ls="--", alpha=0.7,
                           label=f"median = {confs.median():.3f}")
                ax.legend()
                ax.grid(alpha=0.3)
                plt.tight_layout()
                report.report_figure(fig, title="Confidence Histogram")

    # 4. Character Frequency
    if "char_chat" in result_df.columns:
        char_counts = result_df["char_chat"].value_counts().head(30)
        if len(char_counts) > 0:
            with report.section("Character Frequency"):
                freq_df = char_counts.reset_index()
                freq_df.columns = ["character", "count"]
                report.report_table(freq_df, title="Top 30 Most Common Characters")


@hydra.main(version_base=None, config_path="../confs", config_name="preprocessing")
def main(cfg: DictConfig):
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # Convert paths
    image_folder = Path(cfg.data.image_folder)
    comps_folder = Path(cfg.data.comps_folder)
    output_path = Path(cfg.data.output_path)

    # Create AutoReport for preprocessing visualization
    report_dir = output_path.parent / "reports"
    report = AutoReport(
        title="Preprocessing Report",
        output_dir=str(report_dir),
        config=ReportConfig(dpi=100, output_format='jpeg', image_quality=75),
    )

    # Instantiate - Hydra handles everything
    log.info("Instantiating components...")
    patch_preprocessor = hydra.utils.instantiate(cfg.patch_preprocessor)
    patch_preprocessor.viz_report = report

    # Process - create_dataframe is called INSIDE PatchPreprocessing.__call__
    log.info("Processing patches...")
    result_df = patch_preprocessor(image_folder, comps_folder)

    # Add summary sections to the report
    _add_summary_sections(report, cfg, result_df)

    # Generate HTML report
    html_path = report.generate_html(filename="preprocessing_report.html")
    log.info(f"Report saved to: {html_path}")

    # Save results
    log.info(f"Saving results to {output_path}")
    output_path.parent.mkdir(exist_ok=True, parents=True)

    save_dataframe(result_df, output_path)

    log.info("Done!")

if __name__ == "__main__":
    main()