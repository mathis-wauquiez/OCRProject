# scripts/run_preprocessing.py

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import logging
import sys
import os

os.environ["HYDRA_FULL_ERROR"] = "1"
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from notebook_utils.parquet_utils import save_dataframe

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../confs", config_name="preprocessing")
def main(cfg: DictConfig):
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))
    
    # Convert paths
    image_folder = Path(cfg.data.image_folder)
    comps_folder = Path(cfg.data.comps_folder)
    output_path = Path(cfg.data.output_path)
    
    # Instantiate - Hydra handles everything
    log.info("Instantiating components...")
    patch_preprocessor = hydra.utils.instantiate(cfg.patch_preprocessor)
    
    # Process - create_dataframe is called INSIDE PatchPreprocessing.__call__
    log.info("Processing patches...")
    result_df = patch_preprocessor(image_folder, comps_folder)
    
    # Save results
    log.info(f"Saving results to {output_path}")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    save_dataframe(result_df, output_path)
    
    log.info("Done!")

if __name__ == "__main__":
    main()