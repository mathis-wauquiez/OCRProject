import numpy as np
import subprocess
import tempfile
from pathlib import Path
from typing import Union, List, Optional, Tuple, Iterator
from PIL import Image
import shutil
import io
import gc
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from omegaconf import DictConfig, ListConfig
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False

from ..patch_processing.svg import SVG


def _convert_to_tuple(value):
    """Convert list/ListConfig to tuple for PIL compatibility."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if OMEGACONF_AVAILABLE and isinstance(value, ListConfig):
        return tuple(value)
    if isinstance(value, list):
        return tuple(value)
    return value


@dataclass
class VectorizerConfig:
    """Configuration for vectorizer. Hydra-friendly."""
    executable_path: str = "./build/main"
    threshold: Optional[float] = None  # Optional threshold; None for binarized images
    smoothing_scale: float = 0.1
    accuracy_threshold: float = 0.5
    refinement_iterations: int = 0
    output_type: str = "shape_merged"
    return_svg: bool = True  # Returns SVG objects
    return_svg_string: bool = False  # Returns SVG as strings instead of objects
    return_rendered: bool = False
    save_dir: Optional[str] = None
    output_size: Optional[Tuple[int, int]] = None
    background_color: Optional[Tuple[int, int, int, int]] = (255, 255, 255, 255)
    dpi: int = 96
    scale: float = 1.0
    output_format: str = 'RGBA'
    n_jobs: int = 1
    show_progress: bool = False
    chunk_size: int = 10
    stream_results: bool = False
    stream_parallel: bool = False  # NEW: Enable parallel streaming
    
    def __post_init__(self):
        """Convert lists to tuples for PIL compatibility."""
        self.output_size = _convert_to_tuple(self.output_size)
        self.background_color = _convert_to_tuple(self.background_color)


class BinaryShapeVectorizer:
    """Memory-efficient vectorizer for binarized images."""
    
    def __init__(self, config: Union[VectorizerConfig, dict, 'DictConfig']):
        # Handle OmegaConf DictConfig
        if OMEGACONF_AVAILABLE and isinstance(config, DictConfig):
            from omegaconf import OmegaConf
            config = OmegaConf.to_container(config, resolve=True)
        
        if isinstance(config, dict):
            config = VectorizerConfig(**config)
        
        self.config = config
        
        # Resolve executable path
        exec_path = Path(config.executable_path)
        if not exec_path.is_absolute():
            exec_path = (Path(__file__).parent.resolve() / exec_path).resolve()
        if not exec_path.exists():
            raise FileNotFoundError(f"Executable not found: {exec_path}")
        self.executable_path = exec_path
        
        # Setup save directory
        if config.save_dir:
            self.save_dir = Path(config.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
        
        self._save_counter = 0
        self._validate_config()
        
        # logger.info(f"Initialized vectorizer:")
        # logger.info(f"  output_size: {self.config.output_size}")
        # logger.info(f"  background_color: {self.config.background_color}")
        # logger.info(f"  output_format: {self.config.output_format}")
    
    def _validate_config(self):
        cfg = self.config
        if not cfg.return_svg and not cfg.return_rendered and not cfg.save_dir:
            raise ValueError("Must return or save something!")
    
    def __call__(self, images, **overrides):
        return self.process(images, **overrides)
    
    def process(self, images, **overrides):
        """Process binarized images with memory management."""
        cfg = self._get_runtime_config(overrides)
        images_list = self._normalize_input(images)
        n_images = len(images_list)
        
        # logger.info(f"Processing {n_images} binarized images")
        # logger.info(f"First image shape: {images_list[0].shape}, dtype: {images_list[0].dtype}")
        
        # Stream results for memory efficiency
        if cfg.stream_results:
            return self._process_stream(images_list, cfg)
        
        # For large batches, process in chunks
        if n_images > cfg.chunk_size and cfg.n_jobs != 1:
            results = self._process_chunked(images_list, cfg)
        else:
            results = self._process_batch(images_list, cfg)
        
        return results[0] if n_images == 1 else results
    
    def _get_runtime_config(self, overrides):
        if not overrides:
            return self.config
        cfg_dict = asdict(self.config)
        cfg_dict.update(overrides)
        return VectorizerConfig(**cfg_dict)
    
    def _normalize_input(self, images):
        """Normalize input and squeeze single-channel dimensions."""
        if isinstance(images, list):
            return [self._squeeze_image(img) for img in images]
        
        if not isinstance(images, np.ndarray):
            raise TypeError(f"Unsupported type: {type(images)}")
        
        ndim = images.ndim
        # logger.info(f"Input shape: {images.shape}, ndim: {ndim}")
        
        if ndim == 2:
            return [images]
        elif ndim == 3:
            if images.shape[2] in [1, 3, 4]:
                return [self._squeeze_image(images)]
            else:
                return [self._squeeze_image(images[i]) for i in range(images.shape[0])]
        elif ndim == 4:
            return [self._squeeze_image(images[i]) for i in range(images.shape[0])]
        else:
            raise ValueError(f"Unsupported shape: {images.shape}")
    
    def _squeeze_image(self, img):
        """Squeeze single-channel dimensions (H, W, 1) -> (H, W)."""
        if img.ndim == 3 and img.shape[2] == 1:
            return np.ascontiguousarray(img[:, :, 0])
        return np.ascontiguousarray(img)
    
    def _prepare_save_paths(self, n_images):
        """Generate save paths for images."""
        if not self.save_dir:
            return [None] * n_images
        
        paths = []
        for i in range(n_images):
            path = self.save_dir / f"output_{self._save_counter:06d}.svg"
            paths.append(str(path))
            self._save_counter += 1
        
        return paths
    
    def _process_batch(self, images_list, cfg):
        """Process batch with unified progress bar."""
        save_paths = self._prepare_save_paths(len(images_list))
        
        if len(images_list) > 1 and cfg.n_jobs != 1:
            return self._process_parallel(images_list, save_paths, cfg)
        else:
            return self._process_sequential(images_list, save_paths, cfg)
    
    def _process_chunked(self, images_list, cfg):
        """Process in chunks to limit memory usage."""
        n_images = len(images_list)
        chunk_size = cfg.chunk_size
        results = []
        
        pbar = self._create_progress_bar(n_images, cfg)
        
        try:
            for start_idx in range(0, n_images, chunk_size):
                end_idx = min(start_idx + chunk_size, n_images)
                chunk = images_list[start_idx:end_idx]
                chunk_paths = self._prepare_save_paths(len(chunk))
                
                # Process chunk
                if cfg.n_jobs != 1:
                    chunk_results = self._process_parallel(
                        chunk, chunk_paths, cfg, pbar=pbar, show_internal_progress=False
                    )
                else:
                    chunk_results = self._process_sequential(
                        chunk, chunk_paths, cfg, pbar=pbar, show_internal_progress=False
                    )
                
                results.extend(chunk_results)
                
                # Force garbage collection after each chunk
                del chunk, chunk_results, chunk_paths
                gc.collect()
        finally:
            self._close_progress_bar(pbar)
        
        return results
    
    def _process_stream_parallel(self, images_list, cfg) -> Iterator:
        """Stream results with parallel processing."""
        import os
        n_workers = min(cfg.n_jobs if cfg.n_jobs > 0 else os.cpu_count(), 4)
        save_paths = self._prepare_save_paths(len(images_list))
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_process_single_image, img, path, 
                            str(self.executable_path), cfg, idx): idx
                for idx, (img, path) in enumerate(zip(images_list, save_paths))
            }
            
            # Yield as they complete with index
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    yield (idx, result)
                except Exception as e:
                    logger.error(f"Error on image {idx}: {e}")
                    yield (idx, None)
    
    def _process_stream(self, images_list, cfg) -> Iterator:
        """Stream results one at a time (generator)."""
        # Use parallel streaming if enabled and n_jobs > 1
        if cfg.stream_parallel and cfg.n_jobs != 1:
            yield from self._process_stream_parallel(images_list, cfg)
            return
        
        # Sequential streaming
        save_paths = self._prepare_save_paths(len(images_list))
        iterator = zip(enumerate(images_list), save_paths)
        
        if cfg.show_progress and TQDM_AVAILABLE:
            iterator = tqdm(iterator, total=len(images_list), desc="Vectorizing", unit="img")
        
        for (idx, img), save_path in iterator:
            try:
                result = _process_single_image(
                    img, save_path, str(self.executable_path), cfg, idx
                )
                yield (idx, result)
                
                del result
                gc.collect()
            except Exception as e:
                logger.error(f"Error on image {idx}: {e}")
                yield (idx, None)

    def _process_sequential(self, images_list, save_paths, cfg, 
                           pbar=None, show_internal_progress=True):
        """Sequential processing with unified progress bar."""
        results = []
        close_pbar = False
        
        if pbar is None and show_internal_progress:
            pbar = self._create_progress_bar(len(images_list), cfg)
            close_pbar = True
        
        try:
            for idx, (img, save_path) in enumerate(zip(images_list, save_paths)):
                try:
                    result = _process_single_image(
                        img, save_path, str(self.executable_path), cfg, idx
                    )
                    results.append(result)
                    
                    if pbar:
                        pbar.update(1)
                    
                    # Periodic cleanup
                    if len(results) % 10 == 0:
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"Error on image {idx}, shape {img.shape}: {e}")
                    results.append(None)
                    if pbar:
                        pbar.update(1)
        finally:
            if close_pbar:
                self._close_progress_bar(pbar)
        
        return results
    
    def _process_parallel(self, images_list, save_paths, cfg,
                         pbar=None, show_internal_progress=True):
        """Parallel processing with unified progress bar."""
        import os
        n_workers = min(cfg.n_jobs if cfg.n_jobs > 0 else os.cpu_count(), 
                       os.cpu_count() or 1, 4)  # Limit to 4 for memory
        
        results = [None] * len(images_list)
        close_pbar = False
        
        if pbar is None and show_internal_progress:
            pbar = self._create_progress_bar(len(images_list), cfg)
            close_pbar = True
        
        try:
            batch_size = max(n_workers * 2, 10)
            
            for batch_start in range(0, len(images_list), batch_size):
                batch_end = min(batch_start + batch_size, len(images_list))
                
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {}
                    for i in range(batch_start, batch_end):
                        future = executor.submit(
                            _process_single_image,
                            images_list[i],
                            save_paths[i],
                            str(self.executable_path),
                            cfg,
                            i
                        )
                        futures[future] = i
                    
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            results[idx] = future.result()
                        except Exception as e:
                            logger.error(f"Error on image {idx}: {e}")
                        
                        if pbar:
                            pbar.update(1)
                
                gc.collect()
        finally:
            if close_pbar:
                self._close_progress_bar(pbar)
        
        return results
    
    def _create_progress_bar(self, total, cfg):
        """Create progress bar if needed."""
        if cfg.show_progress and TQDM_AVAILABLE:
            return tqdm(total=total, desc="Vectorizing", unit="img")
        return None
    
    def _close_progress_bar(self, pbar):
        """Close progress bar if exists."""
        if pbar:
            pbar.close()


def _process_single_image(image, save_path, executable_path, config, idx=None):
    """Process single binarized image.
    
    Returns:
        - SVG object (if return_svg=True and return_svg_string=False)
        - String (if return_svg=True and return_svg_string=True)
        - Numpy array (if return_rendered=True)
        - Tuple of (SVG/string, array) (if both return_svg and return_rendered=True)
        - Save path string (if only saving to disk)
    """
    try:
        logger.debug(f"[{idx}] Processing image shape: {image.shape}, dtype: {image.dtype}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_png = tmpdir / "input.png"
            output_svg = tmpdir / "output.svg"
            
            # Save input
            _save_image(image, input_png)
            
            # Build command
            cmd = [executable_path, str(input_png)]
            
            # Add threshold flag only if provided
            if config.threshold is not None:
                cmd.extend(['-f', str(config.threshold)])
            
            # Add other parameters
            cmd.extend([
                '-s', str(config.smoothing_scale),
                '-T', str(config.accuracy_threshold),
                '-R', str(config.refinement_iterations),
                {
                    'outline': '-v', 
                    'shape': '-V',
                    'outline_merged': '-o', 
                    'shape_merged': '-O'
                }[config.output_type],
                str(output_svg)
            ])
            
            # Run subprocess
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise RuntimeError(f"Vectorization failed with code {process.returncode}: {error_msg}")
            
            if not output_svg.exists():
                raise RuntimeError(f"SVG not generated. stdout: {stdout.decode()}, stderr: {stderr.decode()}")
            
            # Read SVG
            svg_content = output_svg.read_text()
            svg_obj = SVG.load_from_string(svg_content)
            
            # Prepare SVG return value
            svg_result = None
            if config.return_svg:
                if config.return_svg_string:
                    svg_result = svg_content
                else:
                    svg_result = svg_obj
            
            # Render if needed
            rendered = None
            if config.return_rendered:
                rendered = svg_obj.render(
                    output_size=config.output_size,
                    background_color=config.background_color,
                    dpi=config.dpi,
                    scale=config.scale,
                    output_format=config.output_format
                )
            
            # Save if needed
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(output_svg, save_path)
            
            # Return based on config
            if config.return_svg and config.return_rendered:
                return (svg_result, rendered)
            elif config.return_svg:
                return svg_result
            elif config.return_rendered:
                return rendered
            else:
                return str(save_path)
                
    except Exception as e:
        logger.error(f"[{idx}] Failed processing: {e}")
        raise


def _save_image(image, path):
    """Save array as PNG."""
    try:
        if image.ndim not in [2, 3]:
            raise ValueError(f"Cannot save image with ndim={image.ndim}, shape={image.shape}")
        
        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert to PIL and save
        if image.ndim == 2:
            img = Image.fromarray(image, mode='L')
        elif image.shape[2] == 1:
            img = Image.fromarray(image[:, :, 0], mode='L')
        elif image.shape[2] == 3:
            img = Image.fromarray(image, mode='RGB')
        elif image.shape[2] == 4:
            img = Image.fromarray(image, mode='RGBA')
        else:
            raise ValueError(f"Unsupported channels: {image.shape[2]}")
        
        img.save(path)
        img.close()
        
    except Exception as e:
        logger.error(f"Failed to save image shape {image.shape}: {e}")
        raise


if __name__ == "__main__":
    import time
    
    logger.setLevel(logging.INFO)
        
    config = VectorizerConfig(
        return_svg=True,
        return_rendered=False,
        save_dir="./outputs",
        n_jobs=8,
        chunk_size=10,
        show_progress=True,
        output_format='L',
        output_size=(128, 128)
    )
    
    vectorizer = BinaryShapeVectorizer(config)
    
    # Test with binarized data (binary images)
    batch = (np.random.rand(50, 128, 128) > 0.5).astype(np.uint8) * 255
    
    print(f"Batch shape: {batch.shape}")
    print(f"Batch size: {batch.nbytes / 1024 / 1024:.1f} MB\n")
    
    start = time.time()
    results = vectorizer(batch)
    print(f"\nTime: {time.time() - start:.2f}s")
    print(f"Successful: {sum(1 for r in results if r is not None)}/{len(results)}")
    
    if results and results[0] is not None:
        print(f"Result type: {type(results[0])}")
        if hasattr(results[0], 'paths'):
            print(f"SVG has {len(results[0].paths)} paths")