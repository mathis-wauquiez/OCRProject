import numpy as np
import subprocess
import tempfile
from pathlib import Path
from typing import Union, List, Optional, Tuple, Iterator
from PIL import Image
import shutil
import io
import gc
import traceback
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False

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
    threshold: Optional[float] = None
    smoothing_scale: float = 0.1
    accuracy_threshold: float = 0.5
    refinement_iterations: int = 0
    output_type: str = "shape_merged"
    return_svg: bool = True
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
    
    def __post_init__(self):
        """Convert lists to tuples for PIL compatibility."""
        self.output_size = _convert_to_tuple(self.output_size)
        self.background_color = _convert_to_tuple(self.background_color)


class BinaryShapeVectorizer:
    """Memory-efficient vectorizer."""
    
    def __init__(self, config: Union[VectorizerConfig, dict, 'DictConfig']):
        # Handle OmegaConf DictConfig
        if OMEGACONF_AVAILABLE and isinstance(config, DictConfig):
            from omegaconf import OmegaConf
            config = OmegaConf.to_container(config, resolve=True)
        
        if isinstance(config, dict):
            config = VectorizerConfig(**config)
        
        self.config = config
        
        exec_path = Path(config.executable_path)
        if not exec_path.is_absolute():
            exec_path = (Path(__file__).parent.resolve() / exec_path).resolve()
        if not exec_path.exists():
            raise FileNotFoundError(f"Executable not found: {exec_path}")
        self.executable_path = exec_path
        
        if config.save_dir:
            self.save_dir = Path(config.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
        
        self._save_counter = 0
        self._validate_config()
        
        logger.info(f"Initialized vectorizer:")
        logger.info(f"  output_size: {self.config.output_size}")
        logger.info(f"  background_color: {self.config.background_color}")
        logger.info(f"  output_format: {self.config.output_format}")
    
    def _validate_config(self):
        cfg = self.config
        if not cfg.return_svg and not cfg.return_rendered and not cfg.save_dir:
            raise ValueError("Must return or save something!")
        if cfg.return_rendered and not CAIROSVG_AVAILABLE:
            raise ImportError("Install cairosvg: pip install cairosvg")
    
    def __call__(self, images, **overrides):
        return self.process(images, **overrides)
    
    def process(self, images, **overrides):
        """Process images with memory management."""
        cfg = self._get_runtime_config(overrides)
        images_list = self._normalize_input(images)
        n_images = len(images_list)
        
        logger.info(f"Processing {n_images} images")
        logger.info(f"First image shape: {images_list[0].shape}, dtype: {images_list[0].dtype}")
        
        # For large batches, process in chunks
        if n_images > cfg.chunk_size and cfg.n_jobs != 1:
            return self._process_chunked(images_list, cfg)
        
        # Stream results for memory efficiency
        if cfg.stream_results:
            return self._process_stream(images_list, cfg)
        
        # Standard processing with unified progress bar
        results = self._process_batch(images_list, cfg)
        
        return results[0] if n_images == 1 else results
    
    def _process_batch(self, images_list, cfg):
        """Process batch with unified progress bar."""
        n_images = len(images_list)
        
        # Prepare thresholds and paths without progress bars
        thresholds = self._prepare_thresholds(images_list, cfg, show_progress=False)
        save_paths = self._prepare_save_paths(n_images, cfg)
        
        # Single unified progress bar
        if n_images > 1 and cfg.n_jobs != 1:
            results = self._process_parallel(images_list, thresholds, save_paths, cfg)
        else:
            results = self._process_sequential(images_list, thresholds, save_paths, cfg)
        
        return results
    
    def _process_chunked(self, images_list, cfg):
        """Process in chunks to limit memory usage."""
        n_images = len(images_list)
        chunk_size = cfg.chunk_size
        results = []
        
        # Single progress bar for all chunks
        pbar = None
        if cfg.show_progress and TQDM_AVAILABLE:
            pbar = tqdm(total=n_images, desc="Vectorizing", unit="img")
        
        try:
            for start_idx in range(0, n_images, chunk_size):
                end_idx = min(start_idx + chunk_size, n_images)
                chunk = images_list[start_idx:end_idx]
                
                # Process chunk without internal progress bars
                chunk_thresholds = self._prepare_thresholds(chunk, cfg, show_progress=False)
                chunk_paths = self._prepare_save_paths(len(chunk), cfg)
                
                if cfg.n_jobs != 1:
                    chunk_results = self._process_parallel(
                        chunk, chunk_thresholds, chunk_paths, cfg, 
                        pbar=pbar, show_internal_progress=False
                    )
                else:
                    chunk_results = self._process_sequential(
                        chunk, chunk_thresholds, chunk_paths, cfg,
                        pbar=pbar, show_internal_progress=False
                    )
                
                results.extend(chunk_results)
                
                # Force garbage collection after each chunk
                del chunk, chunk_results, chunk_thresholds, chunk_paths
                gc.collect()
        finally:
            if pbar:
                pbar.close()
        
        return results
    
    def _process_stream(self, images_list, cfg) -> Iterator:
        """Stream results one at a time (generator)."""
        n_images = len(images_list)
        thresholds = self._prepare_thresholds(images_list, cfg, show_progress=False)
        save_paths = self._prepare_save_paths(n_images, cfg)
        
        iterator = zip(enumerate(images_list), thresholds, save_paths)
        
        if cfg.show_progress and TQDM_AVAILABLE:
            iterator = tqdm(iterator, total=n_images, desc="Vectorizing", unit="img")
        
        for (idx, img), thresh, save_path in iterator:
            try:
                result = _process_single_image(
                    img, thresh, save_path, str(self.executable_path), cfg, idx
                )
                yield result
                
                # Clean up after yielding
                del result
                gc.collect()
            except Exception as e:
                logger.error(f"Error on image {idx}: {e}")
                yield None
    
    def _get_runtime_config(self, overrides):
        if not overrides:
            return self.config
        cfg_dict = asdict(self.config)
        cfg_dict.update(overrides)
        new_cfg = VectorizerConfig(**cfg_dict)
        return new_cfg
    
    def _prepare_thresholds(self, images_list, cfg, show_progress=True):
        """Calculate thresholds (optionally with progress)."""
        if cfg.threshold is not None:
            return [cfg.threshold] * len(images_list)
        
        # Calculate Otsu thresholds
        thresholds = []
        
        for idx, img in enumerate(images_list):
            try:
                thresh = _otsu_threshold(img)
                thresholds.append(thresh)
            except Exception as e:
                logger.error(f"Otsu failed on image {idx}, shape {img.shape}: {e}")
                thresholds.append(127.5)  # Fallback
        
        return thresholds
    
    def _prepare_save_paths(self, n_images, cfg):
        if not self.save_dir:
            return [None] * n_images
        
        paths = []
        for i in range(n_images):
            path = self.save_dir / f"output_{self._save_counter:06d}.svg"
            paths.append(str(path))
            self._save_counter += 1
        
        return paths
    
    def _process_sequential(self, images_list, thresholds, save_paths, cfg, 
                           pbar=None, show_internal_progress=True):
        """Sequential processing with unified progress bar."""
        results = []
        
        # Create progress bar if needed and not provided
        close_pbar = False
        if pbar is None and cfg.show_progress and show_internal_progress and TQDM_AVAILABLE:
            pbar = tqdm(total=len(images_list), desc="Vectorizing", unit="img")
            close_pbar = True
        
        try:
            for idx, (img, thresh, save_path) in enumerate(zip(images_list, thresholds, save_paths)):
                try:
                    result = _process_single_image(
                        img, thresh, save_path, str(self.executable_path), cfg, idx
                    )
                    results.append(result)
                    
                    # Update progress bar
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
            if close_pbar and pbar:
                pbar.close()
        
        return results
    
    def _process_parallel(self, images_list, thresholds, save_paths, cfg,
                         pbar=None, show_internal_progress=True):
        """Parallel processing with unified progress bar."""
        import os
        n_workers = min(cfg.n_jobs if cfg.n_jobs > 0 else os.cpu_count(), 
                       os.cpu_count() or 1)
        
        # Limit workers for memory
        n_workers = min(n_workers, 4)
        
        results = [None] * len(images_list)
        
        # Create progress bar if needed and not provided
        close_pbar = False
        if pbar is None and cfg.show_progress and show_internal_progress and TQDM_AVAILABLE:
            pbar = tqdm(total=len(images_list), desc="Vectorizing", unit="img")
            close_pbar = True
        
        try:
            # Process in smaller batches to avoid memory explosion
            batch_size = max(n_workers * 2, 10)
            
            for batch_start in range(0, len(images_list), batch_size):
                batch_end = min(batch_start + batch_size, len(images_list))
                
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {}
                    for i in range(batch_start, batch_end):
                        future = executor.submit(
                            _process_single_image,
                            images_list[i],
                            thresholds[i],
                            save_paths[i],
                            str(self.executable_path),
                            cfg,
                            i
                        )
                        futures[future] = i
                    
                    # Update progress bar as futures complete
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            results[idx] = future.result()
                        except Exception as e:
                            logger.error(f"Error on image {idx}: {e}")
                        
                        # Update progress bar
                        if pbar:
                            pbar.update(1)
                
                # Clean up after batch
                gc.collect()
        finally:
            if close_pbar and pbar:
                pbar.close()
        
        return results
    
    def _normalize_input(self, images):
        """Normalize input and squeeze single-channel dimensions."""
        if isinstance(images, list):
            return [self._squeeze_single_channel(img) for img in images]
        
        if not isinstance(images, np.ndarray):
            raise TypeError(f"Unsupported type: {type(images)}")
        
        ndim = images.ndim
        logger.info(f"Input shape: {images.shape}, ndim: {ndim}")
        
        if ndim == 2:
            return [images]
        elif ndim == 3:
            if images.shape[2] in [1, 3, 4]:
                return [self._squeeze_single_channel(images)]
            else:
                return [self._squeeze_single_channel(images[i]) for i in range(images.shape[0])]
        elif ndim == 4:
            return [self._squeeze_single_channel(images[i]) for i in range(images.shape[0])]
        else:
            raise ValueError(f"Unsupported shape: {images.shape}")
    
    def _squeeze_single_channel(self, img):
        """Squeeze single-channel dimensions (H, W, 1) -> (H, W)."""
        if img.ndim == 3 and img.shape[2] == 1:
            return img[:, :, 0]
        return img


def _otsu_threshold(image: np.ndarray) -> float:
    """Calculate Otsu threshold with error handling."""
    try:
        if not CV2_AVAILABLE:
            return 127.5
        
        # Ensure 2D grayscale
        if image.ndim == 3:
            if image.shape[2] == 3:
                img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                img_gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            elif image.shape[2] == 1:
                img_gray = image[:, :, 0]
            else:
                raise ValueError(f"Unsupported channels: {image.shape[2]}")
        elif image.ndim == 2:
            img_gray = image
        else:
            raise ValueError(f"Unsupported ndim: {image.ndim}, shape: {image.shape}")
        
        # Ensure uint8
        if img_gray.dtype != np.uint8:
            if img_gray.max() <= 1.0:
                img_gray = (img_gray * 255).astype(np.uint8)
            else:
                img_gray = img_gray.astype(np.uint8)
        
        # Calculate Otsu
        thresh, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(thresh)
        
    except Exception as e:
        logger.error(f"Otsu threshold failed for shape {image.shape}: {e}")
        return 127.5


def _process_single_image(image, threshold, save_path, executable_path, config, idx=None):
    """Process single image with detailed error logging."""
    try:
        logger.debug(f"[{idx}] Processing image shape: {image.shape}, dtype: {image.dtype}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_png = tmpdir / "input.png"
            output_svg = tmpdir / "output.svg"
            
            # Save input
            _save_image(image, input_png)
            
            # Build command
            cmd = [
                executable_path, str(input_png),
                '-f', str(threshold),
                '-s', str(config.smoothing_scale),
                '-T', str(config.accuracy_threshold),
                '-R', str(config.refinement_iterations),
                {
                    'outline': '-v', 'shape': '-V',
                    'outline_merged': '-o', 'shape_merged': '-O'
                }[config.output_type],
                str(output_svg)
            ]
            
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
            
            # Render if needed
            rendered = None
            if config.return_rendered:
                rendered = _render_svg(
                    svg_content, image.shape, config.output_size,
                    config.background_color, config.dpi, config.scale, config.output_format
                )
            
            # Save if needed
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(output_svg, save_path)
            
            # Return
            if config.return_svg and config.return_rendered:
                return (svg_content, rendered)
            elif config.return_svg:
                return svg_content
            elif config.return_rendered:
                return rendered
            else:
                return str(save_path)
                
    except Exception as e:
        logger.error(f"[{idx}] Failed processing: {e}")
        raise


def _render_svg(svg_content, orig_shape, output_size, bg_color, dpi, scale, fmt):
    """Render SVG with cleanup and error handling."""
    try:
        # Convert to tuples if needed (for Hydra compatibility)
        output_size = _convert_to_tuple(output_size)
        bg_color = _convert_to_tuple(bg_color)
        
        if output_size:
            w, h = output_size
        else:
            if len(orig_shape) >= 2:
                h, w = orig_shape[:2]
            else:
                raise ValueError(f"Invalid shape for rendering: {orig_shape}")
        
        w, h = int(w * scale), int(h * scale)
        
        # Render
        png_data = cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            output_width=w, output_height=h, dpi=dpi
        )
        
        # Load and convert
        img = Image.open(io.BytesIO(png_data))
        
        if bg_color:
            if not isinstance(bg_color, tuple):
                bg_color = tuple(bg_color) if hasattr(bg_color, '__iter__') else bg_color
            
            bg = Image.new('RGBA', img.size, bg_color)
            bg.paste(img, (0, 0), img)
            img.close()
            img = bg
        
        if fmt != 'RGBA':
            converted = img.convert(fmt)
            img.close()
            img = converted
        
        # Convert to array
        arr = np.array(img)
        img.close()
        
        del png_data
        return arr
        
    except Exception as e:
        logger.error(f"Rendering failed: {e}")
        raise


def _save_image(image, path):
    """Save array as PNG with cleanup and error handling."""
    try:
        # Ensure 2D or 3D
        if image.ndim not in [2, 3]:
            raise ValueError(f"Cannot save image with ndim={image.ndim}, shape={image.shape}")
        
        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert to PIL
        if image.ndim == 2:
            img = Image.fromarray(image, mode='L')
        elif image.ndim == 3:
            if image.shape[2] == 1:
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
    
    print(f"OpenCV: {CV2_AVAILABLE}, Cairo: {CAIROSVG_AVAILABLE}\n")
    
    config = VectorizerConfig(
        return_svg=False,
        return_rendered=True,
        save_dir="./outputs",
        n_jobs=8,
        chunk_size=10,
        show_progress=True,
        output_format='L',
        output_size=(128, 128)
    )
    
    vectorizer = BinaryShapeVectorizer(config)
    
    # Test with simple data
    batch = np.random.randint(0, 255, (50, 128, 128), dtype=np.uint8)
    
    print(f"Batch shape: {batch.shape}")
    print(f"Batch size: {batch.nbytes / 1024 / 1024:.1f} MB\n")
    
    start = time.time()
    results = vectorizer(batch)
    print(f"\nTime: {time.time() - start:.2f}s")
    print(f"Successful: {sum(1 for r in results if r is not None)}/{len(results)}")
    
    if results and results[0] is not None:
        print(f"Result shape: {results[0].shape}")
