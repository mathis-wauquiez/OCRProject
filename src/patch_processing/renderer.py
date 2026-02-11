from dataclasses import dataclass
from typing import Tuple, List, Iterator
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .normalization import compute_moment

from .svg import SVG

from concurrent.futures import ProcessPoolExecutor
import os


@dataclass
class CanvasDimensions:
    """Container for canvas dimension calculations."""
    width: int
    height: int
    center_x: int
    center_y: int


class Renderer(Dataset):
    """Renders SVG images centered on their barycenters."""
    
    def __init__(
        self,
        scale: float,
        dpi: int,
        bin_thresh: float,
        pad_to_multiple: int = None,
        verbose: bool = True,
        svg_imgs: None | List[SVG] = None
    ):
        self.scale = scale
        self.dpi = dpi
        self.bin_thresh = bin_thresh
        self.pad_to_multiple = pad_to_multiple
        self.verbose = verbose
        
        self.fully_initialized = False
        self.svg_imgs = []
        self.barycenters = []
        self.canvas_dims = None

        self.cached_renders = [] 

        if svg_imgs is not None:
            self(svg_imgs)
    
    def __call__(self, svg_imgs: List[SVG]):
        """Initialize renderer with SVG images."""
        self.svg_imgs = list(svg_imgs)
        self._precompute_canvas()
        self.fully_initialized = True
        return self

    def _render_single(self, svg_object) -> np.ndarray:
        """Render a single SVG object to binary numpy array."""
        svg_rendered = svg_object.render(
            dpi=self.dpi, 
            output_format='L', 
            scale=self.scale,
            output_size=None
        )
        return svg_rendered < self.bin_thresh

    def _precompute_canvas(self):
        """Compute canvas dimensions and barycenters once."""        
        shapes, self.barycenters, self.cached_renders = self._compute_barycenters()
        extents = self._compute_extents(shapes, self.barycenters)
        self.canvas_dims = self._compute_canvas_dims(extents)
    
    def _render_with_progress(self, svg_imgs) -> Iterator[np.ndarray]:
        """Returns an iterator of rendered binary images with optional progress bar."""
        iterator = tqdm(svg_imgs, desc="Rendering", unit="img") if self.verbose else svg_imgs
        
        for svg_object in iterator:
            yield self._render_single(svg_object)

    def _compute_barycenter(self, binary_svg: np.ndarray) -> Tuple[float, float]:
        """Compute barycenter (cy_pixel, cx_pixel) of a binary image."""
        m00 = compute_moment(binary_svg, 0, 0)
        h, w = binary_svg.shape
        
        if m00 == 0:
            cy_pixel = h // 2
            cx_pixel = w // 2
        else:
            # Normalized coordinates [-1, 1]
            cy = compute_moment(binary_svg, 1, 0) / m00
            cx = compute_moment(binary_svg, 0, 1) / m00
            
            # Convert to pixel coordinates
            cy_pixel = (cy + 1) * (h - 1) / 2
            cx_pixel = (cx + 1) * (w - 1) / 2
        
        return cy_pixel, cx_pixel


    # def _compute_barycenters(self) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float]], List[np.ndarray]]:
    #     """Compute shapes and barycenters for all images, caching the renders."""
    #     barycenters = []
    #     shapes = []
    #     cached_renders = []

    #     for binary_svg in self._render_with_progress(self.svg_imgs):
    #         cy_pixel, cx_pixel = self._compute_barycenter(binary_svg)
    #         barycenters.append((cy_pixel, cx_pixel))
    #         shapes.append(binary_svg.shape)
    #         cached_renders.append(binary_svg)

    #     return shapes, barycenters, cached_renders


    def _compute_barycenters(self) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float]], List[np.ndarray]]:
        """Compute shapes and barycenters for all images, caching the renders."""

        #? This is the multicore version of the above, commented function.
        
        def process_svg(svg_obj):
            """Worker function for parallel processing."""
            binary_svg = self._render_single(svg_obj)
            cy_pixel, cx_pixel = self._compute_barycenter(binary_svg)
            return binary_svg.shape, (cy_pixel, cx_pixel), binary_svg
        
        n_cores = os.cpu_count()
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            iterator = executor.map(process_svg, self.svg_imgs)
            
            if self.verbose:
                iterator = tqdm(iterator, total=len(self.svg_imgs), desc="Rendering", unit="img")
            
            results = list(iterator)
        
        shapes, barycenters, cached_renders = zip(*results)
        return list(shapes), list(barycenters), list(cached_renders)



    def _compute_extents(
        self, 
        shapes: List[Tuple[int, int]], 
        barycenters: List[Tuple[float, float]]
    ) -> List[dict]:
        """Compute extents (distances from barycenter to edges) for all images."""
        extents = []
        for (h, w), (cy, cx) in zip(shapes, barycenters):
            extents.append({
                'left': cx,
                'right': w - cx,
                'top': cy,
                'bottom': h - cy
            })
        return extents

    def _round_up_to_multiple(self, value: float, multiple: int) -> int:
        """Round value up to nearest multiple."""
        if multiple is None or multiple <= 1:
            return int(value)
        return int(np.ceil(value / multiple) * multiple)

    def _compute_canvas_dims(self, extents: List[dict]) -> CanvasDimensions:
        """Compute final canvas dimensions with optional padding."""
        # Find maximum extents
        max_left = max(e['left'] for e in extents)
        max_right = max(e['right'] for e in extents)
        max_top = max(e['top'] for e in extents)
        max_bottom = max(e['bottom'] for e in extents)
        
        # Base canvas size
        canvas_width = int(np.ceil(max_left + max_right))
        canvas_height = int(np.ceil(max_top + max_bottom))
        
        # Apply padding to multiple if specified
        pad_left = 0
        pad_top = 0
        
        if self.pad_to_multiple is not None and self.pad_to_multiple > 1:
            new_width = self._round_up_to_multiple(canvas_width, self.pad_to_multiple)
            new_height = self._round_up_to_multiple(canvas_height, self.pad_to_multiple)
            
            pad_left = (new_width - canvas_width) // 2
            pad_top = (new_height - canvas_height) // 2
            
            canvas_width = new_width
            canvas_height = new_height
        
        center_x = int(np.ceil(max_left)) + pad_left
        center_y = int(np.ceil(max_top)) + pad_top
        
        return CanvasDimensions(canvas_width, canvas_height, center_x, center_y)
    
    # def _place_on_canvas(self, binary_svg: np.ndarray, barycenter: Tuple[float, float]) -> np.ndarray:
    #     """Place a binary image on the canvas, centered at its barycenter."""
    #     cy, cx = barycenter
    #     h, w = binary_svg.shape
        
    #     start_y = int(np.round(self.canvas_dims.center_y - cy))
    #     start_x = int(np.round(self.canvas_dims.center_x - cx))
        
    #     canvas_img = np.zeros((self.canvas_dims.height, self.canvas_dims.width))
    #     canvas_img[start_y:start_y+h, start_x:start_x+w] = binary_svg
        
    #     return canvas_img
        
    def _place_on_canvas(self, binary_svg: np.ndarray, barycenter: Tuple[float, float]) -> np.ndarray:
        """Place a binary image on the canvas, centered at its barycenter."""
        cy, cx = barycenter
        h, w = binary_svg.shape
        
        start_y = int(np.round(self.canvas_dims.center_y - cy))
        start_x = int(np.round(self.canvas_dims.center_x - cx))
        
        canvas_img = np.zeros((self.canvas_dims.height, self.canvas_dims.width))
        
        # Calculate actual end positions (may exceed canvas due to rounding)
        end_y = start_y + h
        end_x = start_x + w
        
        # Clip to canvas bounds
        src_start_y = max(0, -start_y)
        src_start_x = max(0, -start_x)
        src_end_y = h - max(0, end_y - self.canvas_dims.height)
        src_end_x = w - max(0, end_x - self.canvas_dims.width)
        
        dst_start_y = max(0, start_y)
        dst_start_x = max(0, start_x)
        dst_end_y = min(end_y, self.canvas_dims.height)
        dst_end_x = min(end_x, self.canvas_dims.width)
        
        # Place the clipped portion
        canvas_img[dst_start_y:dst_end_y, dst_start_x:dst_end_x] = \
            binary_svg[src_start_y:src_end_y, src_start_x:src_end_x]
        
        return canvas_img

    def __len__(self) -> int:
        return len(self.svg_imgs)
    
    # def __getitem__(self, idx: int) -> torch.Tensor:
    #     """Get a single rendered image by index."""
    #     if not self.fully_initialized:
    #         raise RuntimeError("Renderer must be initialized with __call__ before accessing items")
        
    #     binary_svg = self._render_single(self.svg_imgs[idx])
    #     canvas_img = self._place_on_canvas(binary_svg, self.barycenters[idx])
        
    #     return torch.from_numpy(canvas_img).float()
    

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single rendered image by index."""
        if not self.fully_initialized:
            raise RuntimeError("Renderer must be initialized with __call__ before accessing items")
        
        # Use cached render instead of re-rendering
        binary_svg = self.cached_renders[idx] 
        canvas_img = self._place_on_canvas(binary_svg, self.barycenters[idx])
        
        return torch.from_numpy(canvas_img).float()


class GridDataset(Dataset):
    def __init__(self, base_dataset, k, l):
        self.base = base_dataset
        self.k = k
        self.l = l
        self.n = k * l
    
    def __len__(self):
        # Return number of complete grids needed to cover all items
        return (len(self.base) + self.n - 1) // self.n  # Ceiling division
    
    def __getitem__(self, idx):
        # Add bounds checking
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
        
        imgs = []
        for j in range(self.k):
            row_imgs = []
            for i in range(self.l):
                img_idx = idx * self.n + j * self.l + i
                if img_idx < len(self.base):
                    row_imgs.append(self.base[img_idx])
                else:
                    # Create zero-filled array with same shape as base dataset items
                    sample_img = self.base[0]
                    zero_img = torch.zeros_like(sample_img)
                    row_imgs.append(zero_img)
            imgs.append(row_imgs)
        
        rows = []
        for row_imgs in imgs:
            row = torch.cat(row_imgs, dim=-1)
            rows.append(row)
        
        # Insert 1-pixel separators between rows
        separator = torch.ones(1, rows[0].shape[-1])  # shape [1, width]
        grid_with_sep = rows[0]
        for r in rows[1:]:
            grid_with_sep = torch.cat([grid_with_sep, separator, r], dim=0)
        
        return grid_with_sep
