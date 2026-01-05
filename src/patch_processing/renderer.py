
from .normalization import compute_moment
from . import filter_binary_patch
from torch.utils.data import Dataset
import numpy as np
import torch

class Renderer(Dataset):
    """Renders SVG images centered on their barycenters."""
    
    def __init__(
        self,
        svg_imgs,
        min_size_black,
        min_size_white,
        scale,
        bin_thresh=128,
        dpi=64,
        tqdm=True,
        pad_to_multiple=None
    ):
        self.svg_imgs = list(svg_imgs)
        self.min_size_black = min_size_black
        self.min_size_white = min_size_white
        self.scale = scale
        self.bin_thresh = bin_thresh
        self.dpi = dpi
        self.use_tqdm = tqdm
        self.pad_to_multiple = pad_to_multiple
        
        # Precompute canvas dimensions
        self._precompute_canvas(svg_imgs)

        print(f'Canvas dimensions: w={self.canvas_width}, h={self.canvas_height}, cx={self.center_x}, cy={self.center_y}')
    
    def _precompute_canvas(self, svg_imgs):
        """Compute canvas dimensions and barycenters once."""
        from tqdm import tqdm
        
        shapes, barycenters = self.compute_barycenters(svg_imgs)
        extents = self.compute_extents(shapes, barycenters)
        self.canvas_dims = self.compute_canvas_dims(extents)
        self.barycenters = barycenters
        
        self.canvas_width, self.canvas_height, self.center_x, self.center_y = self.canvas_dims
    
    def filter_images(self, svg_imgs):
        """Returns an iterator of filtered images."""
        from tqdm import tqdm
        
        it = tqdm(svg_imgs, desc="Rendering", unit="img") if self.use_tqdm else svg_imgs

        for svg_object in it:
            svg_rendered = svg_object.render(
                dpi=self.dpi, 
                output_format='L', 
                scale=self.scale,
                output_size=None
            )
            
            binary_svg = svg_rendered < self.bin_thresh
            
            filtered = filter_binary_patch(
                binary_svg, 
                min_size_black=self.min_size_black, 
                min_size_white=self.min_size_white, 
            )
            
            yield filtered

    def compute_barycenters(self, svg_imgs):
        barycenters = []
        shapes = []

        for filtered in self.filter_images(svg_imgs):
            m00 = compute_moment(filtered, 0, 0)
            if m00 == 0:
                cy, cx = filtered.shape[0] // 2, filtered.shape[1] // 2
            else:
                cy = compute_moment(filtered, 1, 0) / m00
                cx = compute_moment(filtered, 0, 1) / m00
            
            h, w = filtered.shape
            cy_pixel = (cy + 1) * (h - 1) / 2
            cx_pixel = (cx + 1) * (w - 1) / 2

            barycenters.append((cy_pixel, cx_pixel))
            shapes.append((h, w))

        return shapes, barycenters

    def compute_extents(self, shapes, barycenters):
        extents = []
        for (h, w), (cy, cx) in zip(shapes, barycenters):
            left = cx
            right = w - cx
            top = cy
            bottom = h - cy
            
            extents.append({
                'left': left,
                'right': right,
                'top': top,
                'bottom': bottom
            })
        return extents

    def _round_up_to_multiple(self, value, multiple):
        if multiple is None or multiple <= 1:
            return value
        return int(np.ceil(value / multiple) * multiple)

    def compute_canvas_dims(self, extents):
        max_left = max(e['left'] for e in extents)
        max_right = max(e['right'] for e in extents)
        max_top = max(e['top'] for e in extents)
        max_bottom = max(e['bottom'] for e in extents)
        
        canvas_width = int(np.ceil(max_left)) + int(np.ceil(max_right))
        canvas_height = int(np.ceil(max_top)) + int(np.ceil(max_bottom))

        if self.pad_to_multiple is not None and self.pad_to_multiple > 1:
            new_width = self._round_up_to_multiple(canvas_width, self.pad_to_multiple)
            new_height = self._round_up_to_multiple(canvas_height, self.pad_to_multiple)
            
            pad_x = new_width - canvas_width
            pad_y = new_height - canvas_height
            
            pad_left = pad_x // 2
            pad_top = pad_y // 2
            
            canvas_width = new_width
            canvas_height = new_height
        else:
            pad_left = 0
            pad_top = 0

        center_x = int(np.ceil(max_left)) + pad_left
        center_y = int(np.ceil(max_top)) + pad_top
        
        return canvas_width, canvas_height, center_x, center_y
    
    def __len__(self):
        return len(self.svg_imgs)
    
    def __getitem__(self, idx):
        """Get a single rendered image by index."""
        svg_object = self.svg_imgs[idx]
        
        # Render and filter
        svg_rendered = svg_object.render(
            dpi=self.dpi, 
            output_format='L', 
            scale=self.scale,
            output_size=None
        )
        
        binary_svg = svg_rendered < self.bin_thresh
        
        filtered = filter_binary_patch(
            binary_svg, 
            min_size_black=self.min_size_black, 
            min_size_white=self.min_size_white, 
        )
        
        # Paste on canvas
        cy, cx = self.barycenters[idx]
        h, w = filtered.shape
        start_y = int(np.round(self.center_y - cy))
        start_x = int(np.round(self.center_x - cx))
        
        canvas_img = np.zeros((self.canvas_height, self.canvas_width))
        canvas_img[start_y:start_y+h, start_x:start_x+w] = filtered
        
        # Convert to tensor
        return torch.from_numpy(canvas_img).float()
