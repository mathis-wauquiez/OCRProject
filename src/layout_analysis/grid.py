from torch.utils.data import Dataset
from torchvision.utils import make_grid
from ..patch_processing.renderer import Renderer


import torch
from torch.utils.data import Dataset



def get_grids(
        page_patches_df,
        nrows,
        ncols,
        dpi=256,
        scale=1
        ):
    
    dataset = Renderer(
        svg_imgs=page_patches_df['svg'].sort_index(),
        scale=scale,
        dpi=dpi,
        bin_thresh=128,
        pad_to_multiple=None,
    )

    grid_ds = GridDataset(dataset, k=nrows, l=ncols)
    return grid_ds