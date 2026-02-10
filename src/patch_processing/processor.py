import sys
import os
from pathlib import Path

os.environ["HYDRA_FULL_ERROR"] = "1"

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hydra.utils import instantiate


from operator import itemgetter
from PIL import Image

from notebook_utils.parquet_utils import save_dataframe
from notebook_utils.viz import show_random_sample
from notebook_utils.descriptor import compute_hog, visualize_hog

from src.utils import connectedComponent

from .ink_filter import InkFilter
from .hog import HOG
from .params import HOGParameters
from .svg import SVG
from .patch_extraction import extract_patches
from ..layout_analysis.skew import get_document_orientation

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import contextmanager

import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# type-hinting
from src.vectorization.wrapper import BinaryShapeVectorizer
from src.ocr.wrappers import OCRModel
from src.patch_processing.renderer import Renderer
from typing import List

from src.layout_analysis.parsing import ReadingOrder

## == Main Code ==
#
# Functions:
# - create_dataframe
#
# Classes:
# - PatchPreprocessing
#

def create_dataframe(ro_processor: ReadingOrder, image_folder, comps_folder, return_plots=False) -> pd.DataFrame:
    """
    Creates the dataframe containing the results from the extraction process

    Includes:
        - bin_patch | The binary image of the character
        - img_patch | The image patch
        - page      | Corresponding page of the book
        - file      | Filename of the page
        - top, left, width, height
        - label     | Label of the component
    
    """
    assert image_folder.exists()
    assert comps_folder.exists()

    files = next(os.walk(image_folder))[2]

    # main dataframe that we will manipulate in this script
    patches_df: pd.DataFrame = pd.DataFrame(columns=['bin_patch', 'img_patch', 'page', 'file', 'left', 'top', 'width', 'height', 'label', 'page_skew'])

    figs = []

    # Main loop
    for i, file in tqdm(list(enumerate(files))):
        # Load the image
        img_np = np.array(Image.open(image_folder / file))
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)[..., None]

        # Load the components
        img_comp = connectedComponent.load(comps_folder / 'components' / (str(file) + '.npz'))
        img_comp._stats = img_comp._compute_stats_from_labels(img_comp._labels)
        
        craft_comp = connectedComponent.load(comps_folder / 'craft_components' / (str(file) + '.npz'))

        # Extract the images
        _bin_patches, _img_patches = extract_patches(
            characterComponents=img_comp,
            images = [img_np],
            return_bin=True
        )

        # Retrieve the labels
        lbls = [region.label for region in img_comp.regions]
        lbls = list(filter(lambda x: not img_comp.is_deleted(x), lbls))

        # Create the dataframe for this page
        stats = img_comp.stats[1:]

        page_skew = get_document_orientation(img_np)

        page_df = pd.DataFrame({
            'bin_patch': _bin_patches,
            'img_patch': _img_patches,
            'page': i,
            'file': file,
            'left': stats[:,0],
            'top': stats[:, 1],
            'width': stats[:,2],
            'height': stats[:, 3],
            'label': lbls,
            'page_skew': page_skew
        })

        # Populate it
        if return_plots:
            canvas1 = craft_comp.segm_img
            canvas2 = np.array(Image.open(image_folder / file))
            fig = ro_processor(craft_comp.labels, page_df, canvas1, canvas2)
            figs.append((fig, canvas1, canvas2))
        
        else:
            ro_processor(craft_comp.labels, page_df)
        # Concatenate immediately
        patches_df = pd.concat([patches_df, page_df], ignore_index=True)

    if return_plots:
        return patches_df, figs, files
    return patches_df





class PatchPreprocessing:
    
    def __init__(self,
                 reading_order: ReadingOrder,
                 ink_filter: InkFilter, 
                 vectorizer: BinaryShapeVectorizer, 
                 ocr_model_configs: List[dict],
                 ocr_renderer: Renderer,
                 hog_renderer: Renderer,
                 hog_params: HOGParameters,
                 output_viz: None | Path = None,
                 verbose=True):
        
        self.ink_filter = ink_filter
        self.vectorizer = vectorizer
        self.verbose = verbose
        self.output_viz = Path(output_viz)
        self.ocr_model_configs = ocr_model_configs
        self.ocr_renderer = ocr_renderer
        self.hog_renderer = hog_renderer
        self.hog = HOG(hog_params)
        self.reading_order = reading_order

    def _print(self, *args, **kwargs):
        if self.verbose:
            return print(*args, **kwargs)

    def __call__(self, image_folder, comps_folder):

        # Reminder: defaults
        # image_folder = Path('data/datasets/book_small')
        # comps_folder = Path('data/extracted/book1-complete/components/')
        # comps_folder = Path('outputs/book_small/components/')

        # == Form the dataframe ==

        self._print('Loading the extracted characters')

        if self.output_viz is not None:
            patches_dataframe, figs, files = create_dataframe(self.reading_order, image_folder, comps_folder, return_plots=True)
            figures, canvas1, canvas2 = zip(*figs)

            for canvas, file in zip(canvas1, files):
                img = Image.fromarray(canvas)
                folder = self.output_viz / "craft_reading_order"
                folder.mkdir(exist_ok=True, parents=True)
                img.save(folder / file, quality=100)

            for canvas, file in zip(canvas2, files):
                img = Image.fromarray(canvas)
                folder = self.output_viz / "reading_order"
                folder.mkdir(exist_ok=True)
                img.save(folder / file, quality=100)

            for fig, file in zip(figures, files):
                folder = self.output_viz / "plt_reading_order"
                folder.mkdir(exist_ok=True)
                fig.savefig(folder / file)

        else:
            patches_dataframe = create_dataframe(self.reading_order, image_folder, comps_folder, return_plots=False)

        # sort by page / reading order
        patches_dataframe.sort_values(by=['page', 'reading_order'], inplace=True, na_position='last')
        patches_dataframe.reset_index(drop=True, inplace=True)

        # == Apply the ink filter ==

        self._print('Applying the ink filter')
        ink_filtered = self.ink_filter(patches_dataframe['bin_patch'])
        ink_filtered = [patch<.5 for patch in ink_filtered]

        # == Vectorize ==

        self._print('Vectorizing the images')
        vectorization_output = self.vectorizer(ink_filtered)
        if self.verbose:
            vectorization_output = tqdm(vectorization_output, total=len(ink_filtered),
                                        desc="Vectorizing images", unit="img", colour='green')
        vectorization_output = list(vectorization_output) # collect the generator

        # sort by actual patch input number - sometime parallelisation can mess things up
        vectorization_output = sorted(vectorization_output, key=itemgetter(0))
        svg_imgs = [svg_img for patch_number, svg_img in vectorization_output]
        patches_dataframe['svg'] = svg_imgs # important result, to store

        del ink_filtered # we should not need that now

        # == Deskew the SVG images ==

        for _, row in patches_dataframe.iterrows():
            from .svg import rotation
            row['svg'].apply_homography(rotation(-row['page_skew']))

        # == Use OCR models like Qwen, Tesseract, EasyOCR, ... ==

        self._print('Getting the output from the OCR models')
        ocr_renderer = self.ocr_renderer(svg_imgs)

        for item in tqdm(ocr_renderer, desc="testing the ocr renderer"):
            continue

        for ocr_partial in self.ocr_model_configs:
            with self._load_ocr_model(ocr_partial) as ocr_model:
                if hasattr(ocr_model, 'predict_with_scores'):
                    detected_characters, uncertainties = ocr_model.predict_with_scores(ocr_renderer)
                    patches_dataframe[f'unc_{ocr_model.name}'] = uncertainties
                else:
                    detected_characters = ocr_model(ocr_renderer)
                
                patches_dataframe[f'char_{ocr_model.name}'] = detected_characters

        # == Compute the HOG == 

        hog_device = self.hog._params.device

        hog_renderer = self.hog_renderer(svg_imgs)
        dataloader = DataLoader(
            hog_renderer,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        # - Preallocate the histograms -
        first_batch = next(iter(dataloader))
        sample_output = self.hog(first_batch[:1].unsqueeze(1).to(dtype=torch.float32, device=hog_device))

        total_samples = len(svg_imgs)
        histogram_shape = sample_output.histograms[0, 0].shape

        histograms = torch.zeros((total_samples, *histogram_shape), device=hog_device)
        
        # - loop on the dataset -

        if self.verbose:
            dataloader = tqdm(dataloader, desc="Computing the HOG", colour="red")

        start_idx = 0
        for batch in dataloader:
            hogOutput = self.hog(batch.unsqueeze(1).to(dtype=torch.float32, device=hog_device))
            histogram_batch = hogOutput.histograms[:, 0]
            
            batch_size = histogram_batch.shape[0]
            histograms[start_idx:start_idx + batch_size] = histogram_batch
            start_idx += batch_size

        patches_dataframe['histogram'] = list(histograms.cpu().numpy())

        return patches_dataframe


    @contextmanager
    def _load_ocr_model(self, ocr_partial):
        """Instantiate and cleanup OCR model."""
        ocr_model = ocr_partial()  # Call the partial to instantiate
        self._print(f"Loaded {ocr_model.name}")
        
        try:
            yield ocr_model
        finally:
            self._print(f"Unloading {ocr_model.name}")
            del ocr_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
