from .detection.model_wrapper import craftWrapper
from .params import craftParams, craftComponentsParams, imageComponentsParams, PipelineOutput
from .. import utils
from ..utils import connectedComponent

from . import craft_components
from . import image_components

import torch
import numpy as np

from typing import Callable, Optional, Any


class GlobalPipeline:

    def __init__(self,
                 craftDetectorParams = craftParams(),
                 craftComponentAnalysisParams = craftComponentsParams(),
                 imageComponentsPipelineParams = imageComponentsParams(),
                 device='cpu'
                ):
        
        self.craftDetector = craftWrapper(craftDetectorParams)
        self.craftDetector.eval()
        self.craftDetector.model.eval()
        self.craftComponentAnalysisParams = craftComponentAnalysisParams
        self.craftDetector.to(device)
        self.device = device

        self.imageComponentsPipeline = image_components.imageComponentsPipeline(
            imageComponentsPipelineParams, self.craftDetector
        )
        self.progress_callback = None
        self._verbose = False

    def set_progress_callback(self, callback: Optional[Callable[[str, Any, str], None]]):
        """Set a callback function to report progress with intermediate data"""
        self.progress_callback = callback

    def _report_progress(self, message: str, intermediate_data=None, data_type=None):
        """Report progress through callback or print"""
        if self.progress_callback:
            self.progress_callback(message, intermediate_data, data_type)
        elif self._verbose:
            print(message)

    def _print(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    def forward(self, img_pil, verbose=True):
        self._verbose = verbose

        img_tensor = torch.tensor(np.array(img_pil))
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(-1)

        H, W, C = img_tensor.shape
        self._report_progress(f'Image dimensions: {W} x {H} x {C}')

        # ============ CRAFT Detection ============
        self._report_progress('Running CRAFT detection...')
        img_tensor = img_tensor.permute(2,0,1).float().to(self.device)
        with torch.no_grad():
            preprocessed, score_text, score_link = self.craftDetector(img_tensor.unsqueeze(0))

        self._report_progress('CRAFT detection completed', score_text, "score_map")

        # ============ Score Map Preparation ============
        score_text_np = score_text.squeeze(0).cpu().numpy()
        score_link_np = score_link.squeeze(0).cpu().numpy()

        # Optionally combine text and link scores before watershed
        score_for_watershed = score_text_np
        if self.craftComponentAnalysisParams.link_threshold is not None:
            self._report_progress('Combining text and link scores...')
            score_for_watershed = craft_components.combine_text_link_scores(
                score_text_np, score_link_np,
                self.craftComponentAnalysisParams.link_threshold
            )
            self._report_progress(
                'Text+Link scores combined',
                score_for_watershed, "score_map"
            )

        # ============ CRAFT Component Extraction ============
        self._report_progress('Detecting connected components from CRAFT score...')

        text_components = connectedComponent.from_image_watershed(
            score_for_watershed,
            min_distance=self.craftComponentAnalysisParams.min_dist,
            connectivity=1,
            use_intensity=True,
            compute_stats=True,
            binary_threshold=self.craftComponentAnalysisParams.text_threshold,
            mask_threshold=self.craftComponentAnalysisParams.mask_threshold,
        )

        initial_count = len(text_components.regions)
        self._report_progress(
            f'Found {initial_count} initial text components', 
            text_components, "components"
        )

        # ============ CRAFT Filtering ============
        self._report_progress('Filtering components by area...')
        text_components = craft_components.filter_craft_components(
            self.craftComponentAnalysisParams, text_components
        )

        filtered_count = len(text_components.regions)
        deleted_count = len(text_components._deleted_labels)
        self._report_progress(
            f'After filtering: {filtered_count} components remaining ({deleted_count} filtered out)',
            text_components, "components"
        )

        # ============ Image Processing ============
        self._report_progress('Processing image components and binarization...')
        
        # Run the full image components pipeline
        image_result = self.imageComponentsPipeline.forward(
            img_pil, text_components, 
            return_intermediate=True
        )

        # Report progress for each stage
        self._report_progress('Image binarization completed', image_result.binary_img, "binary_image")
        
        img_comp_count = len(image_result.img_components.regions)
        self._report_progress(
            f'Found {img_comp_count} image components', 
            image_result.img_components, "components"
        )
        
        filtered_img_count = len(image_result.filtered_img_components.regions)
        self._report_progress(
            f'After image filtering: {filtered_img_count} components', 
            image_result.filtered_img_components, "components"
        )
        
        char_count = len(image_result.characters.regions)
        self._report_progress(
            f'Final character segmentation: {char_count} components', 
            image_result.characters, "components"
        )

        self._report_progress('Pipeline completed successfully!')

        return PipelineOutput(
            img_pil=img_pil,
            preprocessed=preprocessed,
            binary_img=image_result.binary_img,
            score_text=score_text,
            score_link=score_link,
            craft_components=text_components,
            image_components=image_result.img_components,
            filtered_image_components=image_result.filtered_img_components,
            characters=image_result.characters,
            similarity_matrix=image_result.similarity_matrix,
            characters_before_contour_filter=image_result.characters_before_contour_filter
        )