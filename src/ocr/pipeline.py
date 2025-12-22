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

        self.imageComponentsPipeline = image_components.imageComponentsPipeline(imageComponentsPipelineParams, self.craftDetector)
        self.progress_callback = None

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

        self._report_progress('Running CRAFT detection...')
        img_tensor = img_tensor.permute(2,0,1).float().to(self.device)
        with torch.no_grad():
            preprocessed, score_text, score_link = self.craftDetector(img_tensor.unsqueeze(0))

        # Report CRAFT completion with score map data
        self._report_progress('CRAFT detection completed', score_text, "score_map")

        self._report_progress('Detecting connected components from CRAFT score...')
        score_text_bin = score_text.squeeze(0).cpu().numpy() > self.craftComponentAnalysisParams.text_threshold

        # components = connectedComponent.from_image(score_text_bin.astype(np.uint8), 
        #                                          connectivity=self.craftComponentAnalysisParams.connectivity)
        
        components = connectedComponent.from_image_watershed(
            score_text.squeeze(0).cpu().numpy(),
            min_distance=self.craftComponentAnalysisParams.min_dist,
            connectivity=1,
            use_intensity=True,
            compute_stats=True,
            binary_threshold=self.craftComponentAnalysisParams.text_threshold
        )

        # Report initial components found
        self._report_progress(f'Found {components.nLabels - 1} initial text components', 
                            components, "components")

        self._report_progress('Filtering components by area and aspect ratio...')
        filtered_components = craft_components.filter_craft_components(self.craftComponentAnalysisParams, components)
        
        # Report filtered components
        self._report_progress(f'After filtering: {filtered_components.nLabels - 1} components remaining', 
                            filtered_components, "components")
        
        self._report_progress('Merging nearby components...')
        merged_components = craft_components.merge_craft_components(self.craftComponentAnalysisParams, filtered_components)

        # Report merged components
        self._report_progress(f'After merging: {merged_components.nLabels - 1} final text components', 
                            merged_components, "components")

        self._report_progress('Processing image components and binarization...')
        binary_img, img_components, filtered_img_components, character_components, cc_filtered, filteredCharacters = self.imageComponentsPipeline.forward(img_pil, merged_components)

        # Report binary image
        self._report_progress('Image binarization completed', binary_img, "binary_image")
        
        # Report image components
        self._report_progress(f'Found {img_components.nLabels - 1} image components', 
                            img_components, "components")
        
        # Report filtered image components
        self._report_progress(f'After image filtering: {filtered_img_components.nLabels - 1} components', 
                            filtered_img_components, "components")
        
        # Report final character components
        self._report_progress(f'Final character segmentation: {character_components.nLabels - 1} components', 
                            character_components, "components")

        self._report_progress('Pipeline completed successfully!')

        return PipelineOutput(
            img_pil=img_pil,
            preprocessed=preprocessed,
            binary_img=binary_img,
            score_text=score_text,
            score_text_components=components,
            filtered_text_components=filtered_components,
            merged_text_components=merged_components,
            image_components=img_components,
            cc_filtered=cc_filtered,
            filtered_image_components=filtered_img_components,
            character_components=character_components,
            filteredCharacters=filteredCharacters
        )