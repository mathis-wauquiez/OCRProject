from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_pipeline(results, save_dir='saved_figures/pipeline_viz', filename='output', 
                      save_summary=True, save_deltas=True, save_stages=False):
    """
    Visualize character detection pipeline results.
    
    Args:
        results: PipelineOutput object with all intermediate results
        save_dir: Base directory for saving visualizations
        filename: Base filename (without extension)
        save_summary: Whether to save summary figure
        save_deltas: Whether to save delta visualizations
        save_stages: Whether to save detailed stage-by-stage figures
    
    Returns:
        dict: Paths to saved figures
    """
    
    def create_diff_overlay(labels_before, labels_after, base_image):
        """Show what changed between two label maps"""
        removed = np.isin(labels_before, np.setdiff1d(np.unique(labels_before), np.unique(labels_after)))
        kept = labels_after > 0
        
        overlay = np.array(base_image).copy() if len(base_image.shape) == 3 else np.stack([base_image]*3, axis=-1)
        overlay[removed] = overlay[removed] * 0.3 + np.array([255, 0, 0]) * 0.7  # Red for removed
        overlay[kept] = overlay[kept] * 0.7 + np.array([0, 255, 0]) * 0.3  # Green for kept
        
        return overlay.astype(np.uint8)
    
    # Setup directories
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}
    
    base_img = np.array(results.img_pil)
    
    # 1. Summary Figure
    if save_summary:
        fig = plt.figure(figsize=(20, 12))
        
        stages = [
            ('CRAFT Initial', results.score_text_components),
            ('CRAFT Merged', results.merged_text_components),
            ('Image Components', results.filtered_image_components),
            ('Characters', results.character_components),
            ('Final', results.filteredCharacters),
        ]
        
        for idx, (title, cc) in enumerate(stages):
            ax = plt.subplot(2, 3, idx+1)
            ax.imshow(cc.segm_img)
            ax.set_title(f'{title}\n{cc.nLabels-1} components', fontsize=12, weight='bold')
            ax.axis('off')
        
        # Statistics subplot
        ax = plt.subplot(2, 3, 6)
        ax.axis('off')
        
        stats = f"""
Pipeline Flow:

{results.score_text_components.nLabels-1:4d} CRAFT detections
  ↓ filter
{results.filtered_text_components.nLabels-1:4d} after filtering
  ↓ merge
{results.merged_text_components.nLabels-1:4d} after merging

{results.image_components.nLabels-1:4d} image components
  ↓ filter lines
{results.filtered_image_components.nLabels-1:4d} kept
  ↓ associate
{results.character_components.nLabels-1:4d} characters
  ↓ filter invalid
{results.filteredCharacters.nLabels-1:4d} final characters

Retention: {100*(results.filteredCharacters.nLabels-1)/(results.score_text_components.nLabels-1):.1f}%
        """
        
        ax.text(0.1, 0.5, stats, fontsize=14, family='monospace', 
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        summary_path = save_dir / f'{filename}_summary.png'
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files['summary'] = summary_path
    
    # 2. Delta Visualizations
    if save_deltas:
        deltas_dir = save_dir / 'deltas'
        deltas_dir.mkdir(exist_ok=True)
        
        delta_configs = [
            ('craft_filter', 'CRAFT Filtering', 
             results.score_text_components, results.filtered_text_components,
             results.score_text_components.segm_img),
            
            ('craft_merge', 'CRAFT Merging',
             results.filtered_text_components, results.merged_text_components,
             results.filtered_text_components.segm_img),
            
            ('img_filter', 'Image Component Filtering',
             results.image_components, results.filtered_image_components,
             base_img),
            
            ('char_filter', 'Character Filtering',
             results.character_components, results.filteredCharacters,
             base_img),
        ]
        
        for key, title, cc_before, cc_after, bg_img in delta_configs:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            
            overlay = create_diff_overlay(cc_before.labels, cc_after.labels, bg_img)
            
            ax.imshow(overlay)
            removed_count = cc_before.nLabels - cc_after.nLabels
            ax.set_title(f'{title}\n{cc_before.nLabels-1} → {cc_after.nLabels-1} ({removed_count} removed)\nRed=removed, Green=kept', 
                        fontsize=14)
            ax.axis('off')
            
            plt.tight_layout()
            delta_path = deltas_dir / f'{filename}_{key}.png'
            plt.savefig(delta_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            saved_files[f'delta_{key}'] = delta_path
    
    # 3. Detailed Stage Visualizations (optional)
    if save_stages:
        stages_dir = save_dir / 'stages'
        stages_dir.mkdir(exist_ok=True)
        
        stage_configs = [
            ('craft', 'CRAFT Pipeline', [
                ('initial', results.score_text_components),
                ('filtered', results.filtered_text_components),
                ('merged', results.merged_text_components),
            ]),
            
            ('image', 'Image Components', [
                ('all', results.image_components),
                ('filtered', results.filtered_image_components),
            ]),
            
            ('characters', 'Character Detection', [
                ('detected', results.character_components),
                ('final', results.filteredCharacters),
            ]),
        ]
        
        for key, title, components in stage_configs:
            n_cols = len(components)
            fig, axes = plt.subplots(1, n_cols, figsize=(7*n_cols, 7))
            if n_cols == 1:
                axes = [axes]
            
            for ax, (name, cc) in zip(axes, components):
                ax.imshow(cc.segm_img)
                ax.set_title(f'{name}\n{cc.nLabels-1} components', fontsize=14)
                ax.axis('off')
            
            fig.suptitle(title, fontsize=16, weight='bold')
            plt.tight_layout()
            
            stage_path = stages_dir / f'{filename}_{key}.png'
            plt.savefig(stage_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            saved_files[f'stage_{key}'] = stage_path
    
    return saved_files




import torch
import cv2
from src.ocr.params import PipelineOutput
from src.ocr.pipeline import GlobalPipeline

def score_to_image(pipeline: GlobalPipeline,results: PipelineOutput, cx_score, cy_score, minor_axis_score, major_axis_score):
    orig_w, orig_h = results.img_pil.width, results.img_pil.height

    ratio = min(pipeline.craftDetector.params.mag_ratio * max(orig_h, orig_w), pipeline.craftDetector.params.canvas_size) / max(orig_h, orig_w)
    target_h, target_w = int(orig_h * ratio), int(orig_w * ratio)

    # Transform coordinates: score → preprocessed → cropped → original
    
    # Step 1: Score to preprocessed (2x upscale)
    cy_prep = cy_score * 2
    cx_prep = cx_score * 2
    major_axis_prep = major_axis_score * 2
    minor_axis_prep = minor_axis_score * 2
    
    
    # Scale back to original
    cy_orig = cy_prep * orig_h / target_h
    cx_orig = cx_prep * orig_w / target_w
    major_axis_orig = major_axis_prep * (orig_h / target_h + orig_w / target_w) / 2
    minor_axis_orig = minor_axis_prep * (orig_h / target_h + orig_w / target_w) / 2

    return cx_orig, cy_orig, minor_axis_orig, major_axis_orig



def draw_ellipses(pipeline, results, comps, arr, score=False, color_line=(255,0,0), color_point=None, thickness=1, radius=1):
    # Calculate transformations (same as proper_overlay)
    mag_ratio = pipeline.craftDetector.params.mag_ratio
    canvas_size = pipeline.craftDetector.params.canvas_size
    arr = arr.copy()
    
    # Scaling ratio that preserves aspect ratio
    tensor = torch.tensor(np.array(results.img_pil)).permute(2,0,1)[None] / 255
    orig_h, orig_w = tensor.shape[2:]

    ratio = min(mag_ratio * max(orig_h, orig_w), canvas_size) / max(orig_h, orig_w)
    target_h, target_w = int(orig_h * ratio), int(orig_w * ratio)
    
    # Draw ellipses for each region
    for i, region in enumerate(comps.regions):
        # Get ellipse parameters in CRAFT score space
        cy, cx = region.centroid  # In score map coordinates
        orientation = region.orientation
        major_axis = region.axis_major_length
        minor_axis= region.axis_minor_length
        
        if score:
            cx, cy, minor_axis, major_axis = score_to_image(pipeline,results,cx,cy,minor_axis,major_axis)
        
        # Convert to OpenCV format
        center = (int(cx), int(cy))
        axes = (int(major_axis / 2), int(minor_axis / 2))
        angle = np.degrees(orientation)
        
        # Draw ellipse
        cv2.ellipse(
            arr,
            center=center,
            axes=axes,
            angle=angle+90,
            startAngle=0,
            endAngle=360,
            color=color_line,
            thickness=thickness
        )
        
        # Draw center point
        if color_point is None:
            color_point = comps.colors[i]
            color_point = tuple(map(int, color_point))
        cv2.circle(arr, center, radius=radius, color=color_point, thickness=-1)

    return arr



import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

def proper_overlay(results, model, original_shape):
    """
    Properly account for 32-alignment, aspect ratio, and coordinate mapping
    """
    score = np.array(results.score_text.cpu())[0]  # (H_score, W_score)
    img_pil = results.img_pil
    
    # Get original image dimensions
    orig_h, orig_w = original_shape
    
    # Step 1: Understand the transformations that were applied
    # The score comes from preprocessed image at half resolution
    preprocessed_h, preprocessed_w = score.shape[0] * 2, score.shape[1] * 2
    
    # Step 2: Calculate the actual transformations
    mag_ratio = model.params.mag_ratio
    canvas_size = model.params.canvas_size
    
    # Calculate the ratio used in preprocessing (preserves aspect ratio)
    ratio = min(mag_ratio * max(orig_h, orig_w), canvas_size) / max(orig_h, orig_w)
    target_h, target_w = int(orig_h * ratio), int(orig_w * ratio)
    
    # Step 3: The image was padded to nearest multiple of 32
    target_h_32, target_w_32 = (target_h + 31) // 32 * 32, (target_w + 31) // 32 * 32
    
    print(f"Original: ({orig_h}, {orig_w})")
    print(f"Scaled: ({target_h}, {target_w})") 
    print(f"Padded to: ({target_h_32}, {target_w_32})")
    print(f"Score shape (1/2 res): {score.shape}")
    
    # Step 4: Create proper coordinate mapping
    # Score coordinates → Preprocessed coordinates → Original coordinates
    
    # First, upsample score to preprocessed resolution
    score_preprocessed = F.interpolate(
        torch.tensor(score).float().unsqueeze(0).unsqueeze(0),
        size=(preprocessed_h, preprocessed_w),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()
    
    # Step 5: Remove padding and rescale back to original coordinates
    # Crop to the actual scaled image area (remove padding)
    score_cropped = score_preprocessed[:target_h, :target_w]
    
    # Resize back to original dimensions (this handles the aspect ratio preservation)
    score_original = F.interpolate(
        torch.tensor(score_cropped).float().unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()
    
    # Step 6: Create overlay
    fig = plt.figure(figsize=(16,12))
    
    # Original image
    # ax1.imshow(img_pil)
    # ax1.set_title('Original Image')
    # ax1.axis('off')
    
    # Proper overlay
    plt.imshow(img_pil, alpha=0.8)
    plt.imshow(score_original, alpha=0.6, cmap='viridis')
    plt.axis('off')
    
    # plt.colorbar(label='Segmentation Score')
    plt.tight_layout()
    # plt.show()
    
    return fig