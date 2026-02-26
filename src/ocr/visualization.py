"""
Comprehensive visualization utilities for the character extraction pipeline.
"""

from pathlib import Path
from typing import Union, Dict, Tuple, Optional, List
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import matplotlib.cm as cm
import cv2
import torch

from .params import PipelineOutput
from .image_components import extract_contour_lines, point_to_line_segment_distance
from ..utils import connectedComponent, Timer


# Consistent color scheme for deletion reasons across all visualizations
DELETION_COLORS = {
    # CRAFT filtering
    'area_too_small': (255, 100, 100),      # Light red
    'aspect_ratio_too_low': (255, 150, 0),   # Orange

    # Image components filtering
    'high_aspect_ratio': (200, 100, 255),    # Purple

    # Character filtering
    'too_small': (255, 100, 150),            # Pink
    'too_large': (150, 0, 150),              # Dark magenta
    'too_close_to_contour': (0, 150, 255),   # Blue
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def to_cm(arr, colormap='viridis'):
    """Convert grayscale array to colormap."""
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    
    # Ensure array is 2D
    if arr.ndim > 2:
        arr = arr.squeeze()
    
    colormap_fn = cm.get_cmap(colormap)
    colored = colormap_fn(arr)
    return (colored[:, :, :3] * 255).astype(np.uint8)


def draw_ellipses(comps, arr, color_line=(255, 0, 0), color_point=None, 
                 thickness=1, radius=2, show_labels=False):
    """Draw ellipses for each component region with optional labels."""
    arr = arr.copy()
    
    for i, region in enumerate(comps.regions):
        # Get ellipse parameters
        cy, cx = region.centroid
        orientation = region.orientation
        major_axis = region.axis_major_length
        minor_axis = region.axis_minor_length
        
        # Skip invalid ellipses
        if major_axis == 0 or minor_axis == 0:
            continue
        
        # Convert to OpenCV format
        center = (int(cx), int(cy))
        axes = (int(major_axis / 2), int(minor_axis / 2))
        angle = np.degrees(orientation)
        
        # Draw ellipse
        cv2.ellipse(
            arr,
            center=center,
            axes=axes,
            angle=angle + 90,
            startAngle=0,
            endAngle=360,
            color=color_line,
            thickness=thickness
        )
        
        # Draw center point
        pt_color = color_point
        if pt_color is None:
            if hasattr(comps, 'colors') and region.label < len(comps.colors):
                pt_color = tuple(map(int, comps.colors[region.label]))
            else:
                pt_color = color_line
        cv2.circle(arr, center, radius=radius, color=pt_color, thickness=-1)
        
        # Draw label
        if show_labels:
            cv2.putText(
                arr, str(region.label), 
                (center[0] - 10, center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
            )

    return arr


def draw_component_labels(components: connectedComponent, img: np.ndarray, 
                         font_scale=0.4, color=(255, 255, 0), thickness=1):
    """Draw label numbers on each component for debugging."""
    vis = img.copy()
    for region in components.regions:
        cy, cx = map(int, region.centroid)
        cv2.putText(
            vis, str(region.label), (cx - 10, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
        )
    return vis


def create_legend_patches(color_map: Dict[str, Tuple[int, int, int]]):
    """Create matplotlib patches for legend from color map."""
    patches = [
        Patch(facecolor=np.array(color) / 255, label=label)
        for label, color in color_map.items()
    ]
    return patches


# ============================================================================
# STAGE EXTRACTION UTILITIES
# ============================================================================

def extract_deletion_stages(components: connectedComponent):
    """
    Extract components at different stages based on deletion history.
    
    Returns dict with:
        'initial': All original components
        'deleted_by_reason': Dict mapping reasons to lists of labels
        'kept': Final kept components
    """
    all_labels = set(np.unique(components._labels))
    all_labels.discard(0)  # Remove background
    
    deleted_labels = set(components._deleted_labels)
    kept_labels = all_labels - deleted_labels

    # Group deletions by reason
    deleted_by_reason = {}
    for label, reason in components._delete_reason.items():
        if reason not in deleted_by_reason:
            deleted_by_reason[reason] = []
        deleted_by_reason[reason].append(label)
    
    return {
        'initial': all_labels,
        'deleted_by_reason': deleted_by_reason,
        'kept': kept_labels,
        'total_initial': len(all_labels),
        'total_deleted': len(deleted_labels),
        'total_kept': len(kept_labels)
    }


# ============================================================================
# MERGE VISUALIZATION
# ============================================================================

def visualize_merge_groups(components: connectedComponent, 
                          base_img: Optional[np.ndarray] = None):
    """
    Visualize which components were merged together with matching colors.
    
    Returns:
        vis: RGB visualization image
        groups: Dictionary of merge groups
        color_map: Dictionary mapping representative labels to colors
    """
    if base_img is None:
        base_img = np.zeros((*components._labels.shape, 3), dtype=np.uint8)
    else:
        base_img = base_img.copy()
    
    vis = base_img.copy()
    merged_groups = components.get_merged_groups()
    
    if not merged_groups:
        # No merges, just show regular segmentation
        return components.segm_img, {}, {}
    
    # Generate unique colors for each merge group
    n_groups = len(merged_groups)
    if n_groups <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_groups))[:, :3] * 255
    else:
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_groups))[:, :3] * 255
    
    color_map = {}
    for i, (representative, group_labels) in enumerate(merged_groups.items()):
        color = tuple(map(int, colors[i]))
        color_map[f"Group {representative}"] = color
        
        for label in group_labels:
            mask = components._labels == label
            vis[mask] = color
    
    return vis, merged_groups, color_map


def create_merge_detail_figure(components: connectedComponent, 
                               merged_groups: Dict[int, List[int]],
                               base_img: np.ndarray):
    """Create detailed figure showing each merge group separately."""
    n_groups = len(merged_groups)
    if n_groups == 0:
        return None
    
    # Resize base_img to match labels if needed
    labels_shape = components._labels.shape[:2]
    if base_img.shape[:2] != labels_shape:
        base_img = cv2.resize(base_img, (labels_shape[1], labels_shape[0]), 
                              interpolation=cv2.INTER_LINEAR)
    
    ncols = min(4, n_groups)
    nrows = (n_groups + ncols - 1) // ncols
    
    # ... rest unchanged ...    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n_groups == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (representative, group_labels) in enumerate(merged_groups.items()):
        ax = axes[idx]
        
        # Create mask for this group
        group_mask = np.zeros(components._labels.shape, dtype=bool)
        for label in group_labels:
            group_mask |= (components._labels == label)
        
        # Extract bounding box
        rows, cols = np.where(group_mask)
        if len(rows) == 0:
            ax.axis('off')
            continue
            
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        
        # Add padding
        pad = 10
        r_min = max(0, r_min - pad)
        r_max = min(base_img.shape[0], r_max + pad)
        c_min = max(0, c_min - pad)
        c_max = min(base_img.shape[1], c_max + pad)
        
        # Show group on base image
        vis = base_img.copy()
        vis[group_mask] = (0, 255, 0)
        
        # Crop to relevant area
        vis_cropped = vis[r_min:r_max, c_min:c_max]
        
        ax.imshow(vis_cropped)
        ax.set_title(f'Group {representative}\nLabels: {sorted(group_labels)}', fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_groups, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


# ============================================================================
# SIMILARITY MATRIX VISUALIZATION
# ============================================================================

def visualize_similarity_matrix(similarities: np.ndarray, 
                                image_labels: Optional[np.ndarray] = None,
                                craft_labels: Optional[np.ndarray] = None,
                                threshold: Optional[float] = None):
    """
    Create comprehensive similarity matrix visualization.
    
    Args:
        similarities: NxM similarity matrix (CRAFT x Image)
        image_labels: Labels for image components (columns)
        craft_labels: Labels for CRAFT components (rows)
        threshold: Similarity threshold to highlight
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Main heatmap
    im = axes[0].imshow(similarities, cmap='hot', aspect='auto', interpolation='nearest')
    axes[0].set_xlabel('Image Components', fontsize=12)
    axes[0].set_ylabel('CRAFT Components', fontsize=12)
    axes[0].set_title('Similarity Matrix\n(brighter = better match)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[0])
    cbar.set_label('Similarity Score', fontsize=10)
    
    # Add threshold line if provided
    if threshold is not None:
        # Highlight cells above threshold
        axes[0].contour(similarities >= threshold, levels=[0.5], colors='cyan', linewidths=2)
    
    # Add grid
    axes[0].set_xticks(np.arange(similarities.shape[1]))
    axes[0].set_yticks(np.arange(similarities.shape[0]))
    axes[0].grid(visible=True, which='both', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Best matches visualization
    if similarities.shape[1] > 0:
        best_matches = similarities.argmax(axis=0)
        best_scores = similarities.max(axis=0)
        
        x_positions = np.arange(len(best_scores))
        bars = axes[1].bar(x_positions, best_scores, color='steelblue', alpha=0.7)
        
        # Color bars by threshold
        if threshold is not None:
            for bar, score in zip(bars, best_scores):
                if score < threshold:
                    bar.set_color('lightcoral')
            axes[1].axhline(y=threshold, color='red', linestyle='--', 
                          label=f'Threshold ({threshold:.2f})', linewidth=2)
            axes[1].legend()
        
        axes[1].set_xlabel('Image Component Index', fontsize=12)
        axes[1].set_ylabel('Best Similarity Score', fontsize=12)
        axes[1].set_title('Best Match Scores per Image Component', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')
        axes[1].set_ylim([0, max(1.0, best_scores.max() * 1.1)])
    
    plt.tight_layout()
    return fig


def visualize_similarity_matching(image_components: connectedComponent,
                                 craft_components: connectedComponent,
                                 similarities: np.ndarray,
                                 base_img: np.ndarray,
                                 threshold: float = 0.5):
    """
    Visualize spatial relationships between matched components.
    
    Shows lines connecting matched image components to their CRAFT detections.
    """
    vis = base_img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    
    # Get active labels
    img_regions = [r for r in image_components.regions 
                   if not image_components.is_deleted(r.label)]
    craft_regions = list(craft_components.regions)
    
    if len(img_regions) == 0 or len(craft_regions) == 0:
        return vis
    
    # Find best matches
    best_idx = similarities.argmax(axis=0)
    best_scores = similarities.max(axis=0)
    
    # Draw connections
    for i, (img_region, score) in enumerate(zip(img_regions, best_scores)):
        if score < threshold:
            continue
        
        if best_idx[i] >= len(craft_regions):
            continue
            
        craft_region = craft_regions[best_idx[i]]
        
        # Get centroids (swap y,x to x,y)
        img_cy, img_cx = img_region.centroid
        craft_cy, craft_cx = craft_region.centroid
        
        # Draw line with color based on score
        if score > 0.8:
            color = (0, 255, 0)  # Green: excellent match
        elif score > 0.6:
            color = (255, 255, 0)  # Yellow: good match
        else:
            color = (255, 128, 0)  # Orange: weak match
            
        cv2.line(vis, 
                (int(img_cx), int(img_cy)), 
                (int(craft_cx), int(craft_cy)),
                color, 2, cv2.LINE_AA)
        
        # Draw circles at endpoints
        cv2.circle(vis, (int(img_cx), int(img_cy)), 5, (255, 0, 0), -1)  # Blue: image
        cv2.circle(vis, (int(craft_cx), int(craft_cy)), 5, (0, 0, 255), -1)  # Red: CRAFT
    
    # Add legend
    legend_y = 30
    cv2.putText(vis, "Image Component", (10, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(vis, "CRAFT Component", (10, legend_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return vis


# ============================================================================
# CONTOUR PROXIMITY VISUALIZATION
# ============================================================================

def visualize_contour_filtering(centroids_before: connectedComponent,
                                centroids_after: connectedComponent,
                                reference_components: connectedComponent,
                                distance_threshold: float,
                                min_component_size: int):
    """Contour proximity filter visualization.

    Shows the binary image as background with large-component contours
    highlighted.  Each character is drawn as a bounding-box rectangle:
    green for kept, red for removed.  Removed characters additionally get
    a dashed-style distance circle (radius = *distance_threshold*) so the
    reader can see *why* they were filtered.
    """
    ref_labels = reference_components.labels

    # Start from a light-gray rendering of the binary page
    # (binary labels > 0 → ink)
    bg = np.where(ref_labels > 0, np.uint8(200), np.uint8(245))
    vis = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)

    # ── Large reference components: outline in yellow ──
    counts = np.bincount(ref_labels.ravel())
    large_indices = np.where(
        (counts > min_component_size) & (np.arange(len(counts)) != 0)
    )[0]

    if len(large_indices) > 0:
        large_mask = np.isin(ref_labels, large_indices)
        # Semi-transparent fill to distinguish large components
        vis[large_mask] = (
            vis[large_mask].astype(np.int16) * 6 // 10
            + np.array([60, 50, 120], dtype=np.int16) * 4 // 10
        ).clip(0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            large_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(vis, contours, -1, (180, 120, 255), 2)

    # ── Recompute pairing logic (mirrors filter_centroids_by_contour_proximity) ──
    after_labels = {r.label for r in centroids_after.regions}
    active_regions = [r for r in centroids_before._regions
                      if not centroids_before.is_deleted(r.label)]
    active_labels = np.array([r.label for r in active_regions])
    centroids_arr = np.array([r.centroid for r in active_regions])

    # winner_labels: labels that survived because they were the closest to a
    # large component (i.e. the "anchor" that caused competitors to be deleted)
    winner_labels = set()
    # deleted_to_winner: maps each deleted label → (winner_cx, winner_cy)
    deleted_to_winner = {}

    if len(large_indices) > 0 and len(active_regions) > 0:
        centroids_t = torch.tensor(
            centroids_arr, dtype=torch.float32
        ).flip(-1)  # (row,col) → (x,y)

        all_lines_list, component_indices = [], []
        for idx in large_indices:
            bin_image = (ref_labels == idx).astype(np.uint8)
            lines = extract_contour_lines(bin_image)
            if len(lines) > 0:
                all_lines_list.append(lines)
                component_indices.extend([idx] * len(lines))

        if all_lines_list:
            all_lines = np.concatenate(all_lines_list, axis=0)
            all_lines_t = torch.tensor(all_lines, dtype=torch.float32)
            all_distances = point_to_line_segment_distance(centroids_t, all_lines_t)
            component_indices = np.array(component_indices)

            for idx in np.unique(component_indices):
                line_mask = component_indices == idx
                min_dists = all_distances[:, line_mask].min(dim=1).values
                close_mask = min_dists < distance_threshold
                num_close = close_mask.sum().item()
                if num_close > 1:
                    closest_idx = min_dists.argmin().item()
                    winner_label = active_labels[closest_idx]
                    winner_labels.add(winner_label)
                    w_region = active_regions[closest_idx]
                    w_cy = int(w_region.centroid[0])
                    w_cx = int(w_region.centroid[1])
                    for ci in torch.where(close_mask)[0].tolist():
                        lbl = active_labels[ci]
                        if lbl != winner_label and lbl not in after_labels:
                            deleted_to_winner[lbl] = (w_cx, w_cy)

    # ── Draw characters ──
    for region in centroids_before._regions:
        cy, cx = int(region.centroid[0]), int(region.centroid[1])
        y0, x0, y1, x1 = region.bbox
        kept = region.label in after_labels

        if kept and region.label in winner_labels:
            # Winner: kept AND was the closest to a large component
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 180, 0), 2)
            cv2.circle(vis, (cx, cy), 3, (255, 180, 0), -1)
        elif kept:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 0), 1)
            cv2.circle(vis, (cx, cy), 3, (0, 200, 0), -1)
        else:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (220, 0, 0), 2)
            cv2.line(vis, (x0, y0), (x1, y1), (220, 0, 0), 1)
            cv2.line(vis, (x0, y1), (x1, y0), (220, 0, 0), 1)
            # Dashed distance circle
            r = int(distance_threshold)
            for ang in range(0, 360, 12):
                cv2.ellipse(vis, (cx, cy), (r, r), 0, ang, ang + 6,
                            (220, 80, 80), 1, cv2.LINE_AA)
            # Line from deleted centroid to its winner
            if region.label in deleted_to_winner:
                w_cx, w_cy = deleted_to_winner[region.label]
                cv2.line(vis, (cx, cy), (w_cx, w_cy), (255, 180, 0), 1,
                         cv2.LINE_AA)

    # ── Legend (top-left) ──
    ly = 20
    cv2.rectangle(vis, (8, ly - 6), (22, ly + 6), (0, 200, 0), 1)
    cv2.putText(vis, "Kept", (28, ly + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    ly += 24
    cv2.rectangle(vis, (8, ly - 6), (22, ly + 6), (255, 180, 0), 2)
    cv2.putText(vis, "Kept (closest to large CC)", (28, ly + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    ly += 24
    cv2.rectangle(vis, (8, ly - 6), (22, ly + 6), (220, 0, 0), 2)
    cv2.putText(vis, "Filtered (fused competitor)", (28, ly + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    ly += 24
    cv2.rectangle(vis, (5, ly - 2), (25, ly + 10), (180, 120, 255), 2)
    cv2.putText(vis, f"Large component (>{min_component_size} px)", (28, ly + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return vis

# ============================================================================
# PIPELINE SUMMARY VISUALIZATION
# ============================================================================

def create_pipeline_summary(result: PipelineOutput, 
                           cc_distance_threshold: float = 50,
                           cc_min_comp_size: float = 4000):
    """
    Create a comprehensive summary figure showing all pipeline stages.
    
    Layout (3x4 grid):
    Row 1: Input → Preprocessed → Score Map → Binary
    Row 2: CRAFT Stages → Image Components → Similarity → Characters
    Row 3: Detailed Analysis Views
    """
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.25)
    
    # ========== Row 1: Initial Processing ==========
    # Input image
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(result.img_pil)
    ax.set_title('Input Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Preprocessed
    ax = fig.add_subplot(gs[0, 1])
    if result.preprocessed is not None:
        try:
            preprocessed_np = result.preprocessed[0].permute(1, 2, 0).cpu().numpy()
            if preprocessed_np.max() <= 1.0:
                preprocessed_np = (preprocessed_np * 255).astype(np.uint8)
            else:
                preprocessed_np = preprocessed_np.astype(np.uint8)
            ax.imshow(preprocessed_np)
        except:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('CRAFT Preprocessed', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Score map
    ax = fig.add_subplot(gs[0, 2])
    score_vis = to_cm(result.score_text[0].cpu().numpy(), 'hot')
    ax.imshow(score_vis)
    ax.set_title('CRAFT Score Map', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Binary
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(result.binary_img, cmap='gray')
    count = len(result.image_components.regions)
    ax.set_title(f'Binary Image\n({count} components)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # ========== Row 2: Component Processing ==========
    # CRAFT components with deletion visualization
    ax = fig.add_subplot(gs[1, 0])
    del_viz, del_colors = result.craft_components.deletion_viz(
        deleted_colors=DELETION_COLORS
    )
    ax.imshow(del_viz)
    
    # Get stage info
    stages = extract_deletion_stages(result.craft_components)
    title = f'CRAFT Components\n'
    title += f'Initial: {stages["total_initial"]} → '
    title += f'Kept: {stages["total_kept"]}'
    if stages["total_deleted"] > 0:
        title += f'\nFiltered: {stages["total_deleted"]}'
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Add mini legend
    if del_colors:
        legend_patches = create_legend_patches(del_colors)
        ax.legend(handles=legend_patches[:5], loc='upper right', fontsize=8, 
                 framealpha=0.8)
    
    # CRAFT Merged
    ax = fig.add_subplot(gs[1, 1])
    merge_viz, merge_groups, merge_colors = visualize_merge_groups(
        result.craft_components
    )
    ax.imshow(merge_viz)
    n_final = len(np.unique(result.craft_components.labels)) - 1
    n_groups = len(merge_groups)
    title = f'CRAFT After Merge\n{n_final} final'
    if n_groups > 0:
        title += f'\n{n_groups} merge groups'
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Image Components
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(result.filtered_image_components.segm_img)
    count = len(result.filtered_image_components.regions)
    ax.set_title(f'Filtered Image Components\n({count} components)', 
                fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Final Characters
    ax = fig.add_subplot(gs[1, 3])
    char_viz, char_colors = result.characters.deletion_viz(
        deleted_colors=DELETION_COLORS
    )
    ax.imshow(char_viz)
    char_stages = extract_deletion_stages(result.characters)
    title = f'Final Characters\n{char_stages["total_kept"]} kept, {char_stages["total_deleted"]} deleted'
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')

    # Add deletion-reasons legend
    if char_colors:
        char_legend = []
        for name, color in char_colors.items():
            cnt = len(char_stages['deleted_by_reason'].get(name, []))
            if name == 'kept':
                cnt = char_stages['total_kept']
            char_legend.append(
                Patch(facecolor=np.array(color) / 255,
                      label=f"{name} ({cnt})"))
        ax.legend(handles=char_legend, loc='upper right', fontsize=7,
                  framealpha=0.8)
    
    # ========== Row 3: Detailed Analysis ==========
    # Similarity matrix
    ax = fig.add_subplot(gs[2, 0])
    if result.similarity_matrix is not None and result.similarity_matrix.size > 0:
        im = ax.imshow(result.similarity_matrix, cmap='hot', aspect='auto', 
                      interpolation='nearest')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel('Image Comp.', fontsize=9)
        ax.set_ylabel('CRAFT Comp.', fontsize=9)
        ax.set_title('Similarity Matrix', fontsize=11, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No similarity data', ha='center', va='center', 
               transform=ax.transAxes)
        ax.axis('off')
    
    # Similarity matching visualization
    ax = fig.add_subplot(gs[2, 1])
    if result.similarity_matrix is not None and result.similarity_matrix.size > 0:
        # Create a base image for visualization
        base_vis = (result.binary_img * 255).astype(np.uint8)
        base_vis = cv2.cvtColor(base_vis, cv2.COLOR_GRAY2RGB)
        match_vis = visualize_similarity_matching(
            result.filtered_image_components,
            result.craft_components,
            result.similarity_matrix,
            base_vis,
            threshold=0.5
        )
        ax.imshow(match_vis)
        ax.set_title('Component Matching\n(lines show best matches)', 
                    fontsize=11, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No matching data', ha='center', va='center',
               transform=ax.transAxes)
    ax.axis('off')
    
    # Contour filtering visualization
    ax = fig.add_subplot(gs[2, 2])
    if result.characters_before_contour_filter is not None:
        contour_vis = visualize_contour_filtering(
            result.characters_before_contour_filter,
            result.characters,
            result.image_components,
            distance_threshold=cc_distance_threshold,
            min_component_size=cc_min_comp_size
        )
        ax.imshow(contour_vis)
        before_count = len(result.characters_before_contour_filter.regions)
        after_count = len(result.characters.regions)
        filtered = before_count - after_count
        ax.set_title(f'Contour Proximity Filter\n{before_count}→{after_count} ({filtered} removed)', 
                    fontsize=11, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No contour filter data', ha='center', va='center',
               transform=ax.transAxes)
    ax.axis('off')
    
    # Statistics panel
    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    
    # Compile statistics
    stats_text = "Pipeline Statistics\n" + "="*30 + "\n\n"
    
    # CRAFT stages
    craft_stages = extract_deletion_stages(result.craft_components)
    stats_text += "CRAFT Detection:\n"
    stats_text += f"  Initial: {craft_stages['total_initial']}\n"
    stats_text += f"  Filtered: {craft_stages['total_deleted']}\n"
    stats_text += f"  Merged: {len(result.craft_components.get_merged_groups())} groups\n"
    stats_text += f"  Final: {craft_stages['total_kept']}\n\n"
    
    # Deletion breakdown
    if craft_stages['deleted_by_reason']:
        stats_text += "CRAFT Filter Reasons:\n"
        for reason, labels in craft_stages['deleted_by_reason'].items():
            stats_text += f"  {reason}: {len(labels)}\n"
        stats_text += "\n"
    
    # Image components
    stats_text += "Image Components:\n"
    stats_text += f"  Extracted: {len(result.image_components.regions)}\n"
    stats_text += f"  Filtered: {len(result.filtered_image_components.regions)}\n\n"
    
    # Characters
    char_stages = extract_deletion_stages(result.characters)
    stats_text += "Character Extraction:\n"
    stats_text += f"  After similarity: {char_stages['total_initial']}\n"
    if result.characters_before_contour_filter:
        contour_filtered = (len(result.characters_before_contour_filter.regions) - 
                          len(result.characters.regions))
        stats_text += f"  Contour filtered: {contour_filtered}\n"
    stats_text += f"  Final: {char_stages['total_kept']}\n\n"
    
    # Character filter breakdown
    if char_stages['deleted_by_reason']:
        stats_text += "Character Filter Reasons:\n"
        for reason, labels in char_stages['deleted_by_reason'].items():
            stats_text += f"  {reason}: {len(labels)}\n"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Character Extraction Pipeline - Complete Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    
    return fig


# ============================================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================================

def visualize_extraction_result(result: PipelineOutput, prefix: str, 
                               output_dir: Union[Path, str], pipeline=None):
    """
    Visualize pipeline results with comprehensive analysis.
    
    Args:
        result: PipelineOutput from pipeline.forward()
        prefix: Filename prefix (e.g., 'page-001')
        output_dir: Base output directory
        pipeline: Optional pipeline object for additional visualizations
        
    Creates comprehensive folder structure with all intermediate stages.
    """
    output_dir = Path(output_dir)

    if pipeline is not None:
        cc_distance_threshold = pipeline.imageComponentsPipeline.params.cc_distance_threshold
        cc_min_comp_size = pipeline.imageComponentsPipeline.params.cc_min_comp_size
    else:
        cc_distance_threshold = 50  # fallback defaults
        cc_min_comp_size = 4000

    # Create subdirectories
    dirs = {
        'inputs': output_dir / 'inputs',
        'binary': output_dir / 'binary',
        'craft': output_dir / 'craft',
        'craft_merged': output_dir / 'craft_merged',
        'image_components': output_dir / 'image_components',
        'similarities': output_dir / 'similarities',
        'characters': output_dir / 'characters',
        'analysis': output_dir / 'analysis',
        'contour_filtering': output_dir / 'contour_filtering',
        'summary': output_dir / 'summary',
        'metadata': output_dir / 'metadata'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # print(f"Generating visualizations for {prefix}...")
    
    # ========== Save Basic Outputs ==========
    timer = Timer(prefix='-|')
    # 1. Input image
    result.img_pil.save(dirs['inputs'] / f'{prefix}.png')
    timer('Image saving {:.2f}')
    
    # 2. Binary image
    binary_pil = Image.fromarray((result.binary_img * 255).astype(np.uint8))
    binary_pil.save(dirs['binary'] / f'{prefix}.png')
    timer('Bin img saving {:.2f}')

    # 3. CRAFT visualization with deletion info
    craft_viz, craft_colors = result.craft_components.deletion_viz(
        deleted_colors=DELETION_COLORS
    )
    Image.fromarray(craft_viz).save(dirs['craft'] / f'{prefix}_deletion.png')
    timer('CRAFT viz saving {:.2f}')

    # Save regular segmentation too
    craft_segm = result.craft_components.segm_img
    Image.fromarray(craft_segm).save(dirs['craft'] / f'{prefix}_segmentation.png')
    timer('Regular segm saving {:.2f}')

    # 4. CRAFT with labels for debugging
    craft_labeled = draw_component_labels(
        result.craft_components, craft_segm
    )
    Image.fromarray(craft_labeled).save(dirs['craft'] / f'{prefix}_labeled.png')
    timer('CRAFT with labels saving {:.2f}')

    # ========== Merge Visualizations ==========
    merged_groups = result.craft_components.get_merged_groups()
    if merged_groups:
        # print(f"  Found {len(merged_groups)} merge groups")
        
        # Main merge visualization
        merge_viz, _, merge_colors = visualize_merge_groups(result.craft_components)
        Image.fromarray(merge_viz).save(dirs['craft_merged'] / f'{prefix}_groups.png')
        
        # Detailed merge figure
        merge_detail_fig = create_merge_detail_figure(
            result.craft_components, merged_groups, 
            np.array(result.img_pil)
        )
        if merge_detail_fig:
            merge_detail_fig.savefig(
                dirs['craft_merged'] / f'{prefix}_detail.png', 
                dpi=150, bbox_inches='tight'
            )
            plt.close(merge_detail_fig)
        
        # Save merge info as JSON
        merge_info = {
            'num_groups': len(merged_groups),
            'groups': {
                str(rep): sorted([int(l) for l in labels])
                for rep, labels in merged_groups.items()
            }
        }
        with open(dirs['metadata'] / f'{prefix}_merge_info.json', 'w') as f:
            json.dump(merge_info, f, indent=2)
    
    timer('Merges viz {:.2f}')

    # ========== Image Components ==========
    img_comp_viz = result.image_components.segm_img
    Image.fromarray(img_comp_viz).save(
        dirs['image_components'] / f'{prefix}_all.png'
    )

    timer('Segm components {:.2f}')

    
    filtered_img_viz = result.filtered_image_components.segm_img
    Image.fromarray(filtered_img_viz).save(
        dirs['image_components'] / f'{prefix}_filtered.png'
    )

    timer('Filtered Segm components {:.2f}')

    
    # ========== Similarity Analysis ==========
    if result.similarity_matrix is not None and result.similarity_matrix.size > 0:
        # print(f"  Generating similarity visualizations...")
        
        # Similarity matrix heatmap
        sim_fig = visualize_similarity_matrix(
            result.similarity_matrix,
            threshold=0.5  # Could be from params
        )
        sim_fig.savefig(
            dirs['similarities'] / f'{prefix}_matrix.png',
            dpi=150, bbox_inches='tight'
        )
        plt.close(sim_fig)
        
        # Spatial matching visualization
        base_vis = (result.binary_img * 255).astype(np.uint8)
        base_vis = cv2.cvtColor(base_vis, cv2.COLOR_GRAY2RGB)
        match_vis = visualize_similarity_matching(
            result.filtered_image_components,
            result.craft_components,
            result.similarity_matrix,
            base_vis,
            threshold=0.5
        )
        Image.fromarray(match_vis).save(
            dirs['similarities'] / f'{prefix}_matching.png'
        )
    
    timer('Segm matrix: {:.2f}')
    # ========== Contour Filtering ==========
    if result.characters_before_contour_filter is not None:
        # print(f"  Generating contour filter visualization...")
        
        contour_viz = visualize_contour_filtering(
            result.characters_before_contour_filter,
            result.characters,
            result.image_components,
            distance_threshold=cc_distance_threshold,
            min_component_size=cc_min_comp_size
        )
        Image.fromarray(contour_viz).save(
            dirs['contour_filtering'] / f'{prefix}.png'
        )
    
    timer('Contour filtering: {:.2f}')
    # ========== Final Characters ==========
    char_viz, char_colors = result.characters.deletion_viz(
        deleted_colors=DELETION_COLORS
    )
    # Save raw image
    Image.fromarray(char_viz).save(dirs['characters'] / f'{prefix}_segmentation_colors.png')

    # Save deletion figure with legend via matplotlib
    char_stages = extract_deletion_stages(result.characters)
    fig_del, ax_del = plt.subplots(1, 1, figsize=(12, 10))
    ax_del.imshow(char_viz)
    ax_del.axis('off')
    # Build legend patches for every reason present
    legend_patches = []
    for name, color in char_colors.items():
        count = len(char_stages['deleted_by_reason'].get(name, []))
        if name == 'kept':
            count = char_stages['total_kept']
        label = f"{name} ({count})"
        legend_patches.append(Patch(facecolor=np.array(color) / 255, label=label))
    ax_del.legend(handles=legend_patches, loc='upper right', fontsize=9,
                  framealpha=0.85, title="Deletion reasons")
    ax_del.set_title(
        f"Character Deletion Reasons — {char_stages['total_kept']} kept, "
        f"{char_stages['total_deleted']} deleted",
        fontsize=12, fontweight='bold')
    fig_del.savefig(dirs['characters'] / f'{prefix}_deletion.png',
                    dpi=150, bbox_inches='tight')
    plt.close(fig_del)

    char_segm = result.characters.segm_img
    Image.fromarray(char_segm).save(dirs['characters'] / f'{prefix}_segmentation.png')

    # Characters with labels
    char_labeled = draw_component_labels(result.characters, char_segm)
    Image.fromarray(char_labeled).save(dirs['characters'] / f'{prefix}_labeled.png')
    
    timer('Final chars: {:.2f}')

    # ========== Comprehensive Summary ==========
    # print(f"  Generating comprehensive summary...")
    summary_fig = create_pipeline_summary(result, cc_distance_threshold, cc_min_comp_size)
    summary_fig.savefig(
        dirs['summary'] / f'{prefix}_complete.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close(summary_fig)

    timer('Summary fig: {:.2f}')

    
    # ========== Save Metadata ==========
    craft_stages = extract_deletion_stages(result.craft_components)
    char_stages = extract_deletion_stages(result.characters)
    
    metadata = {
        'prefix': prefix,
        'craft': {
            'initial': craft_stages['total_initial'],
            'filtered': craft_stages['total_deleted'],
            'final': craft_stages['total_kept'],
            'merge_groups': len(merged_groups) if merged_groups else 0,
            'deletion_reasons': {
                reason: len(labels) 
                for reason, labels in craft_stages['deleted_by_reason'].items()
            }
        },
        'image_components': {
            'extracted': len(result.image_components.regions),
            'filtered': len(result.filtered_image_components.regions)
        },
        'characters': {
            'after_similarity': char_stages['total_initial'],
            'final': char_stages['total_kept'],
            'filtered': char_stages['total_deleted'],
            'deletion_reasons': {
                reason: len(labels)
                for reason, labels in char_stages['deleted_by_reason'].items()
            }
        }
    }
    
    if result.characters_before_contour_filter:
        metadata['characters']['before_contour_filter'] = len(
            result.characters_before_contour_filter.regions
        )
    
    with open(dirs['metadata'] / f'{prefix}_stats.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    timer('Metadata: {:.2f}')

    
    # print(f"✓ Visualization complete for {prefix}")
    # print(f"  Output directory: {output_dir}")