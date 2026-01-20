
from src.patch_processing.hog import HOG, load_or_compute_hog, setup_cache
from pathlib import Path

import torch
import tqdm
from einops import rearrange



def compute_hog(hog_params, files, image_folder, comps_folder):
    hog = HOG(hog_params)

    # Setup
    cache_folder = Path('caches/hog_cache')
    setup_cache(cache_folder, hog_params)

    hist_list = []


    patches = {
        'orientation':  [], # ~ 1Go
        'magnitude':    [],
        'raw':          []
    }
    patches_device = 'cpu'
    patches_dtype  = torch.float16

    # Main loop
    for file in tqdm.tqdm(files):
        hog_output = load_or_compute_hog(file, cache_folder, hog, image_folder, comps_folder)
        histograms = rearrange(hog_output.histograms, 'Npatch C Ncells Nbins -> Npatch (C Ncells) Nbins')
        hist_list.append(histograms)
        if 'orientation' in patches:
            patches['orientation'].append(hog_output.patches_grdt_orientation.to(device=patches_device, dtype=patches_dtype))
        if 'raw' in patches:
            patches['raw'] += [t.to(device=patches_device, dtype=patches_dtype) for t in hog_output.patches_image]
        if 'magnitude' in patches:
            patches['magnitude'].append(hog_output.patches_grdt_magnitude.to(device=patches_device, dtype=patches_dtype))
        
    histograms = torch.cat(hist_list, dim=0)
    for  k in set(patches.keys()):
        patches[k] = torch.cat(patches[k], dim=0)

    return patches, histograms



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def viz_only_hog(hog_params, histograms1, histograms2, img, patch_idx=0, 
                 threshold=0.01, arrow_scale=0.45, figsize=(20, 10)):

    # Extract parameters
    cw = hog_params.cell_width
    ch = hog_params.cell_height
    Nw = img.shape[-1] // cw
    Nh = img.shape[-2] // ch 
    Nbins = histograms1.shape[-1]
    
    # Reshape histograms to spatial grid
    scales1 = rearrange(histograms1, 'Npatchs (Nh Nw) Nbins -> Npatchs Nh Nw Nbins', 
                        Nh=Nh, Nw=Nw).cpu()
    scales2 = rearrange(histograms2, 'Npatchs (Nh Nw) Nbins -> Npatchs Nh Nw Nbins', 
                        Nh=Nh, Nw=Nw).cpu()
    
    # Select the specified patch
    selected_histograms1 = scales1[patch_idx]
    selected_histograms2 = scales2[patch_idx]
    
    print(f"Visualizing patch {patch_idx}")
    print(f"Grid: {Nh} x {Nw} cells, each with {Nbins} orientation bins")
    print(f"Cell size: {ch} x {cw} pixels")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # UNSIGNED: angles from 0 to pi
    bin_angles = np.linspace(0, np.pi, Nbins, endpoint=False)
    
    # Calculate global max magnitude across both histograms for consistent scaling
    max_magnitude = max(selected_histograms1.max(), selected_histograms2.max())
    min_threshold = threshold * max_magnitude
    
    # Plot both histograms
    _plot_hog_on_axis(ax1, img, selected_histograms1, Nh, Nw, ch, cw, 
                      bin_angles, max_magnitude, min_threshold, arrow_scale, 
                      "HOG Visualization 1")
    _plot_hog_on_axis(ax2, img, selected_histograms2, Nh, Nw, ch, cw, 
                      bin_angles, max_magnitude, min_threshold, arrow_scale, 
                      "HOG Visualization 2")
    
    plt.tight_layout()
    return fig


def _plot_hog_on_axis(ax, img, histograms, Nh, Nw, ch, cw, 
                      bin_angles, max_magnitude, min_threshold, arrow_scale, title):

    # Display background image
    ax.imshow(img, cmap='gray', alpha=0.3)
    
    # Draw arrows for each cell
    for cell_y in range(Nh):
        for cell_x in range(Nw):
            # Calculate cell center
            center_x = (cell_x + 0.5) * cw
            center_y = (cell_y + 0.5) * ch
            
            cell_hist = histograms[cell_y, cell_x]
            
            # Draw arrows for each orientation bin
            for bin_idx, magnitude in enumerate(cell_hist):
                if magnitude > min_threshold:
                    angle = bin_angles[bin_idx]
                    
                    # Calculate arrow properties
                    length = magnitude * min(cw, ch) * arrow_scale / max_magnitude
                    dx = length * np.cos(angle)
                    dy = length * np.sin(angle)
                    
                    # Color mapping: [0, π] to [0, 0.5] in HSV color wheel
                    color = plt.cm.hsv(angle / (2 * np.pi))
                    
                    # Draw bidirectional arrows for unsigned gradients
                    arrow_props = {
                        'head_width': 2.0,
                        'head_length': 2.0,
                        'fc': color,
                        'ec': color,
                        'alpha': 0.85,
                        'linewidth': 2.0
                    }
                    
                    ax.arrow(center_x, center_y, dx, dy, **arrow_props)
                    ax.arrow(center_x, center_y, -dx, -dy, **arrow_props)
    
    # Draw grid lines
    for grid_y in range(Nh + 1):
        ax.axhline(y=grid_y * ch - 0.5, color='cyan', linewidth=1.0, alpha=0.6)
    for grid_x in range(Nw + 1):
        ax.axvline(x=grid_x * cw - 0.5, color='cyan', linewidth=1.0, alpha=0.6)
    
    # Set axis properties
    ax.set_xlim(-0.5, Nw * cw - 0.5)
    ax.set_ylim(Nh * ch - 0.5, -0.5)
    ax.set_title(f'{title} (Unsigned)\n({Nh}×{Nw} cells, {len(bin_angles)} bins)', 
                 fontsize=16, fontweight='bold')
    ax.axis('off')



def visualize_hog(hog_params, histograms, patches, hogOutput, i):
    # Get parameters
    cw = hog_params.cell_width
    ch = hog_params.cell_height
    Nw = patches.shape[-1] // cw
    Nh = patches.shape[-2] // ch 
    Nbins = histograms.shape[-1]

    # Reshape histograms to spatial grid
    scales = rearrange(histograms, 'Npatchs (Nh Nw) Nbins -> Npatchs Nh Nw Nbins', Nh=Nh, Nw=Nw).cpu()

    # Select a random patch
    selected_histograms = scales[i]

    print(f"Visualizing patch {i}")
    print(f"Grid: {Nh} x {Nw} cells, each with {Nbins} orientation bins")
    print(f"Cell size: {ch} x {cw} pixels")

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    # 1. Gradient Magnitude
    ax1 = axes[0, 0]
    im1 = ax1.imshow(hogOutput.patches_grdt_magnitude[i].cpu().detach()[0], cmap='hot')
    ax1.set_title('Gradient Magnitude', fontsize=16, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 2. Gradient Orientation (unsigned)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(hogOutput.patches_grdt_orientation[i].cpu().detach()[0], cmap='hsv', vmin=0, vmax=1)
    ax2.set_title('Gradient Orientation\n(Unsigned: 0-180°)', fontsize=16, fontweight='bold')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Normalized [0,1] → [0°,180°]', rotation=270, labelpad=20)

    # 3. HSV Composite (Hue = Orientation, Value = Magnitude)
    ax_hsv = axes[1, 0]
    gradient_magnitude = hogOutput.patches_grdt_magnitude[i].cpu().detach()[0].numpy()
    gradient_orientation = hogOutput.patches_grdt_orientation[i].cpu().detach()[0].numpy()

    # Normalize magnitude to [0, 1] for the Value channel
    magnitude_normalized = gradient_magnitude / gradient_magnitude.max() if gradient_magnitude.max() > 0 else gradient_magnitude

    # Create HSV image
    # H: orientation (0-180° maps to 0-0.5 in HSV color wheel for unsigned gradients)
    # S: full saturation
    # V: gradient magnitude
    hsv_image = np.zeros((*gradient_orientation.shape, 3))
    hsv_image[..., 0] = gradient_orientation  # Map [0,1] to [0,0.5] for 0-180°
    hsv_image[..., 1] = 1.0  # Full saturation
    hsv_image[..., 2] = magnitude_normalized  # Magnitude as brightness

    # Convert to RGB
    rgb_image = hsv_to_rgb(hsv_image)

    im_hsv = ax_hsv.imshow(rgb_image)
    ax_hsv.set_title('HSV Composite\n(Hue=Orientation, Value=Magnitude)', 
                    fontsize=16, fontweight='bold')
    ax_hsv.axis('off')

    # 4. HOG Histogram Visualization
    ax3 = axes[1, 1]
    ax3.imshow(hogOutput.patches_grdt_magnitude[i].cpu().detach()[0], cmap='gray', alpha=0.3)

    # UNSIGNED: angles from 0 to pi
    bin_angles = np.linspace(0, np.pi, Nbins, endpoint=False)

    max_magnitude = selected_histograms.max()

    for cell_y in range(Nh):
        for cell_x in range(Nw):
            center_x = (cell_x + 0.5) * cw
            center_y = (cell_y + 0.5) * ch
            
            cell_hist = selected_histograms[cell_y, cell_x]
            
            for bin_idx, magnitude in enumerate(cell_hist):
                if magnitude > 0.05 * max_magnitude:
                    angle = bin_angles[bin_idx]
                    
                    # Arrow properties
                    length = magnitude * min(cw, ch) * 0.45 / max_magnitude
                    dx = length * np.cos(angle)
                    dy = length * np.sin(angle)
                    
                    # Color: map [0, π] to [0, 0.5] in HSV (to use half the color wheel)
                    color = plt.cm.hsv(angle / (2*np.pi))
                    
                    # Draw BIDIRECTIONAL arrows for unsigned gradients
                    ax3.arrow(center_x, center_y, dx, dy,
                            head_width=2.0, head_length=2.0, 
                            fc=color, ec=color, alpha=0.85, linewidth=2.0)
                    ax3.arrow(center_x, center_y, -dx, -dy,
                            head_width=2.0, head_length=2.0, 
                            fc=color, ec=color, alpha=0.85, linewidth=2.0)

    # Grid lines
    for i in range(Nh + 1):
        ax3.axhline(y=i * ch - 0.5, color='cyan', linewidth=1.0, alpha=0.6)
    for i in range(Nw + 1):
        ax3.axvline(x=i * cw - 0.5, color='cyan', linewidth=1.0, alpha=0.6)

    ax3.set_xlim(-0.5, Nw * cw - 0.5)
    ax3.set_ylim(Nh * ch - 0.5, -0.5)
    ax3.set_title(f'HOG Visualization (Unsigned)\n({Nh}×{Nw} cells, {Nbins} bins)', 
                fontsize=16, fontweight='bold')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

    # Statistics
    print(f"\nHistogram Statistics:")
    print(f"  Min: {selected_histograms.min():.4f}")
    print(f"  Max: {selected_histograms.max():.4f}")
    print(f"  Mean: {selected_histograms.mean():.4f}")
    print(f"  L2 norm: {np.linalg.norm(selected_histograms):.4f}")

    # Bar chart for center cell
    example_cell_y, example_cell_x = Nh // 2, Nw // 2
    example_hist = selected_histograms[example_cell_y, example_cell_x]

    fig2, ax = plt.subplots(figsize=(12, 5))
    bin_angles_deg = np.linspace(0, 180, Nbins, endpoint=False)
    colors = [plt.cm.hsv(a/360) for a in bin_angles_deg]  # Map to first half of HSV

    bars = ax.bar(range(Nbins), example_hist, color=colors,
                edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Orientation Bin', fontsize=13)
    ax.set_ylabel('Magnitude', fontsize=13)
    ax.set_title(f'HOG Histogram - Center Cell (Unsigned, 0-180°)', 
                fontsize=15, fontweight='bold')
    ax.set_xticks(range(Nbins))
    ax.set_xticklabels([f'{int(a)}°' for a in bin_angles_deg], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, example_hist.max() * 1.1)
    plt.tight_layout()
    plt.show()
    return fig