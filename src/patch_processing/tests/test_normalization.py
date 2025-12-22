from IPython.display import SVG, display
from ..normalization import compute_normalization_homography



import matplotlib.pyplot as plt
import numpy as np

def plot_filtering_result(svg_image, image, filtered):
    plt.figure(figsize=(22,4))
    plt.subplot(1,5,1)
    plt.title('Vectorized image')
    plt.imshow(svg_image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    plt.subplot(1,5,2)
    plt.title('Original image')
    plt.imshow(image.astype(np.float32) / 255, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    plt.subplot(1,5,3)
    plt.title('Otsu-binarized')

    thresh = 128
    img_otsu = image.astype(np.float32)[None] > thresh
    
    plt.imshow(img_otsu[0], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    plt.subplot(1,5,4)
    plt.title('Filtered')
    plt.imshow(filtered, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    plt.subplot(1,5,5)
    plt.title('Removed area (green)')
    # Calculate removed pixels
    removed = svg_image.astype(bool) & ~filtered.astype(bool)
    # Create RGB image: black background, white for kept, green for removed
    display_img = np.zeros((*svg_image.shape, 3))
    display_img[filtered.astype(bool)] = [1, 1, 1]  # White for kept areas
    display_img[removed] = [0, 1, 0]  # Green for removed areas
    plt.imshow(display_img, interpolation='nearest')



def plot_eigvects(image, angle, cy, cx):
    H, W = image.shape
    cy = - cy

    cx = cx * W/2 + W/2
    cy = cy * H/2 + H/2


    # Principal eigenvector (direction of major axis after rotation)
    eigvect1 = np.array([np.cos(angle), np.sin(angle)])
    # Orthogonal eigenvector (perpendicular to principal axis)
    eigvect2 = np.array([-np.sin(angle), np.cos(angle)])

    # Scale the eigenvectors for better visualization
    scale = 0.3


    # Plotting
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(label='Image values')
    
    # Plot centroid
    plt.plot(cx, cy, 'ro', markersize=10, label='Centroid', zorder=5)
    
    # Plot principal eigenvector
    plt.arrow(cx, cy, 
            eigvect1[0] * scale * H, 
            eigvect1[1] * scale * W,
            head_width=0.05*H, head_length=0.05*H, fc='C1', ec='C1',
            linewidth=2, label='Principal axis', zorder=4)
    
    # Plot secondary eigenvector
    plt.arrow(cx, cy, 
            eigvect2[0] * scale * H, 
            eigvect2[1] * scale * W,
            head_width=0.05*H, head_length=0.05*H, fc='C2', ec='C2',
            linewidth=2, label='Secondary axis', zorder=4)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('scaled')
    plt.title('Image with centroid and principal axes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


