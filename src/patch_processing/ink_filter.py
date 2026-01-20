import numpy as np
from scipy import ndimage
from skimage import measure

from tqdm import tqdm

def filter_binary_patch(binary_img, min_size_black=50, min_size_white=50):
    """
    Filter a binary image by removing small components and border-touching components.
    
    Parameters:
    -----------
    binary_img : numpy.ndarray
        Binary image (2D array with 0s and 1s)
    min_size : int
        Minimum size (in pixels) for components to keep in both image and inverse
    
    Returns:
    --------
    filtered_img : numpy.ndarray
        Filtered binary image
    """

    # # Step 0: Crop to minimum bounding rectangle
    # filtered_img = crop_to_content(binary_img)

    # # Step 1: Remove small border-touching components FIRST
    # filtered_img = remove_border_components(filtered_img, border_min_size)
    
    # Step 2: Find connected components and remove small ones
    labeled_img = measure.label(binary_img, connectivity=2)
    filtered_img = remove_small_components(binary_img, labeled_img, min_size_white)
    
    # Step 3: Process the inverse image (fill small holes)
    inverse_img = 1 - filtered_img
    labeled_inverse = measure.label(inverse_img, connectivity=2)
    filtered_inverse = remove_small_components(inverse_img, labeled_inverse, min_size_black)
    
    # Convert back to get the filtered original
    filtered_img = 1 - filtered_inverse
    
    return filtered_img

def remove_small_components(binary_img, labeled_img, min_size):
    """Remove components smaller than min_size."""
    # Get properties of all components
    regions = measure.regionprops(labeled_img)
    
    # Create output image
    output = np.zeros_like(binary_img)
    
    # Keep only large enough components
    for region in regions:
        if region.area >= min_size:
            output[labeled_img == region.label] = 1
    
    return output


def remove_border_components(binary_img, min_size):
    """Remove components touching borders if they're smaller than min_size."""
    labeled_img = measure.label(~binary_img, connectivity=2)
    h, w = binary_img.shape
    
    # Find labels that touch borders
    border_labels = set()
    border_labels.update(labeled_img[0, :])   # Top border
    border_labels.update(labeled_img[-1, :])  # Bottom border
    border_labels.update(labeled_img[:, 0])   # Left border
    border_labels.update(labeled_img[:, -1])  # Right border
    border_labels.discard(0)  # Remove background label
    
    # Create output image
    output = binary_img.copy()
    
    # Check each border-touching component
    regions = measure.regionprops(labeled_img)
    for region in regions:
        if region.label in border_labels and region.area < min_size:
            output[labeled_img == region.label] = 1
    
    return output


def crop_to_content(binary_img):
    # Find all non-zero pixels (content)
    rows = np.any(binary_img, axis=1)
    cols = np.any(binary_img, axis=0)
    
    # Find the bounding indices
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    
    # If image is empty, return original
    if len(row_indices) == 0 or len(col_indices) == 0:
        return binary_img, (0, binary_img.shape[0], 0, binary_img.shape[1])
    
    row_min, row_max = row_indices[0], row_indices[-1] + 1
    col_min, col_max = col_indices[0], col_indices[-1] + 1
    
    # Crop the image
    cropped_img = binary_img[row_min:row_max, col_min:col_max]
    
    return cropped_img


class InkFilter:
    def __init__(self, min_size_black, min_size_white):
        self.min_size_black = min_size_black
        self.min_size_white = min_size_white

    def __call__(self, binary_images, verbose=True):
        generator = (filter_binary_patch(bin_img) for bin_img in binary_images)
        if verbose:
            generator = tqdm(generator)
        return list(generator)
