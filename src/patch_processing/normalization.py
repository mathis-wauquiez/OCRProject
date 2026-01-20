import numpy as np

def compute_moment(mask, p, q, cy=0, cx=0):
    """
    Compute the (p, q) moment of a binary image
    """
    H, W = mask.shape
    y = np.linspace(-1, 1, H)[:, None]
    x = np.linspace(-1, 1, W)[None, :]
    
    return (((y-cy)**p * (x-cx)**q)*mask).sum()


def compute_normalization_homography(binary_image, target_cx, target_cy):
    """
    Compute normalization homography as a product of simple matrices:
    H = T2 @ S @ R @ T1
    
    Works in normalized [-1, 1] coordinate space.
    """
    # Calculate centroids
    m00 = compute_moment(binary_image, 0, 0)
    
    if m00 == 0:
        raise ValueError("Binary image is empty (all zeros)")
    
    cy = compute_moment(binary_image, 1, 0) / m00
    cx = compute_moment(binary_image, 0, 1) / m00
    
    # Calculate the orientation
    mu20 = compute_moment(binary_image, 2, 0, cy, cx) / m00
    mu02 = compute_moment(binary_image, 0, 2, cy, cx) / m00
    mu11 = compute_moment(binary_image, 1, 1, cy, cx) / m00

    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    
    # Get scale factor
    scale = np.sqrt(mu20 + mu02)
    target_scale = 100  # desired normalized scale
    s = target_scale / scale
    
    # 1. T1: Translate centroid to origin
    T1 = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0,  1]
    ])
    
    # 2. R: Rotate by theta (to align principal axis)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ])
    
    # 3. S: Scale uniformly
    S = np.array([
        [s, 0, 0],
        [0, s, 0],
        [0, 0, 1]
    ])
    
    # 4. T2: Translate to target position
    T2 = np.array([
        [1, 0, target_cx],
        [0, 1, target_cy],
        [0, 0, 1]
    ])
    
    # Compose: H = T2 @ S @ R @ T1
    H = T2 @ S @ R @ T1
    
    return H, {'T1': T1, 'R': R, 'S': S, 'T2': T2}, (cy, cx, theta)

def compute_svg_normalization_homography(binary_image, svg_bbox, target_cx, target_cy):
    """
    Compute normalization homography that works with SVG coordinates.
        
    Args:
        binary_image: Binary mask in pixel space (values in [0,1] or [0,255])
        svg_bbox: (x_min, y_min, width, height) of the SVG bounding box
        target_cx, target_cy: Target position in SVG coordinates
    
    Returns:
        H_svg: 3x3 homography matrix in SVG coordinate space
    """
    # Ensure binary
    if binary_image.max() > 1:
        binary_image = (binary_image > 127).astype(float)
    
    # Get homography in normalized [-1,1] space (centered at origin)
    H_normalized, transformations, params = compute_normalization_homography(
        binary_image, 
        target_cx=0,  # Center at origin in normalized space
        target_cy=0
    )
    
    # Prepare coordinate conversion matrices
    x_min, y_min, width, height = svg_bbox
    
    # Transform from SVG coords to normalized [-1,1]
    svg_to_norm = np.array([
        [2/width, 0, -2*x_min/width - 1],
        [0, 2/height, -2*y_min/height - 1],
        [0, 0, 1]
    ])
    
    # Transform from normalized [-1,1] back to SVG coords (centered at target)
    norm_to_svg = np.array([
        [width/2, 0, target_cx],
        [0, height/2, target_cy],
        [0, 0, 1]
    ])
    
    # Compose: svg -> norm -> align -> svg
    H_svg = norm_to_svg @ H_normalized @ svg_to_norm
    
    return H_svg




def normalization_homography(svg_bbox):
    """
    map bbox coordinates --> [-1; 1]
    """
    x_min, y_min, width, height = svg_bbox
    
    # Transform from SVG coords to [-1,1]
    to_norm = np.array([
        [2/width, 0, -2*x_min/width - 1],
        [0, 2/height, -2*y_min/height - 1],
        [0, 0, 1]
    ])

    return to_norm

from .svg import SVG
from .tests.test_normalization import plot_filtering_result, plot_eigvects

def process_svg(svg_object: SVG, display=False, image=None, scale=4, dpi=256):
    """
    Normalizes the rotation of a SVG object.
    Args:
        svg_object  : The SVG to be rotated to a canonical orientation
        display     : bool, wether or not to display intermediate steps
        scale       : The scaling factor
        image       : the grayscale image corresponding to the SVG


    Steps:
        Render the SVG (dpi, scale)
        Binarize the rendered image
        Filter it
        Find the moments and compute the homography
        Apply it to the SVG

    """

    svg_rendered = svg_object.render(
        dpi=dpi,
        output_format='L',
        scale=scale
    )

    # Filter the SVG to remove small components
    binary_svg = svg_rendered < 127.5
    filtered = filter_binary_patch(binary_svg, min_size_black=min_size_black, min_size_white=min_size_white, border_min_size=border_min_size)
    if filtered.sum() == 0:
        return
    # Compute the alignment homography
    H, transformations, (cy, cx, theta) = compute_normalization_homography(filtered, target_cx=0, target_cy=0)

    if display:
        print('shape after warping:' ,filtered.shape)
        plot_filtering_result(svg_image=~svg_rendered, image=image, filtered=filtered)
        plot_eigvects(filtered, theta, cy, cx)

    # P = to_normalized(svg._compute_bbox())
    P = normalization_homography(svg_object.original_viewbox)
    P_inv = np.linalg.inv(P)

    # Normalize -> Translate -> Rotation -> Inverse normalization
    H = P_inv @ transformations['R'] @ transformations['T1'] @ P
    svg_object.apply_homography(H)


    # === Render the warped SVG ===

    if display:
        warped_img = ~svg_object.render(
                                output_size = svg_object._compute_bbox()[2:],
                                output_format='L',
                                dpi = 256,
                                scale=scale
                                )
        
        print('shape after warping:' ,warped_img.shape)

        warped_img = filter_binary_patch(warped_img, min_size_black=min_size_black, min_size_white=min_size_white, border_min_size=border_min_size)


        H, transformations, (cy, cx, theta) = compute_normalization_homography(warped_img, target_cx=0, target_cy=0)

        plot_eigvects(warped_img, theta, cy, cx)


