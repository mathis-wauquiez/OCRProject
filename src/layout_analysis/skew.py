import cv2
import numpy as np

def get_document_orientation(
        np_img:          np.ndarray,
        min_line_length: float = 100,
        atol:            float = np.pi/8
        ):
    """
    Computes the document orientation using the lines detected by the Line Segment Detector.
    Args:
        np_img:          The input image
        min_line_length: The minimum line length to utilize the line
        atol:            The absolute tolerance to detect vertical / horizontal lines
    Returns:
        theta: float, document orientation in radians

    Methodology:
        - Compute the line segments using the Advanced refinement
        --> cv2 doc: "Number of false alarms is calculated, lines are refined through increase of precision, decrement in size, etc"
        - Only keep long lines
        - Filter them by orientation (keep vertical/horizontal lines)
        - Document orientation = mean of line orientations, weighted by line length
    """
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    lines, *_ = lsd.detect(np_img[..., 0])

    dx = lines[..., 2] - lines[..., 0]
    dy = lines[..., 3] - lines[..., 1]

    lengths = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx) % np.pi

    # Keep long lines
    mask = lengths > min_line_length
    theta = theta[mask]
    lengths = lengths[mask]

    # Horizontal lines
    h_mask = np.isclose(theta, 0, atol=atol) | np.isclose(theta, np.pi, atol=atol)
    # Vertical lines
    v_mask = np.isclose(theta, np.pi/2, atol=atol)

    angles = []
    weights = []

    if np.any(h_mask):
        angles.append(theta[h_mask])
        weights.append(lengths[h_mask])

    if np.any(v_mask):
        angles.append(theta[v_mask] - np.pi/2)
        weights.append(lengths[v_mask])

    if not angles:
        return 0.0  # or np.nan

    angles = np.concatenate(angles)
    weights = np.concatenate(weights)

    # Circular mean for π-periodic angles
    z = np.sum(weights * np.exp(2j * angles))
    return 0.5 * np.angle(z)
