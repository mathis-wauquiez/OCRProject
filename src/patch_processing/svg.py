"""
SVG Parser with Homography Transformations
Compact, clear, and optimized implementation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
from io import StringIO, BytesIO

try:
    from lxml import etree as ET
except ImportError:
    from xml.etree import ElementTree as ET

from svgpathtools import parse_path, Path as SvgPath, Line, CubicBezier

try:
    import cairosvg
    from PIL import Image
    RENDERING_AVAILABLE = True
except ImportError:
    RENDERING_AVAILABLE = False


@dataclass
class Contour:
    """Contour represented as Bézier curves"""
    curves: np.ndarray  # Shape: (N, 4, 2) - [start, ctrl1, ctrl2, end]
    closed: bool = True
    
    def __len__(self):
        return len(self.curves)


@dataclass
class SVGPath:
    """SVG path with styling"""
    contours: List[Contour]
    fill: str = "black"
    stroke: str = "none"
    fill_rule: str = "evenodd"


class SVG:
    """SVG container with transformation support"""
    
    def __init__(self):
        self.paths: List[SVGPath] = []
        self.original_viewbox: Optional[Tuple[float, float, float, float]] = None
    
    @classmethod
    def load(cls, filepath: str) -> 'SVG':
        """Load SVG from file"""
        tree = ET.parse(filepath)
        root = tree.getroot()
        return cls._parse_root(root)
    
    @classmethod
    def load_from_string(cls, svg_string: str) -> 'SVG':
        """Load SVG from string"""
        root = ET.fromstring(svg_string)
        return cls._parse_root(root)
    
    @classmethod
    def _parse_root(cls, root) -> 'SVG':
        """Parse SVG from root element"""
        svg = cls()
        
        # Extract original viewBox if present
        viewbox_str = root.get('viewBox')
        if viewbox_str:
            try:
                values = [float(x) for x in viewbox_str.split()]
                if len(values) == 4:
                    svg.original_viewbox = tuple(values)
            except (ValueError, AttributeError):
                pass
        
        # Parse all paths
        for path_elem in root.iter('{http://www.w3.org/2000/svg}path'):
            svg.paths.append(_parse_path_element(path_elem))
        
        # Fallback for paths without namespace
        if not svg.paths:
            for path_elem in root.iter('path'):
                svg.paths.append(_parse_path_element(path_elem))
        
        return svg
    
    def _compute_bbox(self) -> Tuple[float, float, float, float]:
        """Compute bounding box from all curve points"""
        if not self.paths:
            return (0, 0, 100, 100)
        
        all_points = []
        for path in self.paths:
            for contour in path.contours:
                all_points.append(contour.curves.reshape(-1, 2))
        
        if not all_points:
            return (0, 0, 100, 100)
        
        points = np.vstack(all_points)
        min_pt = points.min(axis=0)
        max_pt = points.max(axis=0)
        
        # Add small margin (5% on each side)
        width = max_pt[0] - min_pt[0]
        height = max_pt[1] - min_pt[1]
        margin_x = width * 0.0
        margin_y = height * 0.0
        
        return (min_pt[0] - margin_x, min_pt[1] - margin_y,
                width + 2*margin_x, height + 2*margin_y)
    
    def to_string(self) -> str:
        """Convert SVG to string (without saving to file)"""
        viewBox = self._compute_bbox()
        width = viewBox[2]
        height = viewBox[3]
        
        root = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'width': f'{width:.2f}',
            'height': f'{height:.2f}',
            'viewBox': f'{viewBox[0]:.2f} {viewBox[1]:.2f} {viewBox[2]:.2f} {viewBox[3]:.2f}'
        })
        
        for svg_path in self.paths:
            ET.SubElement(root, 'path', {
                'fill': svg_path.fill,
                'stroke': svg_path.stroke,
                'fill-rule': svg_path.fill_rule,
                'd': _contours_to_path_d(svg_path.contours)
            })
        
        # Convert to string
        return ET.tostring(root, encoding='unicode')
    
    def save(self, filepath: str):
        """Save SVG to file (auto-computes viewBox and dimensions from content)"""
        viewBox = self._compute_bbox()
        width = viewBox[2]
        height = viewBox[3]
        
        root = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'width': f'{width:.2f}',
            'height': f'{height:.2f}',
            'viewBox': f'{viewBox[0]:.2f} {viewBox[1]:.2f} {viewBox[2]:.2f} {viewBox[3]:.2f}'
        })
        
        for svg_path in self.paths:
            ET.SubElement(root, 'path', {
                'fill': svg_path.fill,
                'stroke': svg_path.stroke,
                'fill-rule': svg_path.fill_rule,
                'd': _contours_to_path_d(svg_path.contours)
            })
        
        tree = ET.ElementTree(root)
        if hasattr(ET, 'indent'):  # Python 3.9+
            ET.indent(tree, space='  ')
        
        # lxml uses 'utf-8', ElementTree uses 'unicode'
        try:
            tree.write(filepath, encoding='utf-8', xml_declaration=True)
        except TypeError:
            tree.write(filepath, encoding='unicode', xml_declaration=True)
    
    def render(self, 
               output_size: Optional[Tuple[int, int]] = None,
               background_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
               dpi: int = 96,
               scale: float = 1.0,
               respect_aspect_ratio = True,
               output_format: str = 'RGBA') -> np.ndarray:
        """
        Render SVG to numpy array with aspect ratio preservation.
        
        Args:
            output_size: Target size (width, height). If None, uses original viewBox size
            background_color: RGBA background color tuple
            dpi: DPI for rendering
            scale: Additional scale factor
            output_format: Output format ('RGBA', 'RGB', 'L', etc.)
        
        Returns:
            Rendered image as numpy array
        """
        if not RENDERING_AVAILABLE:
            raise ImportError("Install cairosvg and Pillow: pip install cairosvg pillow")
        
        # Get SVG content as string
        svg_content = self.to_string()
        
        # Get original dimensions from viewBox
        if self.original_viewbox:
            orig_x, orig_y, orig_w, orig_h = self.original_viewbox
        else:
            bbox = self._compute_bbox()
            orig_x, orig_y, orig_w, orig_h = bbox
        
        # Calculate dimensions with aspect ratio preservation
        if output_size:
            if respect_aspect_ratio:
                target_w, target_h = output_size
                orig_aspect = orig_w / orig_h
                target_aspect = target_w / target_h
                
                if orig_aspect > target_aspect:
                    w = target_w
                    h = int(target_w / orig_aspect)
                else:
                    h = target_h
                    w = int(target_h * orig_aspect)
            else:
                w, h = output_size
        else:
            w, h = int(orig_w), int(orig_h)
        
        w, h = int(w * scale), int(h * scale)
        
        # Render SVG to PNG
        png_data = cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            output_width=w,
            output_height=h,
            dpi=dpi
        )
        
        # Load and apply background
        img = Image.open(BytesIO(png_data))
        
        if background_color:
            bg = Image.new('RGBA', img.size, background_color)
            bg.paste(img, (0, 0), img)
            img.close()
            img = bg
        
        # Convert format if needed
        if output_format != 'RGBA':
            converted = img.convert(output_format)
            img.close()
            img = converted
        
        # Convert to array
        arr = np.ascontiguousarray(np.array(img))
        img.close()
        
        del png_data
        return arr
    
    def apply_homography(self, H: np.ndarray):
        """
        Apply 3×3 homography to all paths.
        
        Converts points to homogeneous coords, applies H, converts back.
        """
        assert H.shape == (3, 3), f"H must be 3×3, got {H.shape}"
        
        for i, path in enumerate(self.paths):
            transformed_contours = []
            for contour in path.contours:
                # Vectorized transformation
                pts = contour.curves.reshape(-1, 2)  # (N*4, 2)
                pts_h = np.column_stack([pts, np.ones(len(pts))])  # (N*4, 3)
                pts_t = (H @ pts_h.T).T  # (N*4, 3)
                pts_cart = pts_t[:, :2] / pts_t[:, 2:3]  # (N*4, 2)
                curves_t = pts_cart.reshape(contour.curves.shape)  # (N, 4, 2)
                transformed_contours.append(Contour(curves_t, contour.closed))
            
            self.paths[i] = SVGPath(transformed_contours, path.fill, path.stroke, path.fill_rule)
    
    def __repr__(self):
        bbox = self._compute_bbox()
        viewbox_info = f", original_viewbox={self.original_viewbox}" if self.original_viewbox else ""
        return f"SVG({bbox[2]:.1f}×{bbox[3]:.1f}, {len(self.paths)} paths{viewbox_info})"
        
    def render_svg(self):
        """Render SVG in Jupyter notebook (vector, not rasterized)"""
        from IPython.display import SVG, display
        display(SVG(self.to_string()))


def _parse_path_element(elem) -> SVGPath:
    """Parse a single path element"""
    d = elem.get('d', '')
    fill = elem.get('fill', 'black')
    stroke = elem.get('stroke', 'none')
    fill_rule = elem.get('fill-rule', 'evenodd')
    
    contours = _parse_path_d(d)
    return SVGPath(contours, fill, stroke, fill_rule)


def _parse_path_d(d: str) -> List[Contour]:
    """Parse SVG path data string into contours"""
    if not d.strip():
        return []
    
    path = parse_path(d)
    if not path:
        return []
    
    contours = []
    current_curves = []
    
    for i, segment in enumerate(path):
        # Convert segment to numpy curve
        if isinstance(segment, Line):
            # Convert line to Bézier
            start = np.array([segment.start.real, segment.start.imag])
            end = np.array([segment.end.real, segment.end.imag])
            ctrl1 = start + (end - start) / 3
            ctrl2 = start + 2 * (end - start) / 3
            curve = [start, ctrl1, ctrl2, end]
        
        elif isinstance(segment, CubicBezier):
            # Direct Bézier curve
            curve = [
                np.array([segment.start.real, segment.start.imag]),
                np.array([segment.control1.real, segment.control1.imag]),
                np.array([segment.control2.real, segment.control2.imag]),
                np.array([segment.end.real, segment.end.imag])
            ]
        else:
            # Other segment types (QuadraticBezier, Arc, etc.) are converted by svgpathtools
            continue
        
        # Check if this segment connects to previous one
        if current_curves:
            last_end = current_curves[-1][3]
            current_start = curve[0]
            
            # If there's a gap (discontinuity), start a new contour
            if not np.allclose(last_end, current_start, atol=1e-6):
                # Save current contour
                curves_array = np.array(current_curves)
                closed = np.allclose(curves_array[0, 0], curves_array[-1, 3], atol=1e-6)
                contours.append(Contour(curves_array, closed))
                current_curves = []
        
        current_curves.append(curve)
    
    # Add final contour
    if current_curves:
        curves_array = np.array(current_curves)
        closed = np.allclose(curves_array[0, 0], curves_array[-1, 3], atol=1e-6)
        contours.append(Contour(curves_array, closed))
    
    return contours


def _contours_to_path_d(contours: List[Contour]) -> str:
    """Convert contours back to SVG path data"""
    parts = []
    
    for contour in contours:
        if len(contour.curves) == 0:
            continue
        
        # Move to start
        start = contour.curves[0, 0]
        parts.append(f"M{start[0]:.6f} {start[1]:.6f}")
        
        # Add curves
        for curve in contour.curves:
            c1, c2, end = curve[1], curve[2], curve[3]
            parts.append(f"C{c1[0]:.6f} {c1[1]:.6f} {c2[0]:.6f} {c2[1]:.6f} {end[0]:.6f} {end[1]:.6f}")
        
        if contour.closed:
            parts.append("Z")
    
    return " ".join(parts)


def evaluate_bezier(curves: np.ndarray, t: float) -> np.ndarray:
    """
    Evaluate Bézier curves at parameter t ∈ [0, N].
    
    Integer part selects curve, fractional part is position within curve.
    """
    idx = min(int(t), len(curves) - 1)
    t_local = t - idx if idx < len(curves) - 1 else 1.0
    
    P = curves[idx]
    mt = 1 - t_local
    
    # Cubic Bézier: (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    return (mt**3 * P[0] + 3 * mt**2 * t_local * P[1] + 
            3 * mt * t_local**2 * P[2] + t_local**3 * P[3])


# Transformation matrix builders
def translation(dx: float, dy: float) -> np.ndarray:
    """Translation matrix"""
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=float)


def scaling(sx: float, sy: float = None) -> np.ndarray:
    """Scaling matrix (uniform if sy not specified)"""
    sy = sy or sx
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=float)


def rotation(angle: float, center: Tuple[float, float] = None) -> np.ndarray:
    """Rotation matrix (angle in radians, optional center)"""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    
    if center:
        cx, cy = center
        return translation(cx, cy) @ R @ translation(-cx, -cy)
    return R


def shear(shx: float = 0, shy: float = 0) -> np.ndarray:
    """Shear matrix"""
    return np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]], dtype=float)


def mirror_x() -> np.ndarray:
    """Mirror horizontally"""
    return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)


def mirror_y() -> np.ndarray:
    """Mirror vertically"""
    return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)