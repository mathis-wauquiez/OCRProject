"""Shared SVG rendering helpers for figure-generation scripts."""


def render_svg_grayscale(svg_obj, width, height, dpi=96):
    """Render an SVG object to a grayscale numpy array (H, W), dtype uint8."""
    return svg_obj.render(
        dpi=dpi,
        output_format='L',
        scale=1.0,
        output_size=(width, height),
        respect_aspect_ratio=True,
    )
