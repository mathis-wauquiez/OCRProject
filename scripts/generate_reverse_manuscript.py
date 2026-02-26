#!/usr/bin/env python3
"""
Generate the "reverse-printed manuscript" visualization.

For each detected character in the document, overlay the representative
vectorised character (from the glossary) at its position, rendered in
light blue.  Two versions are produced:

  (a) With the original manuscript as background
  (b) On a blank white background (characters only)

Usage:
    python scripts/generate_reverse_manuscript.py \
        --dataframe   results/clustering/book1/clustered_patches \
        --images-dir  data/datasets/book1 \
        --output-dir  paper/figures/generated/reverse_manuscript

    Optional:
        --page         5        Single page index to render (default: all)
        --dpi          300      Output DPI
        --char-color   "#4a90d9"  Colour for overlaid characters (hex)
        --char-alpha   0.65     Opacity of overlaid characters
        --bg-alpha     0.25     Opacity of manuscript background (version a)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from notebook_utils.parquet_utils import load_dataframe, load_columns
from src.clustering.post_clustering import build_glossary
from src.clustering.metrics import UNKNOWN_LABEL


def hex_to_rgb(hex_color):
    """Convert '#RRGGBB' to (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def render_svg_to_pil(svg_obj, target_width, target_height, dpi=96):
    """Render an SVG to a PIL Image (grayscale), fit to target size."""
    try:
        rendered = svg_obj.render(
            dpi=dpi,
            output_format='L',
            scale=1.0,
            output_size=(target_width, target_height),
            respect_aspect_ratio=True,
        )
        return Image.fromarray(rendered, mode='L')
    except Exception:
        return Image.new('L', (target_width, target_height), 255)


def build_representative_lookup(dataframe, glossary_df):
    """Build a mapping from char_chat label to representative SVG + index.

    Returns
    -------
    rep_lookup : dict
        {character_label: representative_dataframe_index}
    """
    rep_lookup = {}
    for _, row in glossary_df.iterrows():
        char = row['character']
        rep_idx = int(row['representative_idx'])
        rep_lookup[char] = rep_idx
    return rep_lookup


def render_page(
    page_df,
    dataframe,
    rep_lookup,
    page_image=None,
    char_color=(74, 144, 217),
    char_alpha=0.65,
    bg_alpha=0.25,
    dpi=96,
):
    """Render a single page in both with-background and without-background modes.

    Parameters
    ----------
    page_df : pd.DataFrame
        Subset of the full dataframe for one page.
        Must have 'left', 'top', 'width', 'height', 'char_chat'.
    dataframe : pd.DataFrame
        Full dataframe with 'svg' column (for representative rendering).
    rep_lookup : dict
        {char_label: representative_index} from build_representative_lookup.
    page_image : PIL.Image or None
        Original page scan. If None, only the without-background version
        is produced.
    char_color : tuple
        (R, G, B) colour for the overlaid characters.
    char_alpha : float
        Opacity of the characters (0-1).
    bg_alpha : float
        Opacity of the manuscript background (0-1).

    Returns
    -------
    img_with_bg : PIL.Image (RGBA) or None
        Overlay on manuscript.
    img_no_bg : PIL.Image (RGBA)
        Characters on white background.
    """
    # Determine canvas size
    if page_image is not None:
        canvas_w, canvas_h = page_image.size
    else:
        max_x = int((page_df['left'] + page_df['width']).max()) + 50
        max_y = int((page_df['top'] + page_df['height']).max()) + 50
        canvas_w, canvas_h = max_x, max_y

    # Create character overlay (RGBA, transparent)
    char_layer = Image.new('RGBA', (canvas_w, canvas_h), (255, 255, 255, 0))

    for _, row in page_df.iterrows():
        left = int(row['left'])
        top = int(row['top'])
        w = int(row['width'])
        h = int(row['height'])

        char_label = row.get('char_chat', UNKNOWN_LABEL)
        if pd.isna(char_label) or char_label == UNKNOWN_LABEL:
            continue

        rep_idx = rep_lookup.get(char_label)
        if rep_idx is None:
            continue

        # Render the representative SVG at this bounding-box size
        svg_obj = dataframe.loc[rep_idx, 'svg']
        char_img = render_svg_to_pil(svg_obj, w, h, dpi=dpi)

        # Convert grayscale mask to coloured RGBA
        char_arr = np.array(char_img)
        # Invert: black strokes (0) → opaque, white bg (255) → transparent
        alpha = ((255 - char_arr) * char_alpha).astype(np.uint8)
        rgba = np.zeros((char_arr.shape[0], char_arr.shape[1], 4), dtype=np.uint8)
        rgba[:, :, 0] = char_color[0]
        rgba[:, :, 1] = char_color[1]
        rgba[:, :, 2] = char_color[2]
        rgba[:, :, 3] = alpha

        char_rgba = Image.fromarray(rgba, mode='RGBA')

        # Paste onto the character layer, centring within the bounding box
        paste_x = left + (w - char_rgba.width) // 2
        paste_y = top + (h - char_rgba.height) // 2
        char_layer.paste(char_rgba, (paste_x, paste_y), char_rgba)

    # Version (b): characters on white background
    white_bg = Image.new('RGBA', (canvas_w, canvas_h), (255, 255, 255, 255))
    img_no_bg = Image.alpha_composite(white_bg, char_layer)

    # Version (a): characters on manuscript
    img_with_bg = None
    if page_image is not None:
        # Fade the manuscript
        manuscript_rgba = page_image.convert('RGBA')
        # Reduce manuscript opacity
        ms_arr = np.array(manuscript_rgba)
        faded = np.full_like(ms_arr, 255)
        faded[:, :, :3] = (
            ms_arr[:, :, :3].astype(np.float32) * bg_alpha
            + 255 * (1 - bg_alpha)
        ).astype(np.uint8)
        faded[:, :, 3] = 255  # fully opaque
        faded_bg = Image.fromarray(faded, mode='RGBA')

        img_with_bg = Image.alpha_composite(faded_bg, char_layer)

    return img_with_bg, img_no_bg


def get_page_files(dataframe):
    """Extract unique page files from the dataframe."""
    if 'file' not in dataframe.columns:
        return {}
    files = dataframe['file'].unique()
    page_map = {}
    for f in files:
        try:
            page_num = int(f.split('_')[-1].split('.')[0])
            page_map[page_num] = f
        except (ValueError, IndexError):
            page_map[f] = f
    return page_map


def main():
    parser = argparse.ArgumentParser(
        description="Generate the reverse-printed manuscript visualization."
    )
    parser.add_argument(
        "--dataframe", required=True, type=Path,
        help="Path to the clustered_patches dataframe directory.",
    )
    parser.add_argument(
        "--images-dir", type=Path, default=Path("data/datasets/book1"),
        help="Directory containing the original page images.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("paper/figures/generated/reverse_manuscript"),
        help="Output directory.",
    )
    parser.add_argument("--page", type=int, default=None,
                        help="Render only this page index.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--char-color", type=str, default="#4a90d9",
                        help="Hex colour for overlaid characters.")
    parser.add_argument("--char-alpha", type=float, default=0.65)
    parser.add_argument("--bg-alpha", type=float, default=0.25)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    char_color = hex_to_rgb(args.char_color)

    print(f"Loading dataframe from {args.dataframe}...")
    dataframe = load_dataframe(args.dataframe)
    print(f"Loaded {len(dataframe)} patches")

    # Build glossary and representative lookup
    print("Building glossary...")
    glossary_df = build_glossary(dataframe)
    rep_lookup = build_representative_lookup(dataframe, glossary_df)
    print(f"Glossary: {len(glossary_df)} characters, "
          f"{len(rep_lookup)} with representatives")

    # Determine pages to render
    page_map = get_page_files(dataframe)
    if args.page is not None:
        pages_to_render = {args.page: page_map.get(args.page)}
    else:
        pages_to_render = page_map

    for page_num, page_file in sorted(pages_to_render.items()):
        print(f"\nRendering page {page_num}...")

        # Filter dataframe for this page
        if 'file' in dataframe.columns and page_file is not None:
            page_df = dataframe[dataframe['file'] == page_file]
        else:
            page_df = dataframe

        if len(page_df) == 0:
            print(f"  No patches found for page {page_num}, skipping.")
            continue

        # Load the original page image
        page_image = None
        if args.images_dir is not None and page_file is not None:
            img_path = args.images_dir / page_file
            if img_path.exists():
                page_image = Image.open(img_path)
                print(f"  Loaded image: {img_path} ({page_image.size})")
            else:
                # Try common patterns
                for pattern in [f"*{page_num:03d}*", f"*{page_num}*"]:
                    matches = list(args.images_dir.glob(pattern))
                    if matches:
                        page_image = Image.open(matches[0])
                        print(f"  Loaded image: {matches[0]}")
                        break

        # Render
        img_with_bg, img_no_bg = render_page(
            page_df, dataframe, rep_lookup,
            page_image=page_image,
            char_color=char_color,
            char_alpha=args.char_alpha,
            bg_alpha=args.bg_alpha,
        )

        # Save
        suffix = f"page_{page_num:03d}"
        if img_with_bg is not None:
            out_a = args.output_dir / f"reverse_with_bg_{suffix}.png"
            img_with_bg.convert('RGB').save(out_a, dpi=(args.dpi, args.dpi))
            print(f"  Saved (with bg): {out_a}")

        out_b = args.output_dir / f"reverse_no_bg_{suffix}.png"
        img_no_bg.convert('RGB').save(out_b, dpi=(args.dpi, args.dpi))
        print(f"  Saved (no bg):   {out_b}")

    print("\nDone.")


if __name__ == "__main__":
    main()
