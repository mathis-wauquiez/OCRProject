#!/usr/bin/env python3
"""
Align external transcription files to the patch dataframe.

For each page, builds the OCR prediction sequence (sorted by reading_order)
and aligns it against the ground-truth transcription using Levenshtein edit
operations (rapidfuzz).  The matched transcription character is written into
a new ``char_transcription`` column.

Usage:
    python scripts/align_transcription.py \
        --dataframe results/preprocessing/book1 \
        --transcriptions data/transcriptions/book1 \
        --output results/preprocessing/book1
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz.distance import Levenshtein

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from notebook_utils.parquet_utils import load_columns, save_dataframe
from src.ocr.wrappers import is_chinese_character

log = logging.getLogger(__name__)

# Placeholder for patches whose char_chat is absent / unknown.
# Must NOT be a CJK character so it never accidentally matches a
# transcription character.
_PLACEHOLDER = '\uffff'


# ------------------------------------------------------------------
#  1. Load transcriptions
# ------------------------------------------------------------------

def load_transcriptions(folder: Path) -> dict[int, str]:
    """Load transcription files named ``p<n>`` or ``p<n>.txt``.

    Returns a dict mapping **0-based** page index to raw text content.
    """
    folder = Path(folder)
    transcriptions: dict[int, str] = {}

    for path in sorted(folder.iterdir()):
        stem = path.stem                       # "p3" or "p3"
        if not stem.startswith('p'):
            continue
        try:
            page_idx = int(stem[1:])
        except ValueError:
            continue
        transcriptions[page_idx] = path.read_text(encoding='utf-8')

    log.info("Loaded %d transcription files from %s", len(transcriptions), folder)
    return transcriptions


# ------------------------------------------------------------------
#  2. Filter to Chinese characters only
# ------------------------------------------------------------------

def filter_chinese(text: str) -> str:
    """Keep only CJK characters, stripping punctuation / whitespace."""
    return ''.join(ch for ch in text if is_chinese_character(ch))


# ------------------------------------------------------------------
#  3. Levenshtein alignment
# ------------------------------------------------------------------

def align_page(
    ocr_chars: list[str | None],
    transcription: str,
) -> list[str | None]:
    """Align a page's OCR predictions to its transcription.

    Parameters
    ----------
    ocr_chars : list[str | None]
        One entry per patch (in reading order).  ``None`` or ``'▯'``
        for patches without a usable OCR prediction.
    transcription : str
        Pre-filtered ground-truth transcription (Chinese chars only).

    Returns
    -------
    list[str | None]
        Same length as *ocr_chars*.  Each entry is the matched
        transcription character, or ``None`` if the patch has no
        counterpart in the transcription.
    """
    n = len(ocr_chars)
    if n == 0 or len(transcription) == 0:
        return [None] * n

    # Build OCR string, replacing unusable predictions with a placeholder
    ocr_string = ''.join(
        ch if (ch is not None and ch != '▯' and is_chinese_character(ch))
        else _PLACEHOLDER
        for ch in ocr_chars
    )

    ops = Levenshtein.editops(ocr_string, transcription)

    # Partition edit-ops by type
    replaced_ocr: dict[int, int] = {}   # ocr_pos → trans_pos
    deleted_ocr: set[int] = set()        # ocr positions with no counterpart

    for tag, i, j in ops:
        if tag == 'replace':
            replaced_ocr[i] = j
        elif tag == 'delete':
            deleted_ocr.add(i)
        # 'insert': transcription char with no OCR patch — skip

    # Positions untouched by any op are 1-to-1 matches.
    # Walk both sequences skipping deleted/inserted positions to pair them.
    ocr_touched = set(replaced_ocr.keys()) | deleted_ocr
    insert_positions: set[int] = {j for tag, _, j in ops if tag == 'insert'}

    ocr_matched = [i for i in range(len(ocr_string)) if i not in ocr_touched]
    trans_matched = [j for j in range(len(transcription)) if j not in insert_positions and j not in replaced_ocr.values()]

    # Build result
    result: list[str | None] = [None] * n

    # Equal (matched) positions
    for oi, ti in zip(ocr_matched, trans_matched):
        result[oi] = transcription[ti]

    # Replace positions — OCR was wrong, transcription is the truth
    for oi, ti in replaced_ocr.items():
        result[oi] = transcription[ti]

    # Deleted positions stay None (patch exists but is not in transcription)

    return result


# ------------------------------------------------------------------
#  4. Visualization
# ------------------------------------------------------------------

def _find_cjk_font_path() -> str | None:
    """Find a .ttf/.otf path for a CJK-capable font (for PIL ImageFont)."""
    preferred = [
        'Noto Sans CJK TC', 'Noto Sans CJK SC', 'Noto Sans CJK JP',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'AR PL UMing TW', 'AR PL UKai TW',
        'Microsoft YaHei', 'SimHei', 'SimSun',
    ]
    available = {f.name: f.fname for f in fm.fontManager.ttflist}

    for name in preferred:
        if name in available:
            return available[name]

    for name, path in available.items():
        if any(kw in name.lower() for kw in ['cjk', 'chinese', 'hei', 'sung', 'ming', 'kai']):
            return path
    return None


# Colour palette
_COLOR_MATCH   = (0, 200, 0)       # green  — OCR == transcription
_COLOR_REPLACE = (255, 165, 0)     # orange — OCR wrong, transcription filled in
_COLOR_DELETE  = (220, 40, 40)     # red    — patch has no transcription counterpart
_COLOR_UNKNOWN = (130, 130, 130)   # grey   — OCR was ▯ / placeholder


def visualize_page_alignment(
    page_image: np.ndarray,
    page_df: pd.DataFrame,
    ocr_chars: list[str | None],
    aligned_chars: list[str | None],
    page_id: int,
    output_path: Path | None = None,
):
    """Overlay alignment results on the page image.

    Each patch gets a bounding-box and a label showing
    ``ocr → transcription`` coloured by match quality.
    """
    # Classify each position
    n = len(ocr_chars)
    categories = []
    for i in range(n):
        ocr_ch = ocr_chars[i]
        ali_ch = aligned_chars[i]
        is_placeholder = (ocr_ch is None or ocr_ch == '▯'
                          or not is_chinese_character(ocr_ch))

        if ali_ch is None or ali_ch == '▯':
            categories.append('delete')
        elif is_placeholder:
            categories.append('unknown')
        elif ocr_ch == ali_ch:
            categories.append('match')
        else:
            categories.append('replace')

    color_map = {
        'match':   _COLOR_MATCH,
        'replace': _COLOR_REPLACE,
        'delete':  _COLOR_DELETE,
        'unknown': _COLOR_UNKNOWN,
    }

    # Convert to RGB canvas if needed
    if page_image.ndim == 2:
        canvas = cv2.cvtColor(page_image, cv2.COLOR_GRAY2RGB)
    else:
        canvas = page_image.copy()

    # Try to load a CJK font for PIL text rendering
    font_path = _find_cjk_font_path()
    font_size = 44
    try:
        pil_font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        pil_font = ImageFont.load_default()

    # Draw bounding boxes with cv2, then overlay CJK text with PIL
    rows = page_df.sort_values('reading_order').reset_index(drop=True)
    for i, (_, row) in enumerate(rows.iterrows()):
        if i >= n:
            break

        left, top = int(row['left']), int(row['top'])
        w, h = int(row['width']), int(row['height'])
        cat = categories[i]
        color = color_map[cat]

        # Bounding box
        thickness = 2
        cv2.rectangle(canvas, (left, top), (left + w, top + h), color, thickness)

    # Switch to PIL for CJK text
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)

    for i, (_, row) in enumerate(rows.iterrows()):
        if i >= n:
            break

        left, top = int(row['left']), int(row['top'])
        w, h = int(row['width']), int(row['height'])
        cat = categories[i]
        color = color_map[cat]

        ocr_ch = ocr_chars[i] if ocr_chars[i] is not None else '?'
        ali_ch = aligned_chars[i] if aligned_chars[i] is not None else '?'

        if cat == 'match':
            label = ali_ch
        elif cat == 'delete':
            label = f'{ocr_ch}|x'
        else:
            label = f'{ocr_ch}|{ali_ch}'

        tx = left + w + 2
        ty = top
        # Semi-transparent background for readability
        bbox = draw.textbbox((tx, ty), label, font=pil_font)
        draw.rectangle(bbox, fill=(255, 255, 255, 200))
        draw.text((tx, ty), label, font=pil_font, fill=color)

    canvas = np.array(pil_img)

    # Build matplotlib figure — match image aspect ratio
    img_h, img_w = canvas.shape[:2]
    fig_w = 20
    fig_h = fig_w * img_h / img_w
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(canvas, interpolation='none', aspect='auto')
    ax.set_axis_off()

    # Legend
    import matplotlib.patches as mpatches
    legend_items = [
        mpatches.Patch(facecolor=np.array(_COLOR_MATCH) / 255,   label='Match (OCR == Transcription)'),
        mpatches.Patch(facecolor=np.array(_COLOR_REPLACE) / 255, label='Replace (OCR wrong)'),
        mpatches.Patch(facecolor=np.array(_COLOR_DELETE) / 255,  label='Delete (no transcription)'),
        mpatches.Patch(facecolor=np.array(_COLOR_UNKNOWN) / 255, label='Unknown OCR (▯ → transcription)'),
    ]
    ax.legend(handles=legend_items, loc='upper left', fontsize=10,
              framealpha=0.8, edgecolor='black')

    # Stats in title
    n_match   = categories.count('match')
    n_replace = categories.count('replace')
    n_delete  = categories.count('delete')
    n_unknown = categories.count('unknown')
    ax.set_title(
        f'Page {page_id} — {n} patches | '
        f'match {n_match}  replace {n_replace}  '
        f'delete {n_delete}  unknown {n_unknown}',
        fontsize=14, fontweight='bold',
    )

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log.info("Saved visualization → %s", output_path)
    else:
        plt.show()

    return fig


# ------------------------------------------------------------------
#  5. Main
# ------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Align transcription files to the patch dataframe."
    )
    parser.add_argument(
        '--dataframe', required=True, type=Path,
        help="Path to the saved dataframe (column-wise directory)."
    )
    parser.add_argument(
        '--transcriptions', required=True, type=Path,
        help="Folder containing transcription files (p1, p2, ...)."
    )
    parser.add_argument(
        '--output', type=Path, default=None,
        help="Where to save the updated dataframe. Defaults to --dataframe (in-place)."
    )
    parser.add_argument(
        '--images', type=Path, default=None,
        help="Folder containing page images for visualization. "
             "If omitted, visualization is skipped."
    )
    parser.add_argument(
        '--viz-output', type=Path, default=None,
        help="Folder to save alignment PNGs. Defaults to <output>/alignment_viz/."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    output_path = args.output or args.dataframe

    transcriptions = load_transcriptions(args.transcriptions)

    dataframe = load_columns(args.dataframe,
                             ['reading_order', 'char_chat', 'file',
                              'left', 'top', 'width', 'height'])

    print(np.unique(dataframe['file'].tolist()))
    # filename: wdl_13516_xxx.jpg
    dataframe['page'] = dataframe['file'].apply(lambda x: int(x.split('_')[-1][:-4]))

    dataframe['char_transcription'] = pd.Series([None] * len(dataframe), dtype=object)

    print(dataframe)

    # Visualization setup
    images_folder = args.images
    viz_output = args.viz_output or (output_path / 'alignment_viz')
    if images_folder is not None:
        viz_output.mkdir(parents=True, exist_ok=True)

    for page, text in transcriptions.items():
        filtered_text = filter_chinese(text)
        print(f"Processing page {page}...")
        print('Page content:')

        page_mask = dataframe['page'] == page
        page_df = dataframe.loc[page_mask].sort_values('reading_order')

        ocr_chars = page_df['char_chat'].tolist()
        aligned_chars = align_page(ocr_chars, filtered_text)
        aligned_chars_display = [ch if ch is not None else '▯' for ch in aligned_chars]

        dataframe.loc[page_df.index, 'char_transcription'] = aligned_chars_display

        print('Number of characters in transcription:', len(filtered_text))
        print('Number of OCR patches:', len(ocr_chars))
        print('Number of aligned characters:', sum(ch != '▯' for ch in aligned_chars_display))
        print('---')

        # --- Visualization ---
        if images_folder is not None:
            page_files = page_df['file'].unique()
            if len(page_files) == 0:
                continue
            img_path = images_folder / page_files[0]
            if not img_path.exists():
                log.warning("Image not found: %s — skipping viz", img_path)
                continue
            page_image = np.array(Image.open(img_path))

            visualize_page_alignment(
                page_image=page_image,
                page_df=page_df,
                ocr_chars=ocr_chars,
                aligned_chars=aligned_chars,
                page_id=page,
                output_path=viz_output / f'page_{page:03d}.png',
            )

    from notebook_utils.parquet_utils import load_dataframe
    full_df = load_dataframe(args.dataframe)
    full_df['char_transcription'] = dataframe['char_transcription']
    save_dataframe(full_df, output_path)
