#!/usr/bin/env python3
"""
Align external transcription files to the patch dataframe.

For each page, builds the OCR prediction sequence (sorted by reading_order)
and aligns it against the ground-truth transcription using Levenshtein edit
operations (rapidfuzz).  The matched transcription character is written into
a new ``char_transcription`` column.

Usage:
    python scripts/align_transcription.py \
        --dataframe outputs/preprocessing/book1 \
        --transcriptions data/transcriptions/book1 \
        --output outputs/preprocessing/book1
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
            page_idx = int(stem[1:]) - 1       # p1 → page 0
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
#  4. Main
# ------------------------------------------------------------------

def main():
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    output_path = args.output or args.dataframe

    # --- Load only the columns we need ---
    df = load_columns(args.dataframe, ['page', 'reading_order', 'char_chat'])

    # --- Load & filter transcriptions ---
    raw_transcriptions = load_transcriptions(args.transcriptions)
    transcriptions = {
        page: filter_chinese(text)
        for page, text in raw_transcriptions.items()
    }

    # --- Align per page ---
    df['char_transcription'] = pd.Series([None] * len(df), dtype=object)

    pages = sorted(df['page'].unique())
    for page in pages:
        mask = df['page'] == page
        page_df = df.loc[mask].sort_values('reading_order')

        if page not in transcriptions:
            log.warning("No transcription for page %d — skipping", page)
            continue

        ocr_chars = page_df['char_chat'].tolist()
        trans = transcriptions[page]

        log.info(
            "Page %d: %d patches, %d transcription chars",
            page, len(ocr_chars), len(trans),
        )

        aligned = align_page(ocr_chars, trans)

        # Write back in the original index order
        df.loc[page_df.index, 'char_transcription'] = aligned

    # --- Summary ---
    n_matched = df['char_transcription'].notna().sum()
    n_total = len(df)
    log.info(
        "Aligned %d / %d patches (%.1f%%)",
        n_matched, n_total, 100.0 * n_matched / n_total if n_total else 0,
    )

    # --- Save ---
    log.info("Saving to %s", output_path)
    # Load the full dataframe for saving (preserves all existing columns)
    from notebook_utils.parquet_utils import load_dataframe
    full_df = load_dataframe(args.dataframe)
    full_df['char_transcription'] = df['char_transcription']
    save_dataframe(full_df, output_path)
    log.info("Done.")


if __name__ == '__main__':
    main()
