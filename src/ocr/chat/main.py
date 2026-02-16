"""
model_wrapper
~~~~~~~~~~~~~

Wraps a kraken recognition model for column-level OCR, aligning
predictions against CRAFT-detected character centres.

Usage
-----
    import pandas as pd
    from model_wrapper import ModelWrapper

    wrapper = ModelWrapper()  # uses default model at ./models/chat_rec.mlmodel

    result = wrapper.predict(
        image=column_crop,                         # binarised column image
        bboxes=[(x0,y0,x1,y1), ...],              # per-character bboxes from CRAFT
        centers=[(cy, cx), ...],                   # per-character centres (y, x)
    )

    df = pd.DataFrame(result._asdict())
"""

from __future__ import annotations

from typing import List, NamedTuple, Tuple

import pathlib

import numpy as np
import torch
from PIL import Image

from kraken.lib.models import load_any
from kraken.lib.dataset import ImageInputTransforms

UNKNOWN_CHAR = "▯"

# _MODULE_DIR = pathlib.Path(__file__).resolve().parent
CWD = pathlib.Path.cwd()
_DEFAULT_REC_MODEL = CWD / "models" / "chat" / "chat_rec.mlmodel"


class PredictionResult(NamedTuple):
    """DataFrame-ready result: each field is a list of length M (= number of CRAFT centres)."""
    char: list[str]
    confidence: list[float]
    center_y: list[float]
    center_x: list[float]
    bbox_x0: list[int]
    bbox_y0: list[int]
    bbox_x1: list[int]
    bbox_y1: list[int]


class ModelWrapper:
    """Wraps a kraken TorchSeqRecognizer for column-level OCR.

    Parameters
    ----------
    rec_model_path : str or None
        Path to a ``.mlmodel`` recognition model.  If ``None``, defaults
        to ``<cwd_dir>/models/chat_rec.mlmodel``.
    device : str or None
        ``"cuda"`` or ``"cpu"``.  Auto-detected if ``None``.
    pad : int
        Horizontal blank padding added to each side of the line image.
    """

    def __init__(
        self,
        rec_model_path: str | None = None,
        device: str | None = None,
        pad: int = 16,
    ):
        if rec_model_path is None:
            rec_model_path = str(_DEFAULT_REC_MODEL)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.pad = pad

        self.net = load_any(rec_model_path)
        self.net.to(self.device)

        batch, channels, height, width = self.net.nn.input
        self.transform = ImageInputTransforms(
            batch, height, width, channels, (pad, 0), valid_norm=True,
        )

        # Mismatch counters
        self.n_kraken_more = 0
        self.n_kraken_less = 0
        self.n_total = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        centers: List[Tuple[float, float]],
    ) -> PredictionResult:
        """Recognise a column and align predicted characters to CRAFT centres.

        Parameters
        ----------
        image : np.ndarray
            Binarised column image, shape ``(H, W)``, dtype uint8 or bool.
            No cropping is performed — pass the column region directly.
        bboxes : list of (x_min, y_min, x_max, y_max)
            Per-character bounding boxes from CRAFT.
        centers : list of (y, x)
            Per-character centres from CRAFT, in reading order.

        Returns
        -------
        PredictionResult
            NamedTuple of lists, directly usable as ``pd.DataFrame(result._asdict())``.
        """
        assert len(bboxes) == len(centers), (
            f"bboxes ({len(bboxes)}) and centers ({len(centers)}) must have same length"
        )
        M = len(centers)
        self.n_total += 1

        # Run recognition
        pred_chars, pred_confs = self._recognise(image)
        N = len(pred_chars)

        # Align
        if N == M:
            chars = pred_chars
            confs = pred_confs
        elif N > M:
            self.n_kraken_more += 1
            print(f"[WARN] Kraken predicted {N} chars but CRAFT detected {M} "
                  f"(+{N - M}) — truncating. "
                  f"[total occurrences: {self.n_kraken_more}]")
            chars = pred_chars[:M]
            confs = pred_confs[:M]
        else:  # N < M
            self.n_kraken_less += 1
            print(f"[WARN] Kraken predicted {N} chars but CRAFT detected {M} "
                  f"({N - M}) — padding with {UNKNOWN_CHAR}. "
                  f"[total occurrences: {self.n_kraken_less}]")
            chars = pred_chars + [UNKNOWN_CHAR] * (M - N)
            confs = pred_confs + [0.0] * (M - N)

        return PredictionResult(
            char=chars,
            confidence=confs,
            center_y=[c[0] for c in centers],
            center_x=[c[1] for c in centers],
            bbox_x0=[b[0] for b in bboxes],
            bbox_y0=[b[1] for b in bboxes],
            bbox_x1=[b[2] for b in bboxes],
            bbox_y1=[b[3] for b in bboxes],
        )

    def print_stats(self):
        """Print mismatch statistics."""
        print(f"Total predict() calls: {self.n_total}")
        print(f"  Kraken > CRAFT:  {self.n_kraken_more}")
        print(f"  Kraken < CRAFT:  {self.n_kraken_less}")
        print(f"  Matched:         {self.n_total - self.n_kraken_more - self.n_kraken_less}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _recognise(self, image: np.ndarray) -> tuple[list[str], list[float]]:
        """Run kraken on the image.  Returns (chars, confidences)."""
        if image.dtype == bool:
            image = image.astype(np.uint8) * 255

        pil_img = Image.fromarray(image)

        if pil_img.size[0] == 0 or pil_img.size[1] == 0:
            return [], []

        try:
            tensor = self.transform(pil_img)
        except Exception:
            return [], []

        if tensor.max() == tensor.min():
            return [], []

        preds = self.net.predict(tensor.unsqueeze(0))[0]

        chars = [char for char, _, _, _ in preds]
        confs = [float(conf) for _, _, _, conf in preds]
        return chars, confs