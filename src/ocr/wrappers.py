from __future__ import annotations
import logging
import os
import re
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Type, Any, Tuple

import numpy as np
from PIL import Image

# ============= SUPPRESS HUGGINGFACE LOGGING BEFORE IMPORTS =============
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizer warnings

# Set up logging suppression
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
# ========================================================================

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

import torch
from torch import nn

from src.patch_processing import GridDataset
import cv2

from tqdm import tqdm
def tqdm_wrap(dataset, ocr_name):
    return tqdm(dataset, desc=f"{ocr_name} - Detection", colour="blue")


hf_tok = os.environ['HF_TOKEN']



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------- utilities ----------------

def _ensure_np(array):
    if type(array) == np.ndarray:
        return array
    return array.cpu().numpy()

def _to_pil(img: np.ndarray) -> Image.Image:
    img = _ensure_np(img)
    if img.ndim == 2:
        return Image.fromarray(img.astype(np.uint8))
    if img.ndim == 3 and img.shape[2] == 1:
        return Image.fromarray(img[:, :, 0].astype(np.uint8))
    if img.ndim == 3 and img.shape[2] == 3:
        return Image.fromarray(img.astype(np.uint8))
    raise ValueError(f"Unsupported image shape {img.shape}")


def _normalize_char(s: str) -> str:
    try:
        s = s.strip()
        return s[0] if s else ""
    except Exception:
        return ""

# ------ things i have no other files for -------


from hanziconv import HanziConv

def is_chinese_character(char):
    """Check if a character is a Chinese character (CJK Unified Ideographs)"""
    code_point = ord(char)
    # Main CJK Unified Ideographs block
    return (0x4E00 <= code_point <= 0x9FFF or
            # CJK Extension A
            0x3400 <= code_point <= 0x4DBF or
            # CJK Extension B and beyond
            0x20000 <= code_point <= 0x2A6DF)

def simplified_to_traditional(text):
    """Convert simplified Chinese to traditional Chinese"""
    return HanziConv.toTraditional(text)

def traditional_to_simplified(text):
    """Convert traditional Chinese to simplified Chinese"""
    return HanziConv.toSimplified(text)


# ---------------- abstract base ----------------

class OCRModel(ABC):
    name: str = "abstract"

    @classmethod
    @abstractmethod
    def available(cls) -> bool:
        ...

    def __call__(
        self, patches: List[np.ndarray]
    ) -> List[str]:
        """
        Default implementation: if predict_with_scores exists, use it.
        Otherwise, subclass must override this method.
        """
        if hasattr(self, 'predict_with_scores'):
            chars, _ = self.predict_with_scores(patches)
            return chars
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement either __call__ or predict_with_scores"
            )

    def predict_with_scores(
        self, patches: List[np.ndarray]
    ) -> Tuple[List[str], List[float]]:
        """
        Returns:
            detected_characters: List[str] - predicted characters
            uncertainties: List[float] - uncertainty scores in [0,1]
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict_with_scores"
        )


# ---------------- Tesseract ----------------

class TesseractOCR(OCRModel):
    name = "tesseract"

    @classmethod
    def available(cls) -> bool:
        try:
            import pytesseract
            return True
        except Exception:
            return False

    def __init__(self, lang="chi_tra", psm=10):
        import pytesseract
        self.tess = pytesseract
        self.lang = lang
        self.psm = psm

    def predict_with_scores(self, patches):
        chars = []
        uncertainties = []
        cfg = f"--psm {self.psm}"

        for p in tqdm_wrap(patches, self.name):
            txt = self.tess.image_to_string(
                _to_pil(p), lang=self.lang, config=cfg
            )
            chars.append(_normalize_char(txt))
            uncertainties.append(0.5)  # no native confidence → uninformative prior

        return chars, uncertainties


# ---------------- EasyOCR ----------------

class EasyOCRModel(OCRModel):
    name = "easyocr"

    @classmethod
    def available(cls) -> bool:
        try:
            import easyocr
            return True
        except Exception:
            return False

    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(["ch_tra"], gpu=True)

    def predict_with_scores(self, patches):
        chars = []
        uncertainties = []

        for p in tqdm_wrap(patches, self.name):
            img = np.array(_to_pil(p).convert("RGB"))
            try:
                # detail=1 → [bbox, text, confidence]
                res = self.reader.readtext(img, detail=1)
                if res:
                    _, txt, score = res[0]
                else:
                    txt, score = "", 1.0
            except Exception:
                txt, score = "", 1.0

            chars.append(_normalize_char(txt))
            uncertainties.append(1.0 - float(score))  # uncertainty = 1 - confidence

        return chars, uncertainties


# ---------------- CnOCR ----------------

class CnOCRModel(OCRModel):
    name = "cnocr"

    @classmethod
    def available(cls) -> bool:
        try:
            from cnocr import CnOcr
            return True
        except Exception:
            return False

    def __init__(self):
        from cnocr import CnOcr
        self.ocr = CnOcr(det_model_name=None)

    def predict_with_scores(self, patches):
        chars = []
        uncertainties = []

        for p in tqdm_wrap(patches, self.name):
            try:
                res = self.ocr.ocr(p.astype(np.uint8))
                if res and isinstance(res, list) and isinstance(res[0], dict):
                    txt = res[0].get("text", "")
                    score = float(res[0].get("score", 0.0))
                else:
                    txt, score = "", 0.0
            except Exception:
                txt, score = "", 0.0

            chars.append(_normalize_char(txt))
            uncertainties.append(1.0 - score)  # uncertainty = 1 - confidence

        return chars, uncertainties


# ---------------- Ensemble ----------------

class EnsembleOCR:
    def __init__(self, models: Dict[str, OCRModel]):
        self.models = models

    def predict(self, patches: List[np.ndarray]):
        # Get predictions from all models
        per_model = {}
        for name, model in self.models.items():
            chars, uncertainties = model.predict_with_scores(patches)
            per_model[name] = {
                "chars": chars,
                "uncertainties": uncertainties
            }

        ensemble_chars = []
        ensemble_uncertainties = []

        for i in range(len(patches)):
            votes: Dict[str, float] = {}

            # Collect votes (weighted by confidence = 1 - uncertainty)
            for name, preds in per_model.items():
                c = preds["chars"][i]
                uncertainty = preds["uncertainties"][i]
                confidence = 1.0 - uncertainty

                if not c:
                    continue
                votes[c] = votes.get(c, 0.0) + confidence

            if not votes:
                ensemble_chars.append("")
                ensemble_uncertainties.append(1.0)
                continue

            # Normalize votes to probabilities
            total = sum(votes.values())
            probs = {c: w / total for c, w in votes.items()}

            # Best character
            best_char = max(probs.items(), key=lambda x: x[1])[0]

            # Calculate entropy-based uncertainty
            entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
            max_entropy = math.log(len(probs))
            uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0

            ensemble_chars.append(best_char)
            ensemble_uncertainties.append(uncertainty)

        return {
            "individual": per_model,
            "ensemble": {
                "chars": ensemble_chars,
                "uncertainties": ensemble_uncertainties
            }
        }


# ---------------- Qwen OCR Base ----------------

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError

class QwenOCRBase(nn.Module, OCRModel):
    """Shared base class for Qwen-based OCR models.

    Handles model/processor loading, availability check, and grid-to-PIL conversion.
    Subclasses must define: name, SYSTEM_PROMPT, main_prompt_template.
    """

    SYSTEM_PROMPT: str = ""
    main_prompt_template: str = ""

    def _init_model(self, max_pixels=768 * 768, device_map="auto"):
        """Shared model and processor initialization."""
        from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map=device_map
        )

        kwargs = {} if max_pixels is None else {"max_pixels": max_pixels}
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            **kwargs
        )

    @classmethod
    def available(cls) -> bool:
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            return True
        except Exception:
            return False

    def _grid_to_pil(self, grid_tensor: torch.Tensor) -> Image.Image:
        """Convert grid tensor to PIL Image with contrast enhancement."""
        canvas = (1 - grid_tensor.squeeze()).numpy().astype(np.uint8) * 255
        canvas = cv2.convertScaleAbs(canvas, alpha=1.2, beta=10)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        return Image.fromarray(canvas)

    @torch.inference_mode()
    def _run_inference(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        """Shared inference logic: build messages, run model, decode output."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            use_cache=True
        )

        gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    def __call__(self, dataset):
        chars, _ = self.predict_with_scores(dataset)
        return chars


# ---------------- QwenOCR (grid mode) ----------------

class QwenOCR(QwenOCRBase):
    name = "qwen"

    SYSTEM_PROMPT = (
        "You are an expert Chinese character OCR system. "
        "You have excellent accuracy and can read complex handwritten and printed Chinese characters. "
        "Output every character you see - be confident. "
        "Only use ▯ when a cell is truly empty or completely unreadable. "
        "Match exact visual forms (traditional vs simplified)."
    )

    main_prompt_template: str = """Read ALL characters from this {nrows}×{ncols} grid (total: {num_chars} characters).

GRID STRUCTURE:
- {nrows} rows (horizontal lines)
- {ncols} columns (vertical lines)
- Read left-to-right, then top-to-bottom
- One character per cell

OUTPUT FORMAT (row by row):
Row 1: [char1] [char2] [char3] [char4]
Row 2: [char5] [char6] [char7] [char8]
Row 3: [char9] [char10] [char11] [char12]

RULES:
✓ Output EXACTLY {num_chars} characters
✓ Be confident - read every visible character
✓ Use ▯ only for truly empty/unreadable cells
✓ Preserve traditional/simplified forms exactly as shown

Begin output now (row by row):"""

    def __init__(self, nrows: int, ncols: int, max_pixels=768 * 768, device_map="auto"):
        super().__init__()
        self.nrows = nrows
        self.ncols = ncols
        self.grid_size = nrows * ncols
        self._init_model(max_pixels=max_pixels, device_map=device_map)

    def _parse_row_format(self, raw: str, num_chars: int) -> str:
        """Parse row-by-row format with robust fallback"""
        chars = []

        # Strategy 1: Parse "Row N: char1 char2 char3..." format
        row_pattern = r'Row\s*\d+\s*:\s*(.+)'
        row_matches = re.findall(row_pattern, raw, re.IGNORECASE)

        if row_matches:
            for row_text in row_matches:
                row_chars = []
                for ch in row_text:
                    if ch == "▯":
                        row_chars.append("▯")
                    elif is_chinese_character(ch):
                        row_chars.append(ch)
                chars.extend(row_chars)

            if len(chars) == num_chars:
                return "".join(chars)

        # Strategy 2: Parse bracket format [char1] [char2]
        bracket_pattern = r'\[([^\]]+)\]'
        bracket_matches = re.findall(bracket_pattern, raw)

        if bracket_matches:
            for match in bracket_matches:
                ch = match.strip()
                if ch == "▯":
                    chars.append("▯")
                elif ch and is_chinese_character(ch[0]):
                    chars.append(ch[0])

            if len(chars) == num_chars:
                return "".join(chars)

        # Strategy 3: Extract all Chinese characters in order (aggressive fallback)
        chars = []
        content = raw
        for skip_phrase in ["Row", "row", "格式", "输出", "Output"]:
            content = content.replace(skip_phrase, " ")

        for ch in content:
            if ch == "▯":
                chars.append("▯")
            elif is_chinese_character(ch):
                chars.append(ch)

        # Pad or trim to exact length
        if len(chars) < num_chars:
            chars += ["▯"] * (num_chars - len(chars))

        return "".join(chars[:num_chars])

    def _process_grid(self, image: Image.Image, num_chars: int) -> str:
        prompt = self.main_prompt_template.format(
            nrows=self.nrows,
            ncols=self.ncols,
            num_chars=num_chars
        )
        raw = self._run_inference(image, prompt, max_new_tokens=num_chars * 4)
        return self._parse_row_format(raw, num_chars)

    def predict_with_scores(self, dataset):
        grid_ds = GridDataset(dataset, k=self.nrows, l=self.ncols)

        chars = []
        uncertainties = []
        total = len(dataset)

        for grid_idx, grid in enumerate(
            tqdm(grid_ds, desc=f"{self.name} - Detection", colour="blue")
        ):
            pil = self._grid_to_pil(grid)
            start = grid_idx * self.grid_size
            n = min(self.grid_size, total - start)

            text = self._process_grid(pil, n)

            for ch in text:
                chars.append(ch)
                uncertainties.append(1.0 if ch == "▯" else 0.0)

        chars = [simplified_to_traditional(c) for c in chars]
        return chars, uncertainties


# ---------------- QwenOCRSingle (single character mode) ----------------

class QwenOCRSingle(QwenOCRBase):
    """Optimized OCR for single character recognition (1×1 grid)"""
    name = "qwen_single"

    SYSTEM_PROMPT = (
        "You are a Chinese character OCR system. "
        "Output exactly one Chinese character that you see in the image. "
        "If unreadable, output ▯. Match the exact visual form shown."
    )

    main_prompt_template: str = """What Chinese character is shown in this image?

Output only the character, nothing else.
If unreadable: ▯"""

    def __init__(self, max_pixels=768 * 768, device_map="auto"):
        super().__init__()
        self.nrows = 1
        self.ncols = 1
        self.grid_size = 1
        self._init_model(max_pixels=max_pixels, device_map=device_map)

    def _parse_single_char(self, raw: str) -> str:
        """Extract single character from response"""
        raw = raw.strip()

        if "▯" in raw:
            return "▯"

        for ch in raw:
            if is_chinese_character(ch):
                return ch

        return "▯"

    def _process_single(self, image: Image.Image) -> str:
        """Process single character image"""
        raw = self._run_inference(image, self.main_prompt_template, max_new_tokens=8)
        return self._parse_single_char(raw)

    def predict_with_scores(self, dataset):
        """Process dataset character by character"""
        chars = []
        uncertainties = []

        for idx, sample in enumerate(
            tqdm(dataset, desc=f"{self.name} - Detection", colour="blue")
        ):
            pil = self._grid_to_pil(sample)
            char = self._process_single(pil)

            chars.append(char)
            uncertainties.append(1.0 if char == "▯" else 0.0)

        chars = [simplified_to_traditional(c) for c in chars]
        return chars, uncertainties


# ---------------- registry ----------------

_ALL = [
    TesseractOCR,
    EasyOCRModel,
    CnOCRModel,
    QwenOCR
]


def available_wrappers() -> Dict[str, OCRModel]:
    return {cls.name: cls() for cls in _ALL if cls.available()}
