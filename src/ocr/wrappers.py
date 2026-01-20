from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Type, Any, Tuple
import math

import numpy as np
from PIL import Image

# ============= SUPPRESS HUGGINGFACE LOGGING BEFORE IMPORTS =============
import os
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
from PIL import Image
import os

import torch
from torch import nn

import re

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

    


class QwenOCR(nn.Module, OCRModel):
    name = "qwen"
    
    main_prompt_template: str = """Transcribe ALL characters from this traditional Chinese text grid.

Grid Layout: {nrows} rows × {ncols} columns = {num_chars} total characters

Reading Instructions:
1. Read row 1 (all {ncols} characters from left to right)
2. Read row 2 (all {ncols} characters from left to right)
3. Continue for ALL {nrows} rows
4. You MUST output EXACTLY {num_chars} characters

Transcription Rules:
- For damaged/unreadable characters, use ▯ (U+25AF)
- Do NOT skip any rows
- Output EXACTLY {num_chars} characters, no more, no less
- Output ONLY the characters, no explanations

Begin transcription:"""


    def __init__(self, nrows: int, ncols: int, max_pixels=512*512, device_map="auto"):
        nn.Module.__init__(self)

        import logging
        # Suppress transformers and huggingface_hub logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        
        self.nrows = nrows
        self.ncols = ncols
        self.grid_size = nrows * ncols

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float16,
            device_map=device_map,
            token=hf_tok
        )

        kwargs = {} if max_pixels is None else {'max_pixels': max_pixels}
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            min_pixels=256*256,
            **kwargs
        )

    @classmethod
    def available(cls) -> bool:
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            return True
        except:
            return False

    def _grid_to_pil(self, grid_tensor: torch.Tensor) -> Image.Image:
        """Convert grid tensor to PIL Image."""
        canvas = (1 - grid_tensor.squeeze()).numpy().astype(np.uint8) * 255
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        return Image.fromarray(canvas)

    @torch.inference_mode()
    def _process_grid(self, image: Image.Image, num_chars: int):
        """Process a single grid image and return characters."""
        # Create prompt with expected character count
        prompt = self.main_prompt_template.format(
            num_chars=num_chars,
            nrows=self.nrows,
            ncols=self.ncols
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        chinese_pattern = r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef▯]+'
        chinese_only = ''.join(re.findall(chinese_pattern, output_text))
        
        return chinese_only

    def predict_with_scores(self, dataset) -> Tuple[List[str], List[float]]:
        """Process dataset in grids and return characters with uncertainties."""
        grid_ds = GridDataset(dataset, k=self.nrows, l=self.ncols)
        
        chars = []
        uncertainties = []
        total_patches = len(dataset)
        
        for grid_idx, grid in enumerate(tqdm(grid_ds, desc=f"{self.name} - Detection", colour="blue")):
            pil_image = self._grid_to_pil(grid)
            
            # How many patches are actually in this grid?
            start_idx = grid_idx * self.grid_size
            expected_chars = min(self.grid_size, total_patches - start_idx)
            
            # Process with expected character count
            chinese_only = self._process_grid(pil_image, expected_chars)[1:]
            
            # Length mismatch means everything uncertain
            length_mismatch = len(chinese_only) != expected_chars
            
            for i in range(expected_chars):
                char = chinese_only[i] if i < len(chinese_only) else "▯"
                is_uncertain = length_mismatch or char == '▯' or char == ""
                
                chars.append(char)
                uncertainties.append(1.0 if is_uncertain else 0.0)
        
        return chars, uncertainties

    def __call__(self, dataset) -> List[str]:
        chars, _ = self.predict_with_scores(dataset)
        return chars
    

# ---------------- registry ----------------

_ALL = [
    TesseractOCR,
    EasyOCRModel,
    CnOCRModel,
    QwenOCR
]


def available_wrappers() -> Dict[str, OCRModel]:
    return {cls.name: cls() for cls in _ALL if cls.available()}

