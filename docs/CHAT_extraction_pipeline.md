# CHAT Character Extraction Pipeline

## Overview

The CHAT OCR model (a Kraken recognition model trained on vertical Chinese text) is integrated into the patch processing pipeline to recognize characters detected by CRAFT. This document describes the complete extraction flow from raw page images to per-character predictions.

## Pipeline Architecture

```
Page Image (H×W grayscale)
    ↓
CRAFT Character Detection → bounding boxes + centroids
    ↓
Layout Analysis → subcolumns (vertical text lanes)
    ↓
For each subcolumn:
    ├─ Crop to tight bbox
    ├─ Binarize (threshold 128)
    ├─ Build synthetic baseline from centroids
    ├─ Run kraken.rpred.rpred() → text + confidences
    └─ Align predictions to CRAFT centroids
```

## Step-by-Step Process

### 1. Subcolumn Layout (`_run_chat_ocr`)

**Input:** Full page image + CRAFT component labels + rectangles dataframe

The `split_to_rectangles` function groups CRAFT detections into columns, and `break_into_subcols` further divides wide columns into vertical lanes to handle multi-column layouts.

**Output:** List of subcolumns, each containing a vertical sequence of character labels

---

### 2. Crop Extraction (`_predict_subcol`)

For each subcolumn lane:

**a) Collect character labels:**
```python
subcol_labels = [lbl for lbl in lane if lbl in page_df]
```

**b) Compute tight bounding box:**
```python
x0 = min(left for all characters)
y0 = min(top for all characters)
x1 = max(left + width for all characters)
y1 = max(top + height for all characters)
subcol_image = page_image[y0:y1, x0:x1]
```

**c) Binarize:**
```python
pil_img = Image.fromarray(subcol_image).point(
    lambda x: 0 if x < 128 else 255, "1"
)
```
PIL mode `"1"` creates a black-and-white bitmap (threshold at grayscale value 128).

---

### 3. Synthetic Baseline Construction

The CHAT model (via Kraken's `rpred` pipeline) expects a **baseline-based segmentation**, but we don't have a segmentation model. Instead, we construct a synthetic baseline from CRAFT centroids:

**a) Extract relative centroids:**
```python
centers_rel = [(cy - y0, cx - x0) for (cy, cx) in CRAFT_centroids]
```

**b) Build vertical baseline:**
```python
cx_median = median([cx for _, cx in centers_rel])
baseline = [(cx_median, 0), (cx_median, height)]
boundary = [(0, 0), (width, 0), (width, height), (0, height)]
```

This creates a straight vertical line at the median x-coordinate, spanning the full crop height. This tells `rpred` where the text column runs without requiring a separate segmentation model.

**c) Create Kraken containers:**
```python
line = BaselineLine(id="0", baseline=baseline, boundary=boundary)
seg = Segmentation(
    type="baselines",
    imagename="",
    text_direction="vertical-rl",
    script_detection=False,
    lines=[line],
)
```

---

### 4. Recognition via `kraken.rpred`

**Why rpred?**

Calling `model.predict()` directly on the raw crop produces 0 detections because:
- The model expects a **dewarped single-line image** prepared by Kraken's internal pipeline
- Without `rpred`, the image gets transformed into a near-constant tensor
- The `tensor.max() == tensor.min()` guard in the model fires → empty output

**Solution:**
```python
pred_it = kraken.rpred.rpred(
    model.net,           # TorchSeqRecognizer
    pil_img,             # PIL Image mode "1"
    seg,                 # Segmentation with synthetic baseline
    pad=model.pad,       # Horizontal padding (default 16)
)
```

**What rpred does internally:**
1. Extracts the line region defined by `baseline` and `boundary`
2. Dewarps the line image (straightening curved baselines)
3. Normalizes to the model's expected input size
4. Runs `model.predict()` on the transformed tensor
5. Scales character positions back to original image coordinates

**Output:**
```python
for record in pred_it:
    pred_chars = list(record._prediction)      # ["年", "月", "日", ...]
    pred_confs = list(record._confidences)     # [0.98, 0.95, 0.87, ...]
```

---

### 5. Alignment to CRAFT Centroids

CRAFT may detect a different number of characters than the model recognizes:

**Truncate if too many predictions:**
```python
if len(pred_chars) > len(subcol_labels):
    pred_chars = pred_chars[:len(subcol_labels)]
    pred_confs = pred_confs[:len(subcol_labels)]
```

**Pad if too few predictions:**
```python
elif len(pred_chars) < len(subcol_labels):
    pred_chars += ["▯"] * (len(subcol_labels) - len(pred_chars))
    pred_confs += [0.0] * (len(subcol_labels) - len(pred_chars))
```

---

### 6. Write-back to DataFrame

Each character label is mapped to its DataFrame index:

```python
for label, char, conf in zip(subcol_labels, pred_chars, pred_confs):
    idx = label_to_idx[label]
    page_df.at[idx, 'char_chat'] = char
    page_df.at[idx, 'conf_chat'] = conf
```

**Result:** Every CRAFT-detected character in `page_df` now has:
- `char_chat`: recognized character (or "▯" if model failed)
- `conf_chat`: confidence score (0.0 to 1.0)

---

## Key Design Decisions

### Why synthetic baselines?

- **No segmentation model needed**: CRAFT already provides character locations
- **Minimal overhead**: Just compute median x-coordinate from centroids
- **Robust**: Works for straight vertical columns (the common case)

### Why binarization?

The CHAT model demo (and Kraken's training pipeline) expects binary images (PIL mode `"1"`). Binarization:
- Removes noise and background texture
- Standardizes input format
- Improves model robustness

### Why rpred instead of direct predict()?

Direct `model.predict()` assumes the input is already a properly formatted line image. `rpred` provides:
- **Line extraction** from baselines (crops and dewarps the region)
- **Input normalization** (resizes, pads, converts to tensor)
- **Coordinate mapping** (scales predictions back to original image space)

Without `rpred`, the model receives improperly formatted input and produces empty predictions.

---

## Example Output

For a vertical column containing "年月日":

```
Input:
  - CRAFT detections: 3 bboxes + centroids
  - Subcolumn crop: 180×80 grayscale image

Synthetic baseline:
  - cx_median: 40 (median of centroid x-coords)
  - baseline: [(40, 0), (40, 180)]
  - boundary: [(0, 0), (80, 0), (80, 180), (0, 180)]

rpred output:
  - pred_chars: ["年", "月", "日"]
  - pred_confs: [0.982, 0.956, 0.873]

Write-back:
  - label 147 → char_chat="年", conf_chat=0.982
  - label 148 → char_chat="月", conf_chat=0.956
  - label 149 → char_chat="日", conf_chat=0.873
```

---

## Performance Characteristics

- **CPU overhead**: Minimal (just crop + binarize)
- **GPU work**: One `rpred` call per subcolumn (~5-20 chars each)
- **Batching**: Currently single-line; could batch multiple subcolumns for speedup
- **Accuracy**: Dependent on CRAFT quality and CHAT model training

---

## Code References

| Component | Location |
|-----------|----------|
| Main pipeline | `src/patch_processing/processor.py:193` (`_run_chat_ocr`) |
| Subcolumn processing | `src/patch_processing/processor.py:233` (`_predict_subcol`) |
| Layout splitting | `src/layout_analysis/parsing.py` (`split_to_rectangles`, `break_into_subcols`) |
| CHAT wrapper | `src/ocr/chat.py` (`ModelWrapper`) |

---

## Known Limitations

1. **Straight baselines only**: Synthetic baseline is a vertical line; won't handle curved text
2. **Single-column assumption**: Each subcolumn treated independently; no context between columns
3. **No character segmentation**: Relies entirely on CRAFT bboxes; can't fix CRAFT errors
4. **Padding strategy**: Simple truncate/pad with "▯"; could use alignment algorithms (e.g., DTW)

---

## Future Improvements

- [ ] Use actual segmentation model (CHAT's `chat_seg.mlmodel`) instead of synthetic baselines
- [ ] Batch multiple subcolumns per `rpred` call for GPU efficiency
- [ ] Implement character-level alignment (CTC-style) instead of simple truncate/pad
- [ ] Handle curved baselines by fitting splines to CRAFT centroids
- [ ] Add confidence-based filtering (reject chars with conf < threshold)
