"""Standard terminal-judge metrics: ΔmAP and ΔCER.

Provides:
  - compute_detection_ap():       single-image, single-class AP (VOC all-point)
  - compute_per_class_map():      single-image, per-class mAP (standard VOC)
  - compute_delta_map():          ΔmAP = mAP(clean) − mAP(adv)
  - compute_delta_cer():          ΔCER from full-text pair (standard Levenshtein)
  - compute_delta_cer_from_elements():  block-aligned ΔCER over matched blocks

Metric definitions:
  - CER:  standard Character Error Rate = EditDistance(ref, hyp) / |ref|
          using Levenshtein distance (insertions + deletions + substitutions).
  - mAP:  per-class Average Precision at IoU=0.5 (VOC all-point interpolation),
          averaged across classes present in the ground truth.
          GT and prediction labels are normalised to a canonical 5-class set
          (text, title, table, figure, equation) via _normalize_type() so that
          PubLayNet / DocLayNet GT strings match parser output strings.

These serve as community-standard terminal judges, independent of the custom
B-SLR diagnostic metric, preventing circular validation.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# ── Category normalisation ────────────────────────────────────────────
# Maps every known GT label (PubLayNet / DocLayNet) and parser output
# label to a canonical lowercase type.  Returns None for categories
# that should be excluded from mAP (non-content elements that parsers
# also skip).
#
# Canonical set: text, title, table, figure, equation

_CANONICAL_MAP: Dict[str, Optional[str]] = {
    # ── PubLayNet GT (Title Case) ──
    "Text":              "text",
    "Title":             "title",
    "List":              "text",
    "Table":             "table",
    "Figure":            "figure",
    # ── DocLayNet GT (Title Case / hyphenated) ──
    "Caption":           "text",
    "Footnote":          "text",
    "Formula":           "equation",
    "List-item":         "text",
    "Page-footer":       None,
    "Page-header":       None,
    "Picture":           "figure",
    "Section-header":    "title",
    # ── Parser output labels ──
    "figure_caption":    "text",
    "table_caption":     "text",
    "reference":         "text",
    "list":              "text",
    # ── MinerU block types (CategoryType enum) ──
    "image":             "figure",
    "formula":           "equation",
    "plain_text":        "text",
    "table_footnote":    "text",
    "isolate_formula":   "equation",
    "formula_caption":   "text",
    "embedding":         "equation",
    "isolated":          "equation",
    # ── Non-content labels (skip — parsers also drop these) ──
    "abandon":           None,
    "header":            None,
    "footer":            None,
    # ── Misc aliases ──
    "index":             "text",
    "normal_text":       "text",
    "seal":              None,
}

# Labels that are already in canonical form (no lookup needed).
_CANONICAL_PASSTHROUGH = frozenset({"text", "title", "table", "figure", "equation"})


def _normalize_type(label: str) -> Optional[str]:
    """Map any GT / prediction category label to a canonical type.

    Returns None for categories that should be excluded from AP
    (e.g. page-header / page-footer, which parsers also skip).
    Labels already in canonical form pass through.  Truly unknown
    labels default to ``"text"`` to prevent orphan classes that
    would systematically deflate mAP.
    """
    if label in _CANONICAL_MAP:
        return _CANONICAL_MAP[label]
    low = label.lower()
    if low in _CANONICAL_PASSTHROUGH:
        return low
    if low in _CANONICAL_MAP:
        return _CANONICAL_MAP[low]
    return "text"


# ── helpers ──────────────────────────────────────────────────────────

def _box_iou(box_a: tuple, box_b: tuple) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _elem_to_box(elem: dict) -> tuple:
    bbox = elem.get("bbox")
    if bbox and len(bbox) >= 4:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    hw, hh = elem["w"] / 2, elem["h"] / 2
    return (elem["x"] - hw, elem["y"] - hh, elem["x"] + hw, elem["y"] + hh)


def _norm_bbox(bbox) -> tuple:
    """Normalise bbox to (x1,y1,x2,y2).

    Accepts both xyxy and COCO xywh formats.
    Heuristic: if v3 > v1 and v4 > v2, assume xyxy; else treat as COCO xywh.
    """
    v1, v2, v3, v4 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    if v3 > v1 and v4 > v2:
        return (v1, v2, v3, v4)
    return (v1, v2, v1 + v3, v2 + v4)


def _elem_to_xyxy(elem: dict) -> tuple:
    """Convert parser element {x,y,w,h} (center coords) → (x1,y1,x2,y2)."""
    bbox = elem.get("bbox")
    if bbox and len(bbox) >= 4:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    hw, hh = elem["w"] / 2, elem["h"] / 2
    return (elem["x"] - hw, elem["y"] - hh, elem["x"] + hw, elem["y"] + hh)


# ── AP / mAP ─────────────────────────────────────────────────────────

def compute_detection_ap(
    gt_boxes: List[Dict],
    pred_boxes: List[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """Compute Average Precision for a single class (VOC all-point interpolation).

    Args:
        gt_boxes: list of {"bbox": [x1,y1,x2,y2], ...}
        pred_boxes: list of {"bbox": [x1,y1,x2,y2], "score": float, ...}
                    If no score field, rank-order scores are assigned.
        iou_threshold: IoU matching threshold (default 0.5 = VOC standard)

    Returns:
        AP (float) in [0, 1]
    """
    if not gt_boxes:
        return 1.0 if not pred_boxes else 0.0
    if not pred_boxes:
        return 0.0

    gt_xyxy = [_norm_bbox(g.get("bbox") or [0, 0, 0, 0]) for g in gt_boxes]

    scored = []
    for i, p in enumerate(pred_boxes):
        b = p.get("bbox") or [0, 0, 0, 0]
        s = p.get("score", 1.0 - i * 1e-6)
        scored.append((s, _norm_bbox(b)))
    scored.sort(key=lambda x: -x[0])

    n_gt = len(gt_xyxy)
    matched = [False] * n_gt
    tp = np.zeros(len(scored))
    fp = np.zeros(len(scored))

    for idx, (_, pbox) in enumerate(scored):
        best_iou = 0.0
        best_gt = -1
        for gi, gbox in enumerate(gt_xyxy):
            if matched[gi]:
                continue
            iou = _box_iou(pbox, gbox)
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_iou >= iou_threshold and best_gt >= 0:
            tp[idx] = 1.0
            matched[best_gt] = True
        else:
            fp[idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / n_gt
    precisions = cum_tp / (cum_tp + cum_fp)

    # All-point AP (monotonically-decreasing precision envelope)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx_change = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[idx_change] - mrec[idx_change - 1]) * mpre[idx_change])
    return float(ap)


def _normalize_boxes(boxes: List[Dict]) -> List[Dict]:
    """Return a copy of *boxes* with ``"type"`` normalised via
    :func:`_normalize_type`.  Boxes whose type maps to ``None``
    (e.g. page-header / page-footer) are excluded.
    """
    out: List[Dict] = []
    for b in boxes:
        t = _normalize_type(b.get("type", "unknown"))
        if t is not None:
            out.append({**b, "type": t})
    return out


def compute_per_class_map(
    gt_boxes: List[Dict],
    pred_boxes: List[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """Standard per-class mAP for a single image (VOC AP@IoU).

    GT and prediction labels are first normalised to a canonical
    5-class set (text, title, table, figure, equation) so that
    PubLayNet / DocLayNet GT strings match parser output strings.

    Args:
        gt_boxes:   list of {"bbox": ..., "type": str}
        pred_boxes: list of {"bbox": ..., "type": str, "score": float}
        iou_threshold: IoU threshold (default 0.5 = standard VOC)

    Returns:
        mAP (float) in [0, 1]
    """
    gt_normed = _normalize_boxes(gt_boxes)
    pred_normed = _normalize_boxes(pred_boxes)

    if not gt_normed:
        return 1.0 if not pred_normed else 0.0
    if not pred_normed:
        return 0.0

    classes = set(b["type"] for b in gt_normed)
    class_aps = []
    for cls in classes:
        gt_cls = [b for b in gt_normed if b["type"] == cls]
        pred_cls = [b for b in pred_normed if b["type"] == cls]
        class_aps.append(compute_detection_ap(gt_cls, pred_cls, iou_threshold))

    return float(np.mean(class_aps))


def compute_detection_map(
    gt_boxes_per_image: Dict[str, List[Dict]],
    pred_boxes_per_image: Dict[str, List[Dict]],
    iou_threshold: float = 0.5,
    per_class: bool = False,
) -> Dict[str, float]:
    """Compute mAP across multiple images using per-class AP.

    Args:
        gt_boxes_per_image:   {image_id: [{"bbox":..., "type":...}, ...]}
        pred_boxes_per_image: {image_id: [{"bbox":..., "type":..., "score":...}, ...]}
        per_class: whether to return per-class breakdown

    Returns:
        {"mAP": float} or {"mAP": float, "per_class": {...}}
    """
    all_ids = set(gt_boxes_per_image.keys()) | set(pred_boxes_per_image.keys())
    if not all_ids:
        return {"mAP": 0.0}

    aps = []
    for img_id in all_ids:
        gt = gt_boxes_per_image.get(img_id, [])
        pred = pred_boxes_per_image.get(img_id, [])
        aps.append(compute_per_class_map(gt, pred, iou_threshold))

    result: Dict[str, float] = {"mAP": float(np.mean(aps)) if aps else 0.0}

    if per_class:
        classes = set()
        for boxes in gt_boxes_per_image.values():
            for b in _normalize_boxes(boxes):
                classes.add(b["type"])
        class_aps_dict: Dict[str, float] = {}
        for cls in sorted(classes):
            cls_aps = []
            for img_id in all_ids:
                gt_cls = [b for b in _normalize_boxes(gt_boxes_per_image.get(img_id, [])) if b["type"] == cls]
                pred_cls = [b for b in _normalize_boxes(pred_boxes_per_image.get(img_id, [])) if b["type"] == cls]
                if gt_cls or pred_cls:
                    cls_aps.append(compute_detection_ap(gt_cls, pred_cls, iou_threshold))
            class_aps_dict[cls] = float(np.mean(cls_aps)) if cls_aps else 0.0
        result["per_class"] = class_aps_dict

    return result


def compute_delta_map(
    gt_boxes: List[Dict],
    clean_pred_boxes: List[Dict],
    adv_pred_boxes: List[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """ΔmAP = mAP(clean) − mAP(adversarial) for a single image.

    Uses standard per-class AP@IoU (VOC all-point interpolation).
    Positive values indicate detection performance degradation.

    Args:
        gt_boxes:         ground-truth annotations (PubLayNet/DocLayNet)
        clean_pred_boxes: parser detections on clean image
        adv_pred_boxes:   parser detections on adversarial image
        iou_threshold:    IoU threshold (default 0.5 = VOC standard)

    Returns:
        delta_map: float (positive = degradation)
    """
    map_clean = compute_per_class_map(gt_boxes, clean_pred_boxes, iou_threshold)
    map_adv = compute_per_class_map(gt_boxes, adv_pred_boxes, iou_threshold)
    return map_clean - map_adv


# ── CER ──────────────────────────────────────────────────────────────

def compute_delta_cer(clean_text: str, adv_text: str) -> float:
    """Standard ΔCER using Levenshtein edit distance.

    CER = EditDistance(clean, adv) / len(clean)
    Uses clean OCR output as pseudo-GT reference.

    Args:
        clean_text: OCR full text from clean image
        adv_text:   OCR full text from adversarial image

    Returns:
        delta_cer: float >= 0   (0 = identical, higher = more damage)
    """
    from experiment.metrics import compute_standard_cer
    return compute_standard_cer(clean_text, adv_text)


def compute_delta_cer_from_elements(
    clean_elements: List[Dict],
    adv_elements: List[Dict],
    iou_threshold: float = 0.1,
) -> float:
    """Block-aligned ΔCER using standard Levenshtein CER (unconditional).

    All clean elements with non-empty text contribute to the average.
    Elements with no overlapping adversarial prediction (IoU = 0) receive
    CER = 1.0, representing total text loss.

    Args:
        clean_elements: clean parser output content_list elements
        adv_elements:   adversarial parser output content_list elements
        iou_threshold:  IoU matching threshold (used only for adv_used dedup)

    Returns:
        mean_delta_cer: float >= 0 (standard CER averaged over all text blocks)
    """
    from experiment.metrics import compute_standard_cer

    if not clean_elements:
        return 0.0
    if not adv_elements:
        return 1.0

    adv_boxes = [_elem_to_box(e) for e in adv_elements]
    adv_used = [False] * len(adv_elements)
    cers: List[float] = []

    for ce in clean_elements:
        ct = ce.get("text", "").strip()
        if not ct:
            continue
        cbox = _elem_to_box(ce)
        best_iou = 0.0
        best_idx = -1
        for ai, abox in enumerate(adv_boxes):
            if adv_used[ai]:
                continue
            iou = _box_iou(cbox, abox)
            if iou > best_iou:
                best_iou = iou
                best_idx = ai
        if best_iou > 0 and best_idx >= 0:
            adv_used[best_idx] = True
            at = adv_elements[best_idx].get("text", "")
            cers.append(compute_standard_cer(ct, at))
        else:
            cers.append(1.0)

    if not cers:
        return 1.0
    return sum(cers) / len(cers)
