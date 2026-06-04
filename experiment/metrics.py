from __future__ import annotations

from typing import List

import cv2
import editdistance
import numpy as np
from skimage.metrics import structural_similarity as calc_ssim


def compute_standard_cer(reference: str, hypothesis: str) -> float:
    """Standard Character Error Rate using Levenshtein edit distance.

    CER = EditDistance(reference, hypothesis) / len(reference)

    Preprocessing: strip whitespace + case-fold to lowercase,
    consistent with TextSim preprocessing used in B-SLR matching.
    Returns 0.0 when reference is empty (undefined CER → no damage).
    """
    ref = reference.strip().lower()
    hyp = hypothesis.strip().lower()
    if not ref:
        return 0.0
    if ref == hyp:
        return 0.0
    dist = editdistance.eval(ref, hyp)
    return dist / len(ref)


def extract_elements_from_result(result) -> list:
    if result is None:
        return []
    content_list = result.raw_output.get("content_list", [])
    elements = []
    for item in content_list:
        if isinstance(item, dict):
            bbox = item.get("bbox", [])
            if bbox and len(bbox) >= 4:
                try:
                    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    elements.append(
                        {
                            "x": int((x1 + x2) / 2),
                            "y": int((y1 + y2) / 2),
                            "w": int(x2 - x1),
                            "h": int(y2 - y1),
                            "type": item.get("type", "text"),
                            "id": len(elements),
                            "text": str(item.get("text", ""))[:120],
                        }
                    )
                except Exception:
                    continue
    return elements


def extract_span_elements_from_result(result) -> list:
    if result is None:
        return []
    mj = result.raw_output.get("middle_json", {})
    elements = []
    for page in mj.get("pdf_info", []):
        for pb in page.get("para_blocks", []):
            for line in pb.get("lines", []):
                for span in line.get("spans", []):
                    bbox = span.get("bbox", [])
                    if not bbox or len(bbox) < 4:
                        continue
                    try:
                        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                        if x2 <= x1 or y2 <= y1:
                            continue
                        elements.append(
                            {
                                "x": int((x1 + x2) / 2),
                                "y": int((y1 + y2) / 2),
                                "w": int(x2 - x1),
                                "h": int(y2 - y1),
                                "type": span.get("type", "text"),
                                "id": len(elements),
                                "text": str(span.get("content", ""))[:120],
                            }
                        )
                    except Exception:
                        continue
    return elements


def extract_full_text_from_result(result) -> str:
    if result is None:
        return ""
    content_list = result.raw_output.get("content_list", [])
    text_items = []
    for item in content_list:
        if isinstance(item, dict) and item.get("text"):
            bbox = item.get("bbox", [0, 0, 0, 0])
            y = float(bbox[1]) if len(bbox) >= 2 else 0
            x = float(bbox[0]) if len(bbox) >= 1 else 0
            text_items.append((y, x, str(item["text"])))
    text_items.sort(key=lambda t: (t[0], t[1]))
    return " ".join(t[2] for t in text_items)


def compute_lcs_length(s1: str, s2: str, max_len: int = 2000) -> int:
    # Truncate to avoid O(m*n) blowup on long OCR texts
    s1, s2 = s1[:max_len], s2[:max_len]
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0
    # Use difflib SequenceMatcher for speed (C-accelerated)
    from difflib import SequenceMatcher
    sm = SequenceMatcher(None, s1, s2, autojunk=False)
    return sum(block.size for block in sm.get_matching_blocks())


def compute_character_slr(gt_text: str, pred_text: str) -> float:
    gt_text = gt_text.strip()
    pred_text = pred_text.strip()
    if not gt_text:
        return 0.0
    if not pred_text:
        return 1.0
    max_len = 2000
    gt_trunc = gt_text[:max_len]
    lcs = compute_lcs_length(gt_trunc, pred_text[:max_len], max_len=max_len)
    return 1.0 - lcs / len(gt_trunc)


def _elem_to_box(elem: dict) -> tuple:
    hw, hh = elem["w"] / 2, elem["h"] / 2
    return elem["x"] - hw, elem["y"] - hh, elem["x"] + hw, elem["y"] + hh


def _box_iou(box_a: tuple, box_b: tuple) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def text_similarity(text1: str, text2: str) -> float:
    """Longest Common Subsequence (LCS) similarity.

    Returns len(LCS) / max(len(text1), len(text2)).
    """
    text1 = text1.strip().lower()
    text2 = text2.strip().lower()
    if not text1 or not text2:
        return 0.0
    if text1 == text2:
        return 1.0
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_length = dp[m][n]
    return lcs_length / max(m, n)


def identify_lost_boxes(orig_elements: List[dict], adv_elements: List[dict], iou_threshold: float = 0.1) -> List[int]:
    if not adv_elements:
        return list(range(len(orig_elements)))
    adv_boxes = [_elem_to_box(e) for e in adv_elements]
    lost_indices = []
    for i, oe in enumerate(orig_elements):
        ob = _elem_to_box(oe)
        if not any(_box_iou(ob, ab) >= iou_threshold for ab in adv_boxes):
            lost_indices.append(i)
    return lost_indices


def calculate_b_slr(
    original_elements: list,
    adversarial_elements: list,
    iou_threshold: float = 0.1,
    text_sim_threshold: float = 0.5,
    iou_only: bool = False,
) -> float:
    """Block-level Text Loss Rate.

    A block is "lost" when:
    1. No adversarial block has IoU >= iou_threshold (geometric miss), OR
      2. The best-IoU match exists but text similarity < text_sim_threshold
         (text degradation) — skipped when ``iou_only=True``.

    Setting ``iou_only=True`` produces an IoU-only B-SLR that is comparable
    with YOLO (which has no text output).  This is the ablation described in
    §5.10 for eliminating the evaluation-standard confound.
    """
    if not original_elements:
        return 0.0
    if not adversarial_elements:
        return 1.0
    total = len(original_elements)
    adv_boxes = [_elem_to_box(e) for e in adversarial_elements]
    lost_count = 0
    for orig_elem in original_elements:
        orig_box = _elem_to_box(orig_elem)
        orig_text = orig_elem.get("text", "").strip()
        best_iou = 0.0
        best_adv_idx = -1
        for j, adv_box in enumerate(adv_boxes):
            iou = _box_iou(orig_box, adv_box)
            if iou > best_iou:
                best_iou = iou
                best_adv_idx = j
        if best_iou < iou_threshold:
            lost_count += 1
        elif not iou_only and orig_text:
            adv_text = adversarial_elements[best_adv_idx].get("text", "").strip()
            if text_similarity(orig_text, adv_text) < text_sim_threshold:
                lost_count += 1
    return lost_count / total


def compute_eir_from_detection(original_spans: list, adversarial_spans: list, iou_threshold: float = 0.1) -> float:
    if not original_spans:
        return 0.0
    if not adversarial_spans:
        return 1.0
    total = len(original_spans)
    adv_boxes = [_elem_to_box(e) for e in adversarial_spans]
    interfered_count = 0
    for orig_span in original_spans:
        orig_box = _elem_to_box(orig_span)
        if max((_box_iou(orig_box, ab) for ab in adv_boxes), default=0.0) < iou_threshold:
            interfered_count += 1
    return interfered_count / total


compute_ncsic_from_detection = compute_eir_from_detection  # backward compat alias


def calculate_ssim(original: np.ndarray, adversarial: np.ndarray) -> float:
    try:
        g1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
        g2 = cv2.cvtColor(adversarial, cv2.COLOR_BGR2GRAY) if len(adversarial.shape) == 3 else adversarial
        score, _ = calc_ssim(g1, g2, full=True)
        return float(score)
    except Exception:
        return 1.0


# L1 — TOR (Total Occlusion Ratio): |S'| / (W × H)
def compute_tor(mask: np.ndarray, img_shape: tuple[int, int] | tuple[int, int, int]) -> float:
    h, w = img_shape[:2]
    return float(mask.astype(np.uint8).sum() / (h * w))


# L3 — BPO (Boundary Pixel Overlap): |S' ∩ M_anchor| / |M_anchor|
def compute_ncsic_mask(mask: np.ndarray, m_anchor: np.ndarray) -> float:
    denom = int(m_anchor.astype(np.uint8).sum())
    if denom == 0:
        return 0.0
    overlap = np.logical_and(mask > 0, m_anchor > 0).sum()
    return float(overlap / denom)


# Alias kept for backward compat; primary name is compute_bpo.
compute_bpo = compute_ncsic_mask
compute_bpo_raw = compute_ncsic_mask  # L3 — BPO base computation


def compute_acr(mask: np.ndarray, annotations: list) -> float:
    """L2 — ACR (Annotation Coverage Ratio).

    ACR(P, L) = |S' ∩ ⋃_i bbox(b_i)| / |⋃_i bbox(b_i)|

    Fraction of annotated region area covered by the probe mask.
    """
    if mask is None or not annotations:
        return 0.0
    h, w = mask.shape[:2]
    # Build union annotation mask
    ann_mask = np.zeros((h, w), dtype=np.uint8)
    for ann in annotations:
        bbox = ann.get("bbox", [])
        if not bbox or len(bbox) < 4:
            continue
        x1 = max(0, int(round(bbox[0])))
        y1 = max(0, int(round(bbox[1])))
        x2 = min(w, int(round(bbox[2])))
        y2 = min(h, int(round(bbox[3])))
        if x2 > x1 and y2 > y1:
            ann_mask[y1:y2, x1:x2] = 1
    denom = int(ann_mask.sum())
    if denom == 0:
        return 0.0
    overlap = int(np.logical_and(mask > 0, ann_mask > 0).sum())
    return float(overlap / denom)


compute_ioa = compute_acr  # backward compat alias


def compute_boc(mask: np.ndarray, annotations: list, overlap_px: int = 1) -> float:
    """L4 — BOC (Block Overlap Count).

    BOC(P, L) = |{b_i ∈ L : bbox(b_i) ∩ S' ≠ ∅}| / |L|

    Fraction of GT blocks whose bbox overlaps the probe mask.
    This is the block-level analogue of EIR (which operates on parser output elements).
    """
    if mask is None or not annotations:
        return 0.0
    h, w = mask.shape[:2]
    interfered = 0
    for ann in annotations:
        bbox = ann.get("bbox", [])
        if not bbox or len(bbox) < 4:
            continue
        x1 = max(0, int(round(bbox[0])))
        y1 = max(0, int(round(bbox[1])))
        x2 = min(w, int(round(bbox[2])))
        y2 = min(h, int(round(bbox[3])))
        if x2 > x1 and y2 > y1 and int(mask[y1:y2, x1:x2].sum()) >= overlap_px:
            interfered += 1
    total = len(annotations)
    return float(interfered / total) if total > 0 else 0.0


def compute_eir(orig_spans: list, mask: np.ndarray, overlap_px: int = 1) -> float:
    """L5 — EIR (Element Interference Ratio): fraction of parser output
    structural elements whose bbox overlaps the probe mask.

    EIR(P, L) = |{e ∈ E : bbox(e) ∩ S' ≠ ∅}| / |E|

    Measures how many content_list elements the perturbation *interferes with*.
    A thin line through dense text can have low TOR but high EIR; a large blob
    in whitespace can have high TOR but low EIR.
    """
    if not orig_spans or mask is None:
        return 0.0
    h, w = mask.shape[:2]
    interfered = 0
    for span in orig_spans:
        x1 = max(0, span["x"] - span["w"] // 2)
        y1 = max(0, span["y"] - span["h"] // 2)
        x2 = min(w, x1 + span["w"])
        y2 = min(h, y1 + span["h"])
        if x2 > x1 and y2 > y1 and int(mask[y1:y2, x1:x2].sum()) >= overlap_px:
            interfered += 1
    return interfered / len(orig_spans)


compute_geometric_ncsic = compute_eir  # backward compat alias


def _elem_is_occluded(elem: dict, probe_mask: np.ndarray, threshold: float = 0.3) -> bool:
    """Check whether *threshold* fraction of an element's bbox is covered by *probe_mask*."""
    h, w = probe_mask.shape[:2]
    x1 = max(0, elem["x"] - elem["w"] // 2)
    y1 = max(0, elem["y"] - elem["h"] // 2)
    x2 = min(w, x1 + elem["w"])
    y2 = min(h, y1 + elem["h"])
    if x2 <= x1 or y2 <= y1:
        return False
    area = (x2 - x1) * (y2 - y1)
    if area == 0:
        return False
    covered = int(probe_mask[y1:y2, x1:x2].astype(bool).sum())
    return (covered / area) >= threshold


def calculate_decomposed_b_slr(
    original_elements: list,
    adversarial_elements: list,
    probe_mask: np.ndarray | None = None,
    iou_threshold: float = 0.1,
    text_sim_threshold: float = 0.5,
    occlusion_frac: float = 0.3,
) -> dict:
    """Pathway decomposition of B-SLR following Eqs. (12)–(20) of the paper.

    The failure set ``U`` is defined *exactly* by the same criterion that
    ``calculate_b_slr`` uses for ``R_fail``:

        x in U  iff   best_iou(x) < tau_iou  OR
                      (best_iou(x) >= tau_iou AND orig_text != "" AND
                       text_similarity(orig, adv) < tau_text).

    ``U`` is first partitioned by the occlusion indicator ``omega(x)`` (Eq. 13):

        U_miss = {x in U : rho(x, P) >= eta_occ}
        U_topo = U \\ U_miss.

    ``U_topo`` is then sub-partitioned into the layout-aware failure types
    (Eqs. 18–20):

        merge    : exists u != x with m(u) = m(x)              (many-to-one)
        misclass : x not in merge, c(x) != c(m(x)), IoU >= tau (type flipped)
        degraded : remainder of U_topo.

    By construction (Eq. 16):

        B-SLR = SLR_miss + SLR_topo
              = (n_miss + n_merge + n_misclass + n_degraded) / |E|,

    and this implementation preserves that identity exactly.
    """
    zeros = {
        "SLR_miss": 0.0, "SLR_topo": 0.0, "B_SLR": 0.0,
        "n_miss": 0, "n_merge": 0, "n_misclass": 0, "n_degraded": 0,
        "n_intact": 0, "n_total": 0,
    }
    if not original_elements:
        return zeros
    total = len(original_elements)

    if not adversarial_elements:
        if probe_mask is not None:
            n_occ = sum(
                1 for e in original_elements
                if _elem_is_occluded(e, probe_mask, occlusion_frac)
            )
        else:
            n_occ = 0
        n_miss = n_occ
        n_degraded = total - n_occ
        return {
            "SLR_miss": n_miss / total,
            "SLR_topo": n_degraded / total,
            "B_SLR": 1.0,
            "n_miss": n_miss, "n_merge": 0, "n_misclass": 0,
            "n_degraded": n_degraded, "n_intact": 0, "n_total": total,
        }

    adv_boxes = [_elem_to_box(e) for e in adversarial_elements]

    gt_best: list[tuple[int, float]] = []
    for oe in original_elements:
        ob = _elem_to_box(oe)
        best_iou, best_j = 0.0, -1
        for j, ab in enumerate(adv_boxes):
            iou = _box_iou(ob, ab)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        gt_best.append((best_j, best_iou))

    adv_to_gt: dict[int, list[int]] = {}
    for i, (bj, bi) in enumerate(gt_best):
        if bj >= 0 and bi >= iou_threshold:
            adv_to_gt.setdefault(bj, []).append(i)

    failed: list[int] = []
    for i, oe in enumerate(original_elements):
        best_j, best_iou = gt_best[i]
        if best_iou < iou_threshold:
            failed.append(i)
            continue
        orig_text = oe.get("text", "").strip()
        if orig_text:
            adv_text = adversarial_elements[best_j].get("text", "").strip()
            if text_similarity(orig_text, adv_text) < text_sim_threshold:
                failed.append(i)

    n_miss = n_merge = n_misclass = n_degraded = 0
    for i in failed:
        oe = original_elements[i]
        best_j, best_iou = gt_best[i]

        if probe_mask is not None and _elem_is_occluded(oe, probe_mask, occlusion_frac):
            n_miss += 1
            continue

        if best_j >= 0 and best_iou >= iou_threshold and len(adv_to_gt.get(best_j, [])) > 1:
            n_merge += 1
            continue

        if best_j >= 0 and best_iou >= iou_threshold:
            ot = oe.get("type", "")
            at = adversarial_elements[best_j].get("type", "")
            if ot and at and ot != at:
                n_misclass += 1
                continue

        n_degraded += 1

    n_intact = total - len(failed)
    assert n_miss + n_merge + n_misclass + n_degraded == len(failed)

    return {
        "SLR_miss": n_miss / total,
        "SLR_topo": (n_merge + n_misclass + n_degraded) / total,
        "B_SLR": len(failed) / total,
        "n_miss": n_miss,
        "n_merge": n_merge,
        "n_misclass": n_misclass,
        "n_degraded": n_degraded,
        "n_intact": n_intact,
        "n_total": total,
    }


def compute_splash_ratio(orig_elements: list, adv_elements: list, mask: np.ndarray, margin: int = 20) -> float:
    if not orig_elements:
        return 0.0
    lost = identify_lost_boxes(orig_elements, adv_elements)
    if not lost:
        return 0.0
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    far_lost = 0
    for idx in lost:
        e = orig_elements[idx]
        x, y = e["x"], e["y"]
        near = (min_x - margin <= x <= max_x + margin) and (min_y - margin <= y <= max_y + margin)
        if not near:
            far_lost += 1
    return float(far_lost / max(1, len(lost)))


def calculate_ocr_ablation_metrics(
    original_elements: list,
    adversarial_elements: list,
    iou_threshold: float = 0.1,
    text_sim_threshold: float = 0.5,
) -> dict:
    """Text-degradation ablation: unconditional CER over all text-bearing blocks.

    E_matched = {e in E_clean | txt(e) != epsilon}  (no detection filter).
    For each original block with non-empty text, find the best-IoU adversarial
    match.  If IoU > 0, compute CER against that match; if IoU = 0 (total
    detection miss), assign CER_e = 1.  TextSim is computed only for
    geometrically-matched pairs (IoU >= iou_threshold) for B-SLR diagnostics.

    Returns:
        TextSim_matched_mean : mean TextSim across IoU-matched pairs (1.0 = no damage)
        TextSim_matched_min  : worst-case TextSim among IoU-matched pairs
        CER_matched_mean     : unconditional CER across all text elements (0.0 = no damage)
        n_matched            : number of text-bearing clean elements (= |E_matched|)
        n_text_degraded      : IoU-matched blocks with TextSim < text_sim_threshold
        B_SLR_text_only      : n_text_degraded / n_total (OCR-only loss rate)
    """
    result = {
        "TextSim_matched_mean": 1.0,
        "TextSim_matched_min": 1.0,
        "CER_matched_mean": 0.0,
        "n_matched": 0,
        "n_text_degraded": 0,
        "B_SLR_text_only": 0.0,
    }
    if not original_elements or not adversarial_elements:
        if original_elements and not adversarial_elements:
            result["TextSim_matched_mean"] = 0.0
            result["TextSim_matched_min"] = 0.0
            result["CER_matched_mean"] = 1.0
            n_text = sum(1 for e in original_elements if e.get("text", "").strip())
            result["n_matched"] = n_text
            result["B_SLR_text_only"] = 0.0
        return result

    total = len(original_elements)
    adv_boxes = [_elem_to_box(e) for e in adversarial_elements]

    sims: list[float] = []
    cers: list[float] = []
    n_degraded = 0

    for orig_elem in original_elements:
        orig_box = _elem_to_box(orig_elem)
        orig_text = orig_elem.get("text", "").strip()
        if not orig_text:
            continue

        best_iou, best_j = 0.0, -1
        for j, ab in enumerate(adv_boxes):
            iou = _box_iou(orig_box, ab)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou > 0 and best_j >= 0:
            adv_text = adversarial_elements[best_j].get("text", "").strip()
            cer = compute_standard_cer(orig_text, adv_text)
            cers.append(cer)
            if best_iou >= iou_threshold:
                sim = text_similarity(orig_text, adv_text)
                sims.append(sim)
                if sim < text_sim_threshold:
                    n_degraded += 1
        else:
            cers.append(1.0)

    if not cers:
        result["CER_matched_mean"] = 1.0
        result["n_matched"] = 0
        return result

    if sims:
        result["TextSim_matched_mean"] = sum(sims) / len(sims)
        result["TextSim_matched_min"] = min(sims)
    result["CER_matched_mean"] = sum(cers) / len(cers)
    result["n_matched"] = len(cers)
    result["n_text_degraded"] = n_degraded
    result["B_SLR_text_only"] = n_degraded / total if total > 0 else 0.0

    return result
