from __future__ import annotations


def _norm_bbox(bbox: list[float] | tuple[float, ...] | None) -> tuple[float, float, float, float] | None:
    """Normalize a bbox-like value to xyxy floats, or return None if invalid."""
    if bbox is None or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except Exception:
        return None
    return x1, y1, x2, y2


def bbox_area(bbox: list[float] | tuple[float, ...] | None) -> float:
    """Return area for an xyxy bbox; invalid or inverted boxes return 0."""
    norm = _norm_bbox(bbox)
    if norm is None:
        return 0.0
    x1, y1, x2, y2 = norm
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(
    bbox1: list[float] | tuple[float, ...] | None,
    bbox2: list[float] | tuple[float, ...] | None,
) -> float:
    """Return intersection area for two xyxy bboxes."""
    b1 = _norm_bbox(bbox1)
    b2 = _norm_bbox(bbox2)
    if b1 is None or b2 is None:
        return 0.0
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(
    bbox1: list[float] | tuple[float, ...] | None,
    bbox2: list[float] | tuple[float, ...] | None,
) -> float:
    """Return intersection-over-union for two xyxy bboxes."""
    inter = intersection_area(bbox1, bbox2)
    if inter <= 0:
        return 0.0
    union = bbox_area(bbox1) + bbox_area(bbox2) - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio(
    source_bbox: list[float] | tuple[float, ...] | None,
    target_bbox: list[float] | tuple[float, ...] | None,
) -> float:
    """Return intersection(source, target) divided by source bbox area."""
    source_area = bbox_area(source_bbox)
    if source_area <= 0:
        return 0.0
    return intersection_area(source_bbox, target_bbox) / source_area


def is_non_overlap(
    evidence_bbox: list[float] | tuple[float, ...] | None,
    support_bbox: list[float] | tuple[float, ...] | None,
    max_overlap_ratio: float = 0.05,
    max_iou: float = 0.01,
) -> bool:
    """Return True when evidence and support boxes barely overlap."""
    return (
        overlap_ratio(evidence_bbox, support_bbox) <= max_overlap_ratio
        and iou(evidence_bbox, support_bbox) <= max_iou
    )
