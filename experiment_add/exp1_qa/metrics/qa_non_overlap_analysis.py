from __future__ import annotations

from typing import Any

from experiment_add.shared.metrics.overlap_utils import bbox_area, iou, intersection_area


def is_non_overlap_subset(evidence_bbox: Any, support_bbox: Any) -> bool:
    """Return True when support barely overlaps evidence using debug subset rules."""
    evidence_area = bbox_area(evidence_bbox)
    if evidence_area <= 0:
        return False
    overlap = intersection_area(evidence_bbox, support_bbox) / evidence_area
    return overlap < 0.05 or iou(evidence_bbox, support_bbox) < 0.01


def evidence_bbox_for_pipeline(qa_row: dict[str, Any], pipeline: str) -> Any:
    if pipeline == "mineru":
        return qa_row.get("evidence_bbox_mineru")
    if pipeline == "ppstructure":
        return qa_row.get("evidence_bbox_ppstructure")
    return None
