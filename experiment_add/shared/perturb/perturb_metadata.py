from __future__ import annotations

from typing import Any

import numpy as np


def mask_bbox(mask: np.ndarray) -> list[float] | None:
    """Return overall xyxy bbox for a binary support mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def compute_tor(mask: np.ndarray, width: int, height: int) -> float:
    """Compute actual total occlusion ratio from a support mask."""
    if width <= 0 or height <= 0:
        return 0.0
    return float((mask > 0).sum() / (width * height))


def make_metadata_record(
    page_id: str,
    condition: str,
    probe_id: str,
    probe_family: str,
    image_path: str,
    perturbed_image_path: str,
    page_width: int,
    page_height: int,
    support_mask: np.ndarray,
    target_block: dict[str, Any] | None,
    placement_info: str,
    random_seed: int,
) -> dict[str, Any]:
    """Create a schema-compatible perturbation metadata record."""
    bbox = mask_bbox(support_mask)
    return {
        "page_id": page_id,
        "condition": condition,
        "probe_id": probe_id,
        "probe_family": probe_family,
        "image_path": image_path,
        "perturbed_image_path": perturbed_image_path,
        "page_width": page_width,
        "page_height": page_height,
        "support_bbox": bbox,
        "support_bboxes": [bbox] if bbox else [],
        "support_mask_area": int((support_mask > 0).sum()),
        "TOR": compute_tor(support_mask, page_width, page_height),
        "target_block_id": target_block.get("block_id") if target_block else None,
        "target_layout_type": target_block.get("layout_type") if target_block else None,
        "placement_info": placement_info,
        "random_seed": random_seed,
    }
