from __future__ import annotations

from copy import deepcopy
from typing import Any


def _bbox_key(block: dict[str, Any]) -> tuple[float, float]:
    bbox = block.get("bbox") or [0, 0, 0, 0]
    try:
        return float(bbox[1]), float(bbox[0])
    except Exception:
        return 0.0, 0.0


def sort_blocks_by_bbox(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a new list of text-bearing blocks sorted by y then x bbox coords."""
    text_blocks = [deepcopy(block) for block in blocks if str(block.get("text", "")).strip()]
    return sorted(text_blocks, key=_bbox_key)


def assign_reading_order(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return copied blocks sorted by bbox with `reading_order` assigned.

    The input list and its dictionaries are not modified.
    """
    ordered = sort_blocks_by_bbox(blocks)
    for idx, block in enumerate(ordered):
        block["reading_order"] = idx
    return ordered


def build_page_text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    """Build page text by joining sorted non-empty block texts with newlines."""
    ordered = sort_blocks_by_bbox(blocks)
    return "\n".join(str(block.get("text", "")).strip() for block in ordered)
