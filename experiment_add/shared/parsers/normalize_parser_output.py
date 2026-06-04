from __future__ import annotations

from pathlib import Path
from typing import Any

from experiment_add.shared.text.block_ordering import build_page_text_from_blocks, sort_blocks_by_bbox


def _get_value(item: Any, key: str, default: Any = None) -> Any:
    """Read a key from a dict-like or object-like item."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def extract_text(parse_result: Any) -> str:
    """Extract full text from a ParseResult-like object."""
    text = _get_value(parse_result, "text", "")
    return "" if text is None else str(text)


def extract_layout_items(parse_result: Any) -> list[Any]:
    """Extract layout items from ParseResult.layout or raw_output.content_list."""
    layout = _get_value(parse_result, "layout", None)
    if layout:
        return list(layout)
    raw_output = _get_value(parse_result, "raw_output", None) or {}
    if isinstance(raw_output, dict):
        content_list = raw_output.get("content_list", [])
        if content_list:
            return list(content_list)
    return []


def normalize_bbox(bbox: Any) -> list[float] | None:
    """Normalize bbox-like values to `[x1, y1, x2, y2]` pixel coordinates."""
    if bbox is None:
        return None
    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()
    try:
        values = list(bbox)
    except TypeError:
        return None
    if len(values) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def normalize_layout_type(raw_type: Any) -> str:
    """Normalize a parser layout type to a non-empty string."""
    if raw_type is None:
        return "unknown"
    value = str(raw_type).strip()
    return value if value else "unknown"


def normalize_block_text(raw_text: Any) -> str:
    """Normalize block text to a string without treating None as literal text."""
    if raw_text is None:
        return ""
    return str(raw_text).strip()


def build_blocks_from_layout(layout_items: list[Any]) -> list[dict[str, Any]]:
    """Build normalized blocks from parser layout items.

    Invalid bboxes are skipped. The returned list is sorted by bbox y/x and each
    block receives a zero-based reading_order. Input items are not modified.
    """
    blocks: list[dict[str, Any]] = []
    for item in layout_items:
        bbox = normalize_bbox(_get_value(item, "bbox", None))
        if bbox is None:
            continue
        blocks.append(
            {
                "block_id": "",
                "layout_type": normalize_layout_type(_get_value(item, "type", _get_value(item, "layout_type", None))),
                "bbox": bbox,
                "text": normalize_block_text(_get_value(item, "text", "")),
                "confidence": _get_value(item, "confidence", _get_value(item, "score", None)),
                "reading_order": 0,
            }
        )

    ordered = sort_blocks_by_bbox(blocks)
    for idx, block in enumerate(ordered):
        block["block_id"] = f"b{idx + 1:04d}"
        block["reading_order"] = idx
    return ordered


def _base_record(
    page_id: str,
    pipeline: str,
    condition: str,
    image_path: str,
    width: int | str,
    height: int | str,
) -> dict[str, Any]:
    """Create the common output record skeleton."""
    return {
        "page_id": page_id,
        "pipeline": pipeline,
        "condition": condition,
        "image_path": image_path,
        "width": int(float(width)) if str(width).strip() else 0,
        "height": int(float(height)) if str(height).strip() else 0,
        "page_text": "",
        "blocks": [],
        "parser_status": "failed",
        "error_message": None,
        "raw_output_path": None,
    }


def normalize_parse_result(
    parse_result: Any,
    page_id: str,
    pipeline: str,
    condition: str,
    image_path: str,
    width: int | str,
    height: int | str,
    raw_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convert a ParseResult-like object into the experiment_add clean parse schema."""
    record = _base_record(page_id, pipeline, condition, image_path, width, height)
    layout_items = extract_layout_items(parse_result)
    blocks = build_blocks_from_layout(layout_items)
    result_text = extract_text(parse_result).strip()
    page_text = result_text if result_text else build_page_text_from_blocks(blocks)

    record.update(
        {
            "page_text": page_text,
            "blocks": blocks,
            "parser_status": "success" if blocks and page_text else "empty",
            "error_message": None,
            "raw_output_path": str(raw_output_path) if raw_output_path else None,
        }
    )
    return record


def make_failed_record(
    page_id: str,
    pipeline: str,
    condition: str,
    image_path: str,
    width: int | str,
    height: int | str,
    error_message: str,
) -> dict[str, Any]:
    """Create a schema-compliant failed parser record."""
    record = _base_record(page_id, pipeline, condition, image_path, width, height)
    record["parser_status"] = "failed"
    record["error_message"] = error_message
    return record


def make_empty_record(
    page_id: str,
    pipeline: str,
    condition: str,
    image_path: str,
    width: int | str,
    height: int | str,
) -> dict[str, Any]:
    """Create a schema-compliant empty parser record."""
    record = _base_record(page_id, pipeline, condition, image_path, width, height)
    record["parser_status"] = "empty"
    return record
