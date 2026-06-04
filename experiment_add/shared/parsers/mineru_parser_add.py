from __future__ import annotations

from pathlib import Path
from typing import Any

from experiment_add.shared.parsers.normalize_parser_output import make_failed_record, normalize_parse_result


def create_mineru_parser(config: dict[str, Any] | None = None):
    """Create the original MinerUParser using parser.yaml settings."""
    from parsers.mineru_parser import MinerUParser

    cfg = (config or {}).get("mineru", config or {})
    return MinerUParser(
        formula_enable=cfg.get("formula_enable", True),
        table_enable=cfg.get("table_enable", False),
        lang=cfg.get("lang", "en"),
    )


def parse_mineru_page(
    record: dict[str, Any],
    config: dict[str, Any],
    condition: str = "clean",
    parser: Any | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Parse one manifest record with the original MinerUParser."""
    page_id = str(record.get("page_id", ""))
    image_path = str(record.get("image_path", ""))
    width = record.get("width", 0)
    height = record.get("height", 0)
    root = Path(project_root) if project_root is not None else Path.cwd()
    resolved_image = Path(image_path)
    if not resolved_image.is_absolute():
        resolved_image = root / resolved_image

    try:
        active_parser = parser if parser is not None else create_mineru_parser(config)
        parse_result = active_parser.parse(str(resolved_image))
        return normalize_parse_result(parse_result, page_id, "mineru", condition, image_path, width, height)
    except Exception as exc:
        return make_failed_record(page_id, "mineru", condition, image_path, width, height, str(exc))
