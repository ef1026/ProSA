from __future__ import annotations

from pathlib import Path
from typing import Any

from experiment_add.shared.parsers.normalize_parser_output import make_failed_record, normalize_parse_result
from experiment_add.shared.parsers.parser_worker_utils import setup_ppocr_python_env


def create_ppstructure_parser(config: dict[str, Any] | None = None, logger: Any = None):
    """Create the original PPStructureParser after preparing worker env vars."""
    from parsers.ppstructure_parser import PPStructureParser

    config = config or {}
    setup_ppocr_python_env(config, logger=logger)
    cfg = config.get("ppstructure", config)
    return PPStructureParser(
        lang=cfg.get("lang", "en"),
        use_gpu=bool(cfg.get("use_gpu", True)),
        show_log=bool(cfg.get("show_log", False)),
    )


def parse_ppstructure_page(
    record: dict[str, Any],
    config: dict[str, Any],
    condition: str = "clean",
    parser: Any | None = None,
    project_root: str | Path | None = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Parse one manifest record with the original PPStructureParser."""
    page_id = str(record.get("page_id", ""))
    image_path = str(record.get("image_path", ""))
    width = record.get("width", 0)
    height = record.get("height", 0)
    root = Path(project_root) if project_root is not None else Path.cwd()
    resolved_image = Path(image_path)
    if not resolved_image.is_absolute():
        resolved_image = root / resolved_image

    try:
        active_parser = parser if parser is not None else create_ppstructure_parser(config, logger=logger)
        parse_result = active_parser.parse(str(resolved_image))
        return normalize_parse_result(parse_result, page_id, "ppstructure", condition, image_path, width, height)
    except Exception as exc:
        return make_failed_record(page_id, "ppstructure", condition, image_path, width, height, str(exc))
