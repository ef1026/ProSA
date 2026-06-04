from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Any

from experiment_add.shared.utils.path_manager import PathManager


REQUIRED_MANIFEST_FIELDS = [
    "page_id",
    "dataset",
    "split",
    "complexity",
    "image_path",
    "width",
    "height",
    "n_orig_spans",
    "source_doc_id",
    "page_index",
]


def _project_root_from_manifest(path: Path) -> Path:
    """Infer project root from a manifest path under experiment_add/data."""
    resolved = path.resolve()
    for parent in resolved.parents:
        if parent.name == "experiment_add":
            return parent.parent
    return Path.cwd()


def _resolve_image_path(image_path: str, project_root: Path) -> Path:
    """Resolve a manifest image path relative to project root."""
    path = Path(image_path)
    return path if path.is_absolute() else project_root / path


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load a CSV manifest and return records as dictionaries."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        records = [dict(row) for row in reader]
    validate_manifest_records(records, project_root=_project_root_from_manifest(manifest_path))
    return records


def load_debug_or_full_manifest(config: str | Path | dict[str, Any], debug: bool = False) -> list[dict[str, str]]:
    """Load the debug or full manifest using `base.yaml` or a config dict."""
    if isinstance(config, (str, Path)):
        pm = PathManager(config)
        path = pm.page_manifest_debug20 if debug else pm.page_manifest_500
        return load_manifest(path)

    data_cfg = config.get("data", {})
    key = "page_manifest_debug20" if debug else "page_manifest_500"
    if key not in data_cfg:
        warnings.warn(f"Missing manifest path in config data.{key}", stacklevel=2)
        return []
    return load_manifest(data_cfg[key])


def validate_manifest_records(records: list[dict[str, Any]], project_root: str | Path | None = None) -> bool:
    """Validate required fields and image existence, warning on problems.

    The function does not call parsers. It returns True if all required fields
    are present and every non-empty image path exists.
    """
    ok = True
    root = Path(project_root) if project_root is not None else Path.cwd()

    for idx, record in enumerate(records):
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in record]
        if missing:
            warnings.warn(f"Manifest row {idx} is missing fields: {missing}", stacklevel=2)
            ok = False
        image_path = str(record.get("image_path", "")).strip()
        if not image_path:
            warnings.warn(f"Manifest row {idx} has empty image_path", stacklevel=2)
            ok = False
            continue
        if not _resolve_image_path(image_path, root).exists():
            warnings.warn(f"Manifest row {idx} image_path does not exist: {image_path}", stacklevel=2)
            ok = False

    return ok
