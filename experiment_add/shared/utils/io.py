from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml


PathLike = str | Path


def _to_path(path: PathLike) -> Path:
    """Convert a string or Path-like value to a Path."""
    return path if isinstance(path, Path) else Path(path)


def ensure_dir(path: PathLike) -> Path:
    """Create *path* as a directory if needed and return it as a Path."""
    target = _to_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    """Read a JSONL file.

    Missing files return an empty list. Existing files must contain one valid
    JSON object per non-empty line; JSON parse errors are intentionally not
    swallowed so callers can detect corrupted artifacts.
    """
    target = _to_path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def write_jsonl(path: PathLike, records: Iterable[dict[str, Any]]) -> None:
    """Write records to JSONL, one valid JSON object per line."""
    target = _to_path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: PathLike, record: dict[str, Any]) -> None:
    """Append one JSON-serializable record to a JSONL file."""
    target = _to_path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_yaml(path: PathLike) -> dict[str, Any]:
    """Read a YAML file and return an empty dict for an empty document."""
    target = _to_path(path)
    with target.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def safe_read_json(path: PathLike, default: Any = None) -> Any:
    """Read JSON, returning *default* when the file is missing or invalid."""
    target = _to_path(path)
    try:
        with target.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def safe_write_json(path: PathLike, obj: Any) -> bool:
    """Write JSON and return False instead of raising on failure."""
    try:
        atomic_write_json(path, obj)
        return True
    except Exception:
        return False


def read_text(path: PathLike) -> str:
    """Read a UTF-8 text file."""
    target = _to_path(path)
    return target.read_text(encoding="utf-8")


def write_text(path: PathLike, text: str) -> None:
    """Write UTF-8 text, creating the parent directory when needed."""
    target = _to_path(path)
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def atomic_write_jsonl(path: PathLike, records: Iterable[dict[str, Any]]) -> None:
    """Atomically write JSONL by writing a temp file then renaming it."""
    target = _to_path(path)
    ensure_dir(target.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: PathLike, obj: Any) -> None:
    """Atomically write a JSON file by writing a temp file then renaming it."""
    target = _to_path(path)
    ensure_dir(target.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2)
            f.write("\n")
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
