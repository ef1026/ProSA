from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_text(text: str | None) -> str:
    """Return a SHA-256 hash for UTF-8 text; None is treated as empty text."""
    value = "" if text is None else str(text)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_json(obj: Any) -> str:
    """Return a stable SHA-256 hash for a JSON-serializable object."""
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 hash for a file, reading it in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
