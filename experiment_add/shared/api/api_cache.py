from __future__ import annotations

from pathlib import Path
from typing import Any

from experiment_add.shared.utils.hash_utils import hash_json
from experiment_add.shared.utils.io import atomic_write_json, safe_read_json


def build_cache_key(model: str, temperature: float, prompt_hash: str, input_hash: str, task: str) -> str:
    """Build a stable cache key from required API cache fields."""
    return hash_json(
        {
            "model": model,
            "temperature": temperature,
            "prompt_hash": prompt_hash,
            "input_hash": input_hash,
            "task": task,
        }
    )


def cache_path(cache_dir: str | Path, cache_key: str) -> Path:
    """Return cache file path for a key."""
    return Path(cache_dir) / f"{cache_key}.json"


def read_cache(cache_dir: str | Path, cache_key: str) -> dict[str, Any] | None:
    """Read cached API response if it exists."""
    value = safe_read_json(cache_path(cache_dir, cache_key), default=None)
    return value if isinstance(value, dict) else None


def write_cache(cache_dir: str | Path, cache_key: str, payload: dict[str, Any]) -> None:
    """Write cached API response atomically."""
    atomic_write_json(cache_path(cache_dir, cache_key), payload)
