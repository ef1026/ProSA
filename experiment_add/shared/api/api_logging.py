from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from experiment_add.shared.utils.io import append_jsonl


def log_api_event(log_path: str | Path, event: dict[str, Any]) -> None:
    """Append a redacted API event to JSONL."""
    safe_event = dict(event)
    safe_event.pop("api_key", None)
    safe_event.pop("authorization", None)
    safe_event.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    append_jsonl(log_path, safe_event)
