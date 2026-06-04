from __future__ import annotations

import json
import importlib.util
import sysconfig
import time
from pathlib import Path
from typing import Any


_STDLIB_LOGGING_PATH = Path(sysconfig.get_path("stdlib")) / "logging" / "__init__.py"
_SPEC = importlib.util.spec_from_file_location("_experiment_add_stdlib_logging", _STDLIB_LOGGING_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load stdlib logging module from {_STDLIB_LOGGING_PATH}")
logging = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(logging)


def _has_handler(logger: Any, marker: str, target: str | None = None) -> bool:
    """Return True when a matching experiment_add handler already exists."""
    for handler in logger.handlers:
        if getattr(handler, "_experiment_add_marker", None) != marker:
            continue
        if target is None or getattr(handler, "_experiment_add_target", None) == target:
            return True
    return False


def get_logger(name: str, log_file: str | Path | None = None) -> Any:
    """Create a reusable logger with console and optional file output.

    Calling this function repeatedly with the same logger name does not add
    duplicate handlers. File handler parents are created automatically.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not _has_handler(logger, "console"):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler._experiment_add_marker = "console"  # type: ignore[attr-defined]
        logger.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(path.resolve())
        if not _has_handler(logger, "file", resolved):
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler._experiment_add_marker = "file"  # type: ignore[attr-defined]
            file_handler._experiment_add_target = resolved  # type: ignore[attr-defined]
            logger.addHandler(file_handler)

    return logger


def log_stage_start(logger: Any, stage_name: str) -> float:
    """Log the start of a stage and return the start timestamp."""
    start_time = time.time()
    logger.info("Stage started: %s", stage_name)
    return start_time


def log_stage_summary(logger: Any, summary_dict: dict[str, Any]) -> None:
    """Log a structured stage summary.

    The summary may include `stage_name`, `start_time`, `end_time`,
    `elapsed_seconds`, `input_count`, `success_count`, `failed_count`,
    `skipped_count`, `output_paths`, and `notes`. Missing fields are filled with
    neutral defaults where possible.
    """
    summary = dict(summary_dict)
    if summary.get("end_time") is None:
        summary["end_time"] = time.time()
    if summary.get("elapsed_seconds") is None and summary.get("start_time") is not None:
        summary["elapsed_seconds"] = float(summary["end_time"]) - float(summary["start_time"])

    for key in ("input_count", "success_count", "failed_count", "skipped_count"):
        summary.setdefault(key, 0)
    summary.setdefault("output_paths", [])
    summary.setdefault("notes", "")

    logger.info("Stage summary: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
