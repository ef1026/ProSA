from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from experiment_add.shared.utils.io import read_yaml


def load_parser_config(parser_config_path: str | Path) -> dict[str, Any]:
    """Load parser.yaml."""
    return read_yaml(parser_config_path)


def get_paddle_python_from_config(config: dict[str, Any]) -> str:
    """Return configured Paddle Python path, expanding simple env placeholders."""
    pp_cfg = config.get("ppstructure", {})
    raw = str(pp_cfg.get("paddle_python_path") or "").strip()
    if raw in ("${PPOCR_PYTHON:-}", "$PPOCR_PYTHON"):
        return os.environ.get(pp_cfg.get("env_var_name", "PPOCR_PYTHON"), "")
    return os.path.expandvars(os.path.expanduser(raw))


def validate_paddle_python_path(path: str | Path | None) -> bool:
    """Return True when a Paddle Python path is non-empty and exists."""
    if path is None or not str(path).strip():
        return False
    return Path(path).exists()


def setup_ppocr_python_env(config: dict[str, Any], logger: Any = None) -> str:
    """Set PPOCR_PYTHON from config when valid and return the selected value."""
    pp_cfg = config.get("ppstructure", {})
    env_name = pp_cfg.get("env_var_name", "PPOCR_PYTHON")
    configured = get_paddle_python_from_config(config)
    existing = os.environ.get(env_name, "")

    selected = ""
    if validate_paddle_python_path(configured):
        selected = configured
        os.environ[env_name] = configured
        if logger is not None:
            logger.info("Using %s from parser.yaml: %s", env_name, configured)
    elif existing:
        selected = existing
        if validate_paddle_python_path(existing):
            if logger is not None:
                logger.info("Using existing %s: %s", env_name, existing)
        elif logger is not None:
            logger.warning("Existing %s does not exist: %s", env_name, existing)
    elif logger is not None:
        logger.warning(
            "No valid PPStructure Paddle Python configured. Set %s or parser.yaml ppstructure.paddle_python_path.",
            env_name,
        )

    for key, value in pp_cfg.get("worker_environment", {}).items():
        os.environ.setdefault(str(key), str(value))
    return selected


def describe_parser_environment(logger: Any = None) -> dict[str, Any]:
    """Describe parser-relevant runtime environment without running parsers."""
    info = {
        "python_executable": sys.executable,
        "PPOCR_PYTHON": os.environ.get("PPOCR_PYTHON", ""),
        "cwd": str(Path.cwd()),
    }
    if logger is not None:
        logger.info("Parser environment: %s", info)
    return info
