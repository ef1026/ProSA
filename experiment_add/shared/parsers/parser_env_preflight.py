"""AutoDL parser-environment preflight for full500 runs.

Checks (read-only, no parser invocation):

* MinerU parser is importable in the main (advdoc) Python.
* PPStructure parser glue is importable in the main Python.
* ``$PPOCR_PYTHON`` (or the path configured in ``parser.yaml``) points at an
  existing executable.
* That executable can ``import paddle`` and ``import paddleocr`` (verified
  via subprocess so we don't pollute the calling Python's import space).
* Useful environment variables (``HF_ENDPOINT``, ``HF_HOME``, ``PPOCR_PYTHON``)
  are reported.

Exit code is ``0`` when all hard checks pass; with ``--strict`` any failure
causes exit code ``1``. The script never imports ``paddle`` directly in the
calling Python and never invokes a parser on a real page, so it has no side
effects beyond writing a small JSON summary into
``experiment_add/logs/shared/parser_env_preflight.json`` for auditability.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiment_add.shared.parsers.parser_worker_utils import (
    get_paddle_python_from_config,
    load_parser_config,
    validate_paddle_python_path,
)
from experiment_add.shared.utils.io import atomic_write_json, ensure_dir


def _try_import(module_name: str, attr: str | None = None) -> tuple[bool, str]:
    """Return ``(ok, error_message)`` for an attempted import."""
    try:
        module = importlib.import_module(module_name)
        if attr is not None:
            getattr(module, attr)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _verify_paddle_subprocess(paddle_python: str, timeout_seconds: int = 30) -> dict[str, Any]:
    """Run ``import paddle; import paddleocr`` inside the worker Python."""
    result: dict[str, Any] = {
        "paddle_python": paddle_python,
        "paddle_import_ok": False,
        "paddle_version": None,
        "paddleocr_import_ok": False,
        "paddleocr_version": None,
        "stdout": "",
        "stderr": "",
        "returncode": None,
    }
    if not paddle_python or not validate_paddle_python_path(paddle_python):
        result["stderr"] = f"paddle python path is invalid or missing: {paddle_python!r}"
        return result
    code = (
        "import json, sys\n"
        "out = {}\n"
        "try:\n"
        "    import paddle\n"
        "    out['paddle_ok'] = True\n"
        "    out['paddle_version'] = getattr(paddle, '__version__', None)\n"
        "except Exception as exc:\n"
        "    out['paddle_ok'] = False\n"
        "    out['paddle_error'] = f'{type(exc).__name__}: {exc}'\n"
        "try:\n"
        "    import paddleocr\n"
        "    out['paddleocr_ok'] = True\n"
        "    out['paddleocr_version'] = getattr(paddleocr, '__version__', None)\n"
        "except Exception as exc:\n"
        "    out['paddleocr_ok'] = False\n"
        "    out['paddleocr_error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(out))\n"
    )
    try:
        proc = subprocess.run(
            [paddle_python, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        result["stderr"] = f"timeout after {timeout_seconds}s: {exc}"
        return result
    except Exception as exc:
        result["stderr"] = f"{type(exc).__name__}: {exc}"
        return result
    result["stdout"] = proc.stdout
    result["stderr"] = proc.stderr
    result["returncode"] = proc.returncode
    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if proc.returncode == 0 and last_line:
        try:
            payload = json.loads(last_line)
            result["paddle_import_ok"] = bool(payload.get("paddle_ok"))
            result["paddle_version"] = payload.get("paddle_version")
            result["paddleocr_import_ok"] = bool(payload.get("paddleocr_ok"))
            result["paddleocr_version"] = payload.get("paddleocr_version")
            if not result["paddle_import_ok"]:
                result["paddle_error"] = payload.get("paddle_error")
            if not result["paddleocr_import_ok"]:
                result["paddleocr_error"] = payload.get("paddleocr_error")
        except json.JSONDecodeError as exc:
            result["paddle_error"] = f"could not parse worker JSON: {exc}; output={last_line!r}"
    return result


def run_preflight(parser_config_path: str | Path, strict: bool = False) -> dict[str, Any]:
    """Run all preflight checks and return a structured summary."""
    parser_config_path = Path(parser_config_path).resolve()
    parser_config = load_parser_config(parser_config_path)

    mineru_ok, mineru_err = _try_import("parsers.mineru_parser", "MinerUParser")
    ppstr_ok, ppstr_err = _try_import("parsers.ppstructure_parser", "PPStructureParser")
    glue_mineru_ok, glue_mineru_err = _try_import(
        "experiment_add.shared.parsers.mineru_parser_add", "create_mineru_parser"
    )
    glue_ppstr_ok, glue_ppstr_err = _try_import(
        "experiment_add.shared.parsers.ppstructure_parser_add", "create_ppstructure_parser"
    )

    pp_cfg = parser_config.get("ppstructure", {})
    env_var_name = pp_cfg.get("env_var_name", "PPOCR_PYTHON")
    configured_paddle = get_paddle_python_from_config(parser_config)
    env_paddle = os.environ.get(env_var_name, "")
    selected_paddle = configured_paddle if validate_paddle_python_path(configured_paddle) else env_paddle
    paddle_python_exists = validate_paddle_python_path(selected_paddle)

    paddle_subprocess = _verify_paddle_subprocess(selected_paddle) if paddle_python_exists else {
        "paddle_python": selected_paddle,
        "paddle_import_ok": False,
        "paddleocr_import_ok": False,
        "stderr": "no valid PPOCR_PYTHON / paddle_python_path",
    }

    env_vars = {
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
        "HF_HOME": os.environ.get("HF_HOME", ""),
        env_var_name: env_paddle,
    }

    hard_checks = {
        "mineru_parser_import_ok": mineru_ok,
        "ppstructure_parser_import_ok": ppstr_ok,
        "experiment_add_mineru_glue_ok": glue_mineru_ok,
        "experiment_add_ppstructure_glue_ok": glue_ppstr_ok,
        "ppocr_python_exists": paddle_python_exists,
        "paddle_import_ok": paddle_subprocess.get("paddle_import_ok", False),
        "paddleocr_import_ok": paddle_subprocess.get("paddleocr_import_ok", False),
    }
    ready = all(hard_checks.values())

    failures: list[str] = []
    if not mineru_ok:
        failures.append(f"mineru_parser import failed: {mineru_err}")
    if not ppstr_ok:
        failures.append(f"ppstructure_parser import failed: {ppstr_err}")
    if not glue_mineru_ok:
        failures.append(f"experiment_add mineru glue import failed: {glue_mineru_err}")
    if not glue_ppstr_ok:
        failures.append(f"experiment_add ppstructure glue import failed: {glue_ppstr_err}")
    if not paddle_python_exists:
        failures.append(f"PPOCR_PYTHON / paddle_python_path is missing or does not exist: {selected_paddle!r}")
    if not paddle_subprocess.get("paddle_import_ok", False):
        failures.append(f"paddle import failed in worker python: {paddle_subprocess.get('paddle_error', paddle_subprocess.get('stderr', '?'))}")
    if not paddle_subprocess.get("paddleocr_import_ok", False):
        failures.append(f"paddleocr import failed in worker python: {paddle_subprocess.get('paddleocr_error', paddle_subprocess.get('stderr', '?'))}")

    project_root = parser_config_path.parents[2] if parser_config_path.parent.name == "configs" and parser_config_path.parent.parent.name == "experiment_add" else Path.cwd()
    log_dir = project_root / "experiment_add/logs/shared"
    ensure_dir(log_dir)
    summary = {
        "current_working_directory": str(Path.cwd()),
        "parser_config": str(parser_config_path),
        "main_python_executable": sys.executable,
        "main_python_version": sys.version.split()[0],
        "env_vars": env_vars,
        "configured_paddle_python_path": configured_paddle,
        "selected_paddle_python_path": selected_paddle,
        "ppocr_python_exists": paddle_python_exists,
        "paddle_subprocess": paddle_subprocess,
        "imports": {
            "parsers.mineru_parser.MinerUParser": {"ok": mineru_ok, "error": mineru_err},
            "parsers.ppstructure_parser.PPStructureParser": {"ok": ppstr_ok, "error": ppstr_err},
            "experiment_add.shared.parsers.mineru_parser_add.create_mineru_parser": {"ok": glue_mineru_ok, "error": glue_mineru_err},
            "experiment_add.shared.parsers.ppstructure_parser_add.create_ppstructure_parser": {"ok": glue_ppstr_ok, "error": glue_ppstr_err},
        },
        "hard_checks": hard_checks,
        "failures": failures,
        "ready_for_autodl_clean_parse": "YES" if ready else "NO",
        "strict_mode": strict,
    }
    atomic_write_json(log_dir / "parser_env_preflight.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoDL parser-environment preflight (read-only).")
    parser.add_argument("--config", default="experiment_add/configs/parser.yaml")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any hard-check failure.")
    args = parser.parse_args()
    summary = run_preflight(args.config, strict=args.strict)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.strict and summary["ready_for_autodl_clean_parse"] != "YES":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
