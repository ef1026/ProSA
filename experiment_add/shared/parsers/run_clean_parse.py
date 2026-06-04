from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiment_add.shared.data.load_manifest import load_manifest
from experiment_add.shared.parsers.mineru_parser_add import create_mineru_parser, parse_mineru_page
from experiment_add.shared.parsers.parser_worker_utils import (
    describe_parser_environment,
    get_paddle_python_from_config,
    load_parser_config,
    setup_ppocr_python_env,
    validate_paddle_python_path,
)
from experiment_add.shared.parsers.ppstructure_parser_add import create_ppstructure_parser, parse_ppstructure_page
from experiment_add.shared.parsers.normalize_parser_output import make_failed_record
from experiment_add.shared.utils.io import append_jsonl, atomic_write_json, atomic_write_jsonl, ensure_dir
from experiment_add.shared.utils.logging import get_logger, log_stage_start, log_stage_summary
from experiment_add.shared.utils.path_manager import PathManager
from experiment_add.shared.utils.retry import retry_call


VALID_PIPELINES = {"mineru", "ppstructure"}


def _base_config_path(parser_config_path: Path) -> Path:
    """Infer base.yaml next to parser.yaml."""
    return parser_config_path.parent / "base.yaml"


def _safe_page_filename(page_id: str) -> str:
    """Convert a page_id to a safe JSON filename."""
    return page_id.replace("/", "_").replace("\\", "_") + ".json"


def _load_existing_record(path: Path) -> dict[str, Any] | None:
    """Load an existing page JSON record if it is valid JSON."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_failed_log(path: Path, row: dict[str, Any]) -> None:
    """Append a failed parser row to CSV."""
    ensure_dir(path.parent)
    write_header = not path.exists() or path.stat().st_size == 0
    fieldnames = ["timestamp", "pipeline", "mode", "page_id", "image_path", "parser_status", "error_message"]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def _make_summary(
    pipeline: str,
    mode: str,
    input_pages: int,
    records: list[dict[str, Any]],
    skipped_pages: int,
    output_pages_dir: Path,
    merged_jsonl_path: Path,
    failed_log_path: Path,
) -> dict[str, Any]:
    """Build a clean parse summary dictionary."""
    statuses = [record.get("parser_status") for record in records]
    success_pages = statuses.count("success")
    failed_pages = statuses.count("failed")
    empty_pages = statuses.count("empty")
    parsed_records = [record for record in records if record.get("parser_status") in {"success", "empty"}]
    avg_blocks = (
        sum(len(record.get("blocks", [])) for record in parsed_records) / len(parsed_records)
        if parsed_records
        else 0.0
    )
    avg_text_len = (
        sum(len(record.get("page_text", "")) for record in parsed_records) / len(parsed_records)
        if parsed_records
        else 0.0
    )
    return {
        "pipeline": pipeline,
        "mode": mode,
        "input_pages": input_pages,
        "success_pages": success_pages,
        "failed_pages": failed_pages,
        "empty_pages": empty_pages,
        "skipped_pages": skipped_pages,
        "average_blocks": avg_blocks,
        "average_page_text_length": avg_text_len,
        "output_pages_dir": str(output_pages_dir),
        "merged_jsonl_path": str(merged_jsonl_path),
        "failed_log_path": str(failed_log_path),
    }


def _write_summary(path: Path, summary: dict[str, Any], notes: list[str]) -> None:
    """Write a markdown clean parse summary."""
    lines = [
        f"# Clean Parse Summary: {summary['pipeline']} {summary['mode']}",
        "",
        f"- pipeline: `{summary['pipeline']}`",
        f"- mode: `{summary['mode']}`",
        f"- input_pages: `{summary['input_pages']}`",
        f"- success_pages: `{summary['success_pages']}`",
        f"- failed_pages: `{summary['failed_pages']}`",
        f"- empty_pages: `{summary['empty_pages']}`",
        f"- skipped_pages: `{summary['skipped_pages']}`",
        f"- average_blocks: `{summary['average_blocks']:.3f}`",
        f"- average_page_text_length: `{summary['average_page_text_length']:.3f}`",
        f"- output_pages_dir: `{summary['output_pages_dir']}`",
        f"- merged_jsonl_path: `{summary['merged_jsonl_path']}`",
        f"- failed_log_path: `{summary['failed_log_path']}`",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_parser_once(pipeline: str, parser_config: dict[str, Any], logger: Any):
    """Create a parser once for a clean parse run; return None if init fails."""
    try:
        if pipeline == "mineru":
            return create_mineru_parser(parser_config)
        setup_ppocr_python_env(parser_config, logger=logger)
        describe_parser_environment(logger=logger)
        return create_ppstructure_parser(parser_config, logger=logger)
    except Exception as exc:
        logger.exception("Failed to initialize %s parser: %s", pipeline, exc)
        return None


def run_clean_parse(
    pipeline: str,
    parser_config_path: str | Path,
    debug: bool = False,
    skip_existing_any_status: bool = False,
    max_retries: int = 1,
) -> dict[str, Any]:
    """Run clean parsing for debug20 or full manifest."""
    if pipeline not in VALID_PIPELINES:
        raise ValueError(f"Unsupported pipeline: {pipeline}")

    parser_config_path = Path(parser_config_path)
    base_config_path = _base_config_path(parser_config_path)
    pm = PathManager(base_config_path, create_dirs=True)
    parser_config = load_parser_config(parser_config_path)
    mode = "debug" if debug else "full"
    manifest_path = pm.page_manifest_debug20 if debug else pm.page_manifest_500
    records = load_manifest(manifest_path)

    output_pages_dir = pm.clean_parse_pages_dir(pipeline)
    ensure_dir(output_pages_dir)
    merged_jsonl_path = pm.clean_parse_merged_path(pipeline)
    failed_log_path = pm.shared_log_root / "failed_parser_outputs.csv"
    summary_path = pm.shared_log_root / f"clean_parse_{pipeline}_{mode}_summary.md"
    logger = get_logger(f"experiment_add.clean_parse.{pipeline}", pm.shared_log_root / f"clean_parse_{pipeline}_{mode}.log")
    start = log_stage_start(logger, f"clean_parse:{pipeline}:{mode}")
    notes = [
        "No DeepSeek, perturbation, QA generation, QA answering, or evaluation code was called.",
        f"Manifest: {manifest_path}",
    ]
    if pipeline == "ppstructure":
        configured_paddle = get_paddle_python_from_config(parser_config)
        import os

        existing_paddle = os.environ.get(parser_config.get("ppstructure", {}).get("env_var_name", "PPOCR_PYTHON"), "")
        if not validate_paddle_python_path(configured_paddle) and not validate_paddle_python_path(existing_paddle):
            notes.append(
                "PPStructure worker environment issue: no valid paddle_python_path or PPOCR_PYTHON was found. "
                "Set parser.yaml ppstructure.paddle_python_path or export PPOCR_PYTHON to a Paddle/PaddleOCR Python executable."
            )

    parser_obj = _create_parser_once(pipeline, parser_config, logger)
    if parser_obj is None:
        notes.append(f"{pipeline} parser initialization failed; per-page failed JSON records were written.")

    output_records: list[dict[str, Any]] = []
    skipped_pages = 0
    for record in records:
        page_id = record["page_id"]
        page_json_path = output_pages_dir / _safe_page_filename(page_id)
        existing = _load_existing_record(page_json_path)
        if existing is not None:
            existing_status = existing.get("parser_status")
            if existing_status == "success" or skip_existing_any_status:
                output_records.append(existing)
                skipped_pages += 1
                continue

        def _parse_one():
            if parser_obj is None:
                return make_failed_record(
                    page_id,
                    pipeline,
                    "clean",
                    record.get("image_path", ""),
                    record.get("width", 0),
                    record.get("height", 0),
                    f"{pipeline} parser initialization failed",
                )
            if pipeline == "mineru":
                result = parse_mineru_page(record, parser_config, condition="clean", parser=parser_obj, project_root=pm.project_root)
            else:
                result = parse_ppstructure_page(
                    record,
                    parser_config,
                    condition="clean",
                    parser=parser_obj,
                    project_root=pm.project_root,
                    logger=logger,
                )
            if result.get("parser_status") == "failed":
                raise RuntimeError(result.get("error_message") or "parser failed")
            return result

        try:
            parsed = retry_call(_parse_one, max_retries=max_retries, sleep_seconds=0.5, backoff=True, logger=logger)
        except Exception as exc:
            # Wrappers normally return failed records, but keep this guard so one
            # unexpected exception never aborts the batch.
            parsed = make_failed_record(
                page_id,
                pipeline,
                "clean",
                record.get("image_path", ""),
                record.get("width", 0),
                record.get("height", 0),
                str(exc),
            )

        atomic_write_json(page_json_path, parsed)
        output_records.append(parsed)
        if parsed.get("parser_status") == "failed":
            _write_failed_log(
                failed_log_path,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "pipeline": pipeline,
                    "mode": mode,
                    "page_id": parsed.get("page_id"),
                    "image_path": parsed.get("image_path"),
                    "parser_status": parsed.get("parser_status"),
                    "error_message": parsed.get("error_message"),
                },
            )
        append_jsonl(pm.shared_log_root / f"clean_parse_{pipeline}_{mode}_events.jsonl", parsed)

    atomic_write_jsonl(merged_jsonl_path, output_records)
    summary = _make_summary(
        pipeline,
        mode,
        input_pages=len(records),
        records=output_records,
        skipped_pages=skipped_pages,
        output_pages_dir=output_pages_dir,
        merged_jsonl_path=merged_jsonl_path,
        failed_log_path=failed_log_path,
    )
    _write_summary(summary_path, summary, notes)
    end = time.time()
    log_stage_summary(
        logger,
        {
            "stage_name": f"clean_parse:{pipeline}:{mode}",
            "start_time": start,
            "end_time": end,
            "elapsed_seconds": end - start,
            "input_count": len(records),
            "success_count": summary["success_pages"],
            "failed_count": summary["failed_pages"],
            "skipped_count": skipped_pages,
            "output_paths": [str(output_pages_dir), str(merged_jsonl_path), str(summary_path)],
            "notes": notes,
        },
    )
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clean page parsing through experiment_add wrappers.")
    parser.add_argument("--pipeline", required=True, choices=sorted(VALID_PIPELINES))
    parser.add_argument("--config", default="experiment_add/configs/parser.yaml", help="Path to parser.yaml")
    parser.add_argument("--debug", action="store_true", help="Use page_manifest_debug20.csv")
    parser.add_argument("--skip-existing-any-status", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()

    summary = run_clean_parse(
        pipeline=args.pipeline,
        parser_config_path=args.config,
        debug=args.debug,
        skip_existing_any_status=args.skip_existing_any_status,
        max_retries=args.max_retries,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
