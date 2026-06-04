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
from experiment_add.shared.parsers.normalize_parser_output import make_failed_record
from experiment_add.shared.parsers.parser_worker_utils import (
    describe_parser_environment,
    get_paddle_python_from_config,
    load_parser_config,
    setup_ppocr_python_env,
    validate_paddle_python_path,
)
from experiment_add.shared.parsers.ppstructure_parser_add import create_ppstructure_parser, parse_ppstructure_page
from experiment_add.shared.utils.io import append_jsonl, atomic_write_json, atomic_write_jsonl, ensure_dir
from experiment_add.shared.utils.logging import get_logger, log_stage_start, log_stage_summary
from experiment_add.shared.utils.path_manager import PathManager
from experiment_add.shared.utils.retry import retry_call


VALID_PIPELINES = {"mineru", "ppstructure"}
PERTURBED_CONDITIONS = ["area_matched_erasure", "structural_probe", "large_area_erasure"]


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


def _relative_or_absolute(path: Path, root: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _perturbed_image_record(row: dict[str, Any], condition: str, pm: PathManager) -> dict[str, Any]:
    """Create a manifest-like row pointing at a perturbed image."""
    page_id = str(row["page_id"])
    image_path = pm.perturbed_images_dir(condition) / f"{Path(page_id).stem}.png"
    record = dict(row)
    record["image_path"] = _relative_or_absolute(image_path, pm.project_root)
    return record


def _write_failed_log(path: Path, row: dict[str, Any]) -> None:
    """Append a perturbed parser failure row to CSV."""
    ensure_dir(path.parent)
    write_header = not path.exists() or path.stat().st_size == 0
    fieldnames = [
        "timestamp",
        "pipeline",
        "mode",
        "condition",
        "page_id",
        "image_path",
        "parser_status",
        "error_message",
    ]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def _create_parser_once(pipeline: str, parser_config: dict[str, Any], logger: Any):
    """Create a parser once for a perturbed parse run; return None if init fails."""
    try:
        if pipeline == "mineru":
            return create_mineru_parser(parser_config)
        setup_ppocr_python_env(parser_config, logger=logger)
        describe_parser_environment(logger=logger)
        return create_ppstructure_parser(parser_config, logger=logger)
    except Exception as exc:
        logger.exception("Failed to initialize %s parser: %s", pipeline, exc)
        return None


def _empty_condition_summary() -> dict[str, Any]:
    return {
        "input_pages": 0,
        "success_pages": 0,
        "empty_pages": 0,
        "failed_pages": 0,
        "skipped_pages": 0,
        "average_blocks": 0.0,
        "average_page_text_length": 0.0,
        "merged_jsonl_path": "",
    }


def _summarize_condition(
    records: list[dict[str, Any]],
    input_pages: int,
    skipped_pages: int,
    merged_jsonl_path: Path,
) -> dict[str, Any]:
    statuses = [record.get("parser_status") for record in records]
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
        "input_pages": input_pages,
        "success_pages": statuses.count("success"),
        "empty_pages": statuses.count("empty"),
        "failed_pages": statuses.count("failed"),
        "skipped_pages": skipped_pages,
        "average_blocks": avg_blocks,
        "average_page_text_length": avg_text_len,
        "merged_jsonl_path": str(merged_jsonl_path),
    }


def _write_summary(path: Path, summary: dict[str, Any], notes: list[str]) -> None:
    """Write a markdown summary for perturbed parsing."""
    conditions = summary["conditions"]
    by_cond = summary["by_condition"]
    lines = [
        f"# Perturbed Parse Summary: {summary['pipeline']} {summary['mode']}",
        "",
        f"- pipeline: `{summary['pipeline']}`",
        f"- mode: `{summary['mode']}`",
        f"- conditions: `{', '.join(conditions)}`",
        f"- input_pages_per_condition: `{summary['input_pages_per_condition']}`",
        f"- failed_log_path: `{summary['failed_log_path']}`",
        "",
        "## Condition Summary",
        "",
    ]
    for condition in conditions:
        item = by_cond.get(condition, _empty_condition_summary())
        lines.extend(
            [
                f"### {condition}",
                "",
                f"- success_pages: `{item['success_pages']}`",
                f"- empty_pages: `{item['empty_pages']}`",
                f"- failed_pages: `{item['failed_pages']}`",
                f"- skipped_pages: `{item['skipped_pages']}`",
                f"- average_blocks: `{item['average_blocks']:.3f}`",
                f"- average_page_text_length: `{item['average_page_text_length']:.3f}`",
                f"- merged_jsonl_path: `{item['merged_jsonl_path']}`",
                "",
            ]
        )
    lines.extend(["## Notes", ""])
    lines.extend(f"- {note}" for note in notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_one_page(
    pipeline: str,
    record: dict[str, Any],
    condition: str,
    parser_obj: Any,
    parser_config: dict[str, Any],
    pm: PathManager,
    logger: Any,
) -> dict[str, Any]:
    page_id = str(record.get("page_id", ""))
    image_path = pm.project_root / str(record.get("image_path", ""))
    if not image_path.exists():
        return make_failed_record(
            page_id,
            pipeline,
            condition,
            record.get("image_path", ""),
            record.get("width", 0),
            record.get("height", 0),
            f"perturbed image not found: {image_path}",
        )
    if parser_obj is None:
        return make_failed_record(
            page_id,
            pipeline,
            condition,
            record.get("image_path", ""),
            record.get("width", 0),
            record.get("height", 0),
            f"{pipeline} parser initialization failed",
        )
    if pipeline == "mineru":
        return parse_mineru_page(record, parser_config, condition=condition, parser=parser_obj, project_root=pm.project_root)
    return parse_ppstructure_page(
        record,
        parser_config,
        condition=condition,
        parser=parser_obj,
        project_root=pm.project_root,
        logger=logger,
    )


def run_perturbed_parse(
    pipeline: str,
    parser_config_path: str | Path,
    debug: bool = False,
    conditions: list[str] | None = None,
    skip_existing_any_status: bool = False,
    max_retries: int = 1,
) -> dict[str, Any]:
    """Run perturbed parsing for debug20 or full manifest."""
    if pipeline not in VALID_PIPELINES:
        raise ValueError(f"Unsupported pipeline: {pipeline}")

    selected_conditions = conditions or list(PERTURBED_CONDITIONS)
    unknown = sorted(set(selected_conditions) - set(PERTURBED_CONDITIONS))
    if unknown:
        raise ValueError(f"Unsupported condition(s): {unknown}")

    parser_config_path = Path(parser_config_path)
    base_config_path = _base_config_path(parser_config_path)
    pm = PathManager(base_config_path, create_dirs=True)
    parser_config = load_parser_config(parser_config_path)
    mode = "debug" if debug else "full"
    manifest_path = pm.page_manifest_debug20 if debug else pm.page_manifest_500
    manifest_records = load_manifest(manifest_path)

    failed_log_path = pm.shared_log_root / "failed_parser_outputs.csv"
    summary_path = pm.shared_log_root / f"perturbed_parse_{pipeline}_{mode}_summary.md"
    logger = get_logger(
        f"experiment_add.perturbed_parse.{pipeline}",
        pm.shared_log_root / f"perturbed_parse_{pipeline}_{mode}.log",
    )
    start = log_stage_start(logger, f"perturbed_parse:{pipeline}:{mode}")
    notes = [
        "No DeepSeek, QA answering, QA evaluation, or full500 code was called.",
        f"Manifest: {manifest_path}",
        "Clean parser wrappers were reused for all page parses.",
    ]
    if pipeline == "ppstructure":
        configured_paddle = get_paddle_python_from_config(parser_config)
        import os

        env_name = parser_config.get("ppstructure", {}).get("env_var_name", "PPOCR_PYTHON")
        existing_paddle = os.environ.get(env_name, "")
        if not validate_paddle_python_path(configured_paddle) and not validate_paddle_python_path(existing_paddle):
            notes.append(
                "PPStructure worker environment issue: no valid paddle_python_path or PPOCR_PYTHON was found. "
                "Set parser.yaml ppstructure.paddle_python_path or export PPOCR_PYTHON to a Paddle/PaddleOCR Python executable."
            )

    parser_obj = _create_parser_once(pipeline, parser_config, logger)
    if parser_obj is None:
        notes.append(f"{pipeline} parser initialization failed; per-page failed JSON records were written.")

    by_condition: dict[str, dict[str, Any]] = {}
    total_success = 0
    total_failed = 0
    total_skipped = 0

    for condition in selected_conditions:
        output_pages_dir = pm.perturbed_parse_pages_dir(pipeline, condition)
        ensure_dir(output_pages_dir)
        merged_jsonl_path = pm.perturbed_parse_dir(pipeline, condition) / "merged.jsonl"
        output_records: list[dict[str, Any]] = []
        skipped_pages = 0

        for manifest_row in manifest_records:
            record = _perturbed_image_record(manifest_row, condition, pm)
            page_id = str(record["page_id"])
            page_json_path = output_pages_dir / _safe_page_filename(page_id)
            existing = _load_existing_record(page_json_path)
            if existing is not None:
                existing_status = existing.get("parser_status")
                if existing_status == "success" or skip_existing_any_status:
                    output_records.append(existing)
                    skipped_pages += 1
                    continue

            def _parse_attempt() -> dict[str, Any]:
                parsed_record = _parse_one_page(
                    pipeline,
                    record,
                    condition,
                    parser_obj,
                    parser_config,
                    pm,
                    logger,
                )
                if parsed_record.get("parser_status") == "failed":
                    raise RuntimeError(parsed_record.get("error_message") or "parser failed")
                return parsed_record

            try:
                parsed = retry_call(_parse_attempt, max_retries=max_retries, sleep_seconds=0.5, backoff=True, logger=logger)
            except Exception as exc:
                parsed = make_failed_record(
                    page_id,
                    pipeline,
                    condition,
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
                        "condition": condition,
                        "page_id": parsed.get("page_id"),
                        "image_path": parsed.get("image_path"),
                        "parser_status": parsed.get("parser_status"),
                        "error_message": parsed.get("error_message"),
                    },
                )
            append_jsonl(pm.shared_log_root / f"perturbed_parse_{pipeline}_{mode}_events.jsonl", parsed)

        atomic_write_jsonl(merged_jsonl_path, output_records)
        condition_summary = _summarize_condition(output_records, len(manifest_records), skipped_pages, merged_jsonl_path)
        by_condition[condition] = condition_summary
        total_success += condition_summary["success_pages"]
        total_failed += condition_summary["failed_pages"]
        total_skipped += skipped_pages

    summary = {
        "pipeline": pipeline,
        "mode": mode,
        "conditions": selected_conditions,
        "input_pages_per_condition": len(manifest_records),
        "by_condition": by_condition,
        "success_pages_by_condition": {c: by_condition[c]["success_pages"] for c in selected_conditions},
        "empty_pages_by_condition": {c: by_condition[c]["empty_pages"] for c in selected_conditions},
        "failed_pages_by_condition": {c: by_condition[c]["failed_pages"] for c in selected_conditions},
        "skipped_pages_by_condition": {c: by_condition[c]["skipped_pages"] for c in selected_conditions},
        "average_blocks_by_condition": {c: by_condition[c]["average_blocks"] for c in selected_conditions},
        "average_page_text_length_by_condition": {
            c: by_condition[c]["average_page_text_length"] for c in selected_conditions
        },
        "merged_jsonl_paths": {c: by_condition[c]["merged_jsonl_path"] for c in selected_conditions},
        "failed_log_path": str(failed_log_path),
        "summary_path": str(summary_path),
    }
    _write_summary(summary_path, summary, notes)
    end = time.time()
    log_stage_summary(
        logger,
        {
            "stage_name": f"perturbed_parse:{pipeline}:{mode}",
            "start_time": start,
            "end_time": end,
            "elapsed_seconds": end - start,
            "input_count": len(manifest_records) * len(selected_conditions),
            "success_count": total_success,
            "failed_count": total_failed,
            "skipped_count": total_skipped,
            "output_paths": [str(summary_path), *summary["merged_jsonl_paths"].values()],
            "notes": notes,
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run perturbed page parsing through experiment_add wrappers.")
    parser.add_argument("--pipeline", required=True, choices=sorted(VALID_PIPELINES))
    parser.add_argument("--config", default="experiment_add/configs/parser.yaml", help="Path to parser.yaml")
    parser.add_argument("--debug", action="store_true", help="Use page_manifest_debug20.csv")
    parser.add_argument("--condition", action="append", choices=PERTURBED_CONDITIONS, help="Condition to run; repeatable")
    parser.add_argument("--skip-existing-any-status", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()

    summary = run_perturbed_parse(
        pipeline=args.pipeline,
        parser_config_path=args.config,
        debug=args.debug,
        conditions=args.condition,
        skip_existing_any_status=args.skip_existing_any_status,
        max_retries=args.max_retries,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
