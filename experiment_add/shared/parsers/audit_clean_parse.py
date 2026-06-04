from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiment_add.shared.data.load_manifest import load_manifest
from experiment_add.shared.text.normalize_text import normalize_answer
from experiment_add.shared.utils.io import ensure_dir, write_text
from experiment_add.shared.utils.path_manager import PathManager


PIPELINES = ("mineru", "ppstructure")
REQUIRED_TOP_LEVEL = [
    "page_id",
    "pipeline",
    "condition",
    "image_path",
    "width",
    "height",
    "page_text",
    "blocks",
    "parser_status",
    "error_message",
    "raw_output_path",
]
REQUIRED_BLOCK_FIELDS = ["block_id", "layout_type", "bbox", "text", "reading_order"]
VALID_STATUSES = {"success", "empty", "failed"}


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read JSONL records and collect malformed line errors."""
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return records, [f"missing_jsonl:{path}"]
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    errors.append(f"line_{line_no}:not_object")
            except json.JSONDecodeError as exc:
                errors.append(f"line_{line_no}:json_error:{exc}")
    return records, errors


def _resolve_image_path(image_path: str, project_root: Path) -> Path:
    """Resolve manifest/output image paths relative to project root."""
    path = Path(image_path)
    return path if path.is_absolute() else project_root / path


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _valid_bbox(bbox: Any, width: int, height: int) -> tuple[bool, str]:
    """Validate bbox shape, numeric order, and rough image bounds."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, "bbox_not_xyxy"
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return False, "bbox_non_numeric"
    if x2 <= x1 or y2 <= y1:
        return False, "bbox_invalid_order"
    tolerance_x = max(10.0, width * 0.02)
    tolerance_y = max(10.0, height * 0.02)
    if x1 < -tolerance_x or y1 < -tolerance_y or x2 > width + tolerance_x or y2 > height + tolerance_y:
        return False, "bbox_out_of_image_range"
    return True, ""


def _text_related(page_text: str, blocks: list[dict[str, Any]]) -> bool:
    """Check whether page text and concatenated block text are roughly related."""
    block_text = " ".join(str(block.get("text", "")) for block in blocks)
    norm_page = normalize_answer(page_text)
    norm_blocks = normalize_answer(block_text)
    if not norm_page and not norm_blocks:
        return True
    if not norm_page or not norm_blocks:
        return False
    page_tokens = set(norm_page.split())
    block_tokens = set(norm_blocks.split())
    if not page_tokens or not block_tokens:
        return norm_page in norm_blocks or norm_blocks in norm_page
    overlap = len(page_tokens & block_tokens)
    return (overlap / max(1, min(len(page_tokens), len(block_tokens)))) >= 0.2


def _median(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _average(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _audit_pipeline(
    pipeline: str,
    path: Path,
    manifest_records: list[dict[str, str]],
    project_root: Path,
) -> dict[str, Any]:
    """Audit one clean parse merged JSONL."""
    records, json_errors = _read_jsonl(path)
    manifest_by_page = {record["page_id"]: record for record in manifest_records}
    manifest_ids = set(manifest_by_page)
    manifest_image_paths = {record["image_path"] for record in manifest_records}

    anomalies: list[dict[str, str]] = []
    page_ids = [str(record.get("page_id", "")) for record in records]
    duplicate_page_ids = [page_id for page_id, count in Counter(page_ids).items() if page_id and count > 1]
    for page_id in duplicate_page_ids:
        anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": "schema", "reason": "duplicate_page_id"})
    for error in json_errors:
        anomalies.append({"page_id": "unknown", "pipeline": pipeline, "status": "json", "reason": error})

    invalid_bbox_count = 0
    empty_block_text_count = 0
    pages_with_zero_blocks = 0
    pages_with_empty_text = 0
    block_counts: list[int] = []
    text_lengths: list[int] = []

    for record in records:
        page_id = str(record.get("page_id", ""))
        status = str(record.get("parser_status", ""))
        missing = [field for field in REQUIRED_TOP_LEVEL if field not in record]
        if missing:
            anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": f"missing_fields:{missing}"})
        if page_id not in manifest_ids:
            anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": "page_id_not_in_manifest"})
        if record.get("image_path") not in manifest_image_paths:
            anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": "image_path_not_in_manifest"})
        image_path = str(record.get("image_path", ""))
        if not _resolve_image_path(image_path, project_root).exists():
            anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": "image_path_missing"})
        if record.get("condition") != "clean":
            anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": "condition_not_clean"})
        if record.get("pipeline") != pipeline:
            anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": "pipeline_mismatch"})
        if status not in VALID_STATUSES:
            anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": "invalid_parser_status"})

        width = _as_int(record.get("width"))
        height = _as_int(record.get("height"))
        if width is None or width <= 0:
            anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": "invalid_width"})
            width = 0
        if height is None or height <= 0:
            anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": "invalid_height"})
            height = 0

        page_text = record.get("page_text")
        blocks = record.get("blocks")
        if not isinstance(page_text, str):
            anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": "page_text_not_string"})
            page_text = ""
        if not isinstance(blocks, list):
            anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": "blocks_not_list"})
            blocks = []

        block_counts.append(len(blocks))
        text_lengths.append(len(page_text))
        if len(blocks) == 0:
            pages_with_zero_blocks += 1
        if not page_text.strip():
            pages_with_empty_text += 1

        if status == "success":
            if not page_text.strip():
                anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": "success_empty_page_text"})
            if not blocks:
                anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": "success_zero_blocks"})
            seen_orders: set[int] = set()
            duplicate_order = False
            for idx, block in enumerate(blocks):
                if not isinstance(block, dict):
                    anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": f"block_{idx}_not_object"})
                    continue
                block_missing = [field for field in REQUIRED_BLOCK_FIELDS if field not in block]
                if block_missing:
                    anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": f"block_{idx}_missing:{block_missing}"})
                ok_bbox, reason = _valid_bbox(block.get("bbox"), width, height)
                if not ok_bbox:
                    invalid_bbox_count += 1
                    anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": f"block_{idx}_{reason}"})
                if not isinstance(block.get("text", ""), str):
                    anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": f"block_{idx}_text_not_string"})
                if isinstance(block.get("text", ""), str) and not block.get("text", "").strip():
                    empty_block_text_count += 1
                if not isinstance(block.get("reading_order"), int):
                    anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": f"block_{idx}_reading_order_not_int"})
                else:
                    order = int(block["reading_order"])
                    if order in seen_orders:
                        duplicate_order = True
                    seen_orders.add(order)
            if duplicate_order:
                anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": "duplicate_reading_order"})
            if not _text_related(page_text, blocks):
                anomalies.append({"page_id": page_id, "pipeline": pipeline, "status": status, "reason": "page_text_unrelated_to_blocks"})

    statuses = Counter(str(record.get("parser_status", "")) for record in records)
    schema_missing_count = sum(1 for a in anomalies if a["reason"].startswith("missing_fields:"))
    return {
        "schema_missing_count": schema_missing_count,
        "pipeline": pipeline,
        "path": str(path),
        "jsonl_exists": path.exists(),
        "json_errors": json_errors,
        "row_count": len(records),
        "expected_rows": len(manifest_records),
        "duplicate_page_ids": duplicate_page_ids,
        "success_pages": statuses.get("success", 0),
        "empty_pages": statuses.get("empty", 0),
        "failed_pages": statuses.get("failed", 0),
        "average_blocks": _average(block_counts),
        "median_blocks": _median(block_counts),
        "average_page_text_length": _average(text_lengths),
        "median_page_text_length": _median(text_lengths),
        "min_page_text_length": min(text_lengths) if text_lengths else 0,
        "max_page_text_length": max(text_lengths) if text_lengths else 0,
        "invalid_bbox_count": invalid_bbox_count,
        "empty_block_text_count": empty_block_text_count,
        "pages_with_zero_blocks": pages_with_zero_blocks,
        "pages_with_empty_text": pages_with_empty_text,
        "records_by_page": {str(record.get("page_id", "")): record for record in records},
        "anomalies": anomalies,
    }


def _ready_for_qa(mineru: dict[str, Any], ppstructure: dict[str, Any], both_success: int, both_non_empty_text: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if both_success < 15:
        reasons.append(f"both_success_pages={both_success} < 15")
    if both_non_empty_text < 15:
        reasons.append(f"both_non_empty_text_pages={both_non_empty_text} < 15")
    for audit in (mineru, ppstructure):
        if audit["invalid_bbox_count"] > max(5, audit["success_pages"]):
            reasons.append(f"{audit['pipeline']} invalid_bbox_count is high: {audit['invalid_bbox_count']}")
        if audit["average_page_text_length"] < 100:
            reasons.append(f"{audit['pipeline']} average_page_text_length is too short: {audit['average_page_text_length']:.1f}")
    return not reasons, reasons


def _write_report(path: Path, mineru: dict[str, Any], ppstructure: dict[str, Any], debug: bool, ready: bool, reasons: list[str], intersections: dict[str, Any]) -> None:
    """Write the clean parse audit markdown report."""
    lines = [
        "# Clean Parse Debug20 Audit Report" if debug else "# Clean Parse Full500 Audit Report",
        "",
        "## Required Summary",
        "",
        f"- mineru_success_pages: `{mineru['success_pages']}`",
        f"- ppstructure_success_pages: `{ppstructure['success_pages']}`",
        f"- both_success_pages: `{intersections['both_success_pages']}`",
        f"- both_non_empty_text_pages: `{intersections['both_non_empty_text_pages']}`",
        f"- mineru_empty_pages: `{mineru['empty_pages']}`",
        f"- ppstructure_empty_pages: `{ppstructure['empty_pages']}`",
        f"- mineru_failed_pages: `{mineru['failed_pages']}`",
        f"- ppstructure_failed_pages: `{ppstructure['failed_pages']}`",
        f"- average_blocks_mineru: `{mineru['average_blocks']:.3f}`",
        f"- average_blocks_ppstructure: `{ppstructure['average_blocks']:.3f}`",
        f"- median_blocks_mineru: `{mineru['median_blocks']:.3f}`",
        f"- median_blocks_ppstructure: `{ppstructure['median_blocks']:.3f}`",
        f"- average_text_length_mineru: `{mineru['average_page_text_length']:.3f}`",
        f"- average_text_length_ppstructure: `{ppstructure['average_page_text_length']:.3f}`",
        f"- median_text_length_mineru: `{mineru['median_page_text_length']:.3f}`",
        f"- median_text_length_ppstructure: `{ppstructure['median_page_text_length']:.3f}`",
        f"- invalid_bbox_count_mineru: `{mineru['invalid_bbox_count']}`",
        f"- invalid_bbox_count_ppstructure: `{ppstructure['invalid_bbox_count']}`",
        f"- pages_with_empty_text_mineru: `{mineru['pages_with_empty_text']}`",
        f"- pages_with_empty_text_ppstructure: `{ppstructure['pages_with_empty_text']}`",
        f"- schema_missing_count_mineru: `{mineru['schema_missing_count']}`",
        f"- schema_missing_count_ppstructure: `{ppstructure['schema_missing_count']}`",
        f"- ready_for_qa_generation: `{'YES' if ready else 'NO'}`",
        "",
        "## File-Level Checks",
        "",
    ]
    for audit in (mineru, ppstructure):
        lines.extend(
            [
                f"### {audit['pipeline']}",
                "",
                f"- merged_jsonl_exists: `{audit['jsonl_exists']}`",
                f"- row_count: `{audit['row_count']}`",
                f"- expected_manifest_rows: `{audit['expected_rows']}`",
                f"- json_error_count: `{len(audit['json_errors'])}`",
                f"- duplicate_page_id_count: `{len(audit['duplicate_page_ids'])}`",
                f"- min_page_text_length: `{audit['min_page_text_length']}`",
                f"- max_page_text_length: `{audit['max_page_text_length']}`",
                f"- empty_block_text_count: `{audit['empty_block_text_count']}`",
                f"- pages_with_zero_blocks: `{audit['pages_with_zero_blocks']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Dual Pipeline Intersections",
            "",
            f"- both_success_pages: `{intersections['both_success_pages']}`",
            f"- both_non_empty_text_pages: `{intersections['both_non_empty_text_pages']}`",
            f"- mineru_only_success_pages: `{intersections['mineru_only_success_pages']}`",
            f"- ppstructure_only_success_pages: `{intersections['ppstructure_only_success_pages']}`",
            f"- both_failed_pages: `{intersections['both_failed_pages']}`",
            "",
            "## Readiness Reasons",
            "",
        ]
    )
    if reasons:
        lines.extend(f"- {reason}" for reason in reasons)
    else:
        lines.append(
            "- QA generation readiness conditions are satisfied for debug20."
            if debug
            else "- QA generation readiness conditions are satisfied for full500."
        )
    lines.extend(["", "## Abnormal Pages", ""])
    anomalies = mineru["anomalies"] + ppstructure["anomalies"]
    if anomalies:
        lines.append("| page_id | pipeline | status | reason |")
        lines.append("| --- | --- | --- | --- |")
        for anomaly in anomalies:
            lines.append(
                f"| `{anomaly['page_id']}` | `{anomaly['pipeline']}` | `{anomaly['status']}` | `{anomaly['reason']}` |"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "No DeepSeek, QA generation, perturbation, full500 parsing, QA answering, or evaluation code was called.",
            "",
            f"Ready for Prompt 11C: {'YES' if ready else 'NO'}" if not debug else f"Ready for Prompt 5: {'YES' if ready else 'NO'}",
        ]
    )
    ensure_dir(path.parent)
    write_text(path, "\n".join(lines) + "\n")


def audit_clean_parse(config_path: str | Path, debug: bool = False) -> dict[str, Any]:
    """Audit clean parse outputs for MinerU and PPStructure."""
    pm = PathManager(config_path, create_dirs=True)
    manifest_path = pm.page_manifest_debug20 if debug else pm.page_manifest_500
    manifest_records = load_manifest(manifest_path)

    mineru = _audit_pipeline("mineru", pm.clean_parse_merged_path("mineru"), manifest_records, pm.project_root)
    ppstructure = _audit_pipeline("ppstructure", pm.clean_parse_merged_path("ppstructure"), manifest_records, pm.project_root)

    mineru_records = mineru["records_by_page"]
    pp_records = ppstructure["records_by_page"]
    all_pages = set(mineru_records) | set(pp_records)
    both_success = {
        page_id
        for page_id in all_pages
        if mineru_records.get(page_id, {}).get("parser_status") == "success"
        and pp_records.get(page_id, {}).get("parser_status") == "success"
    }
    both_non_empty_text = {
        page_id
        for page_id in both_success
        if mineru_records[page_id].get("page_text", "").strip()
        and pp_records[page_id].get("page_text", "").strip()
    }
    mineru_success = {page_id for page_id, record in mineru_records.items() if record.get("parser_status") == "success"}
    pp_success = {page_id for page_id, record in pp_records.items() if record.get("parser_status") == "success"}
    both_failed = {
        page_id
        for page_id in all_pages
        if mineru_records.get(page_id, {}).get("parser_status") == "failed"
        and pp_records.get(page_id, {}).get("parser_status") == "failed"
    }
    intersections = {
        "both_success_pages": len(both_success),
        "both_non_empty_text_pages": len(both_non_empty_text),
        "mineru_only_success_pages": len(mineru_success - pp_success),
        "ppstructure_only_success_pages": len(pp_success - mineru_success),
        "both_failed_pages": len(both_failed),
    }
    ready, reasons = _ready_for_qa(mineru, ppstructure, len(both_success), len(both_non_empty_text))
    report_path = pm.shared_log_root / ("clean_parse_debug20_audit_report.md" if debug else "clean_parse_full500_audit_report.md")
    _write_report(report_path, mineru, ppstructure, debug, ready, reasons, intersections)
    return {
        "report_path": str(report_path),
        "ready_for_qa_generation": ready,
        "mineru": {key: mineru[key] for key in ("success_pages", "empty_pages", "failed_pages", "invalid_bbox_count")},
        "ppstructure": {key: ppstructure[key] for key in ("success_pages", "empty_pages", "failed_pages", "invalid_bbox_count")},
        **intersections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit clean parse outputs.")
    parser.add_argument("--config", default="experiment_add/configs/base.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    result = audit_clean_parse(args.config, debug=args.debug)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ready_for_qa_generation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
