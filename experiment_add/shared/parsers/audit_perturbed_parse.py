from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiment_add.shared.data.load_manifest import load_manifest
from experiment_add.shared.utils.io import ensure_dir, read_jsonl, write_text
from experiment_add.shared.utils.path_manager import PathManager


PIPELINES = ("mineru", "ppstructure")
CONDITIONS = ("area_matched_erasure", "structural_probe", "large_area_erasure")
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


def _read_jsonl_with_errors(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
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
            except json.JSONDecodeError as exc:
                errors.append(f"line_{line_no}:json_error:{exc}")
                continue
            if not isinstance(obj, dict):
                errors.append(f"line_{line_no}:not_object")
                continue
            records.append(obj)
    return records, errors


def _resolve_path(path: str | Path, project_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root / p


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _valid_bbox(bbox: Any, width: int, height: int) -> tuple[bool, str]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, "bbox_not_xyxy"
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return False, "bbox_non_numeric"
    if x2 <= x1 or y2 <= y1:
        return False, "bbox_invalid_order"
    tol_x = max(10.0, width * 0.02)
    tol_y = max(10.0, height * 0.02)
    if x1 < -tol_x or y1 < -tol_y or x2 > width + tol_x or y2 > height + tol_y:
        return False, "bbox_out_of_image_range"
    return True, ""


def _median(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _average(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _perturbed_merged_path(pm: PathManager, pipeline: str, condition: str) -> Path:
    return pm.perturbed_parse_dir(pipeline, condition) / "merged.jsonl"


def _metadata_path(pm: PathManager) -> Path:
    primary = pm.project_root / "experiment_add/outputs/shared/perturbed_pages/merged_perturb_metadata.jsonl"
    if primary.exists():
        return primary
    return pm.merged_perturb_metadata_path


def _expected_perturbed_image(pm: PathManager, page_id: str, condition: str) -> Path:
    return pm.perturbed_images_dir(condition) / f"{Path(page_id).stem}.png"


def _read_clean_reference(pm: PathManager, pipeline: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("page_id", "")): row for row in read_jsonl(pm.clean_parse_merged_path(pipeline))}


def _read_metadata(pm: PathManager) -> dict[tuple[str, str], dict[str, Any]]:
    path = _metadata_path(pm)
    if not path.exists():
        return {}
    return {(str(row.get("page_id", "")), str(row.get("condition", ""))): row for row in read_jsonl(path)}


def _read_qa_page_counts(pm: PathManager) -> Counter[str]:
    if not pm.qa_pairs_shared_path.exists():
        return Counter()
    return Counter(str(row.get("page_id", "")) for row in read_jsonl(pm.qa_pairs_shared_path))


def _audit_pipeline_condition(
    pipeline: str,
    condition: str,
    path: Path,
    manifest_ids: set[str],
    pm: PathManager,
    clean_reference: dict[str, dict[str, Any]],
    metadata: dict[tuple[str, str], dict[str, Any]],
    qa_page_counts: Counter[str],
) -> dict[str, Any]:
    records, json_errors = _read_jsonl_with_errors(path)
    anomalies: list[dict[str, str]] = []
    empty_page_details: list[dict[str, Any]] = []

    page_ids = [str(row.get("page_id", "")) for row in records]
    duplicate_page_ids = [page_id for page_id, count in Counter(page_ids).items() if page_id and count > 1]
    pair_counts = Counter((str(row.get("page_id", "")), str(row.get("pipeline", "")), str(row.get("condition", ""))) for row in records)
    duplicate_page_pipeline_condition = sum(count - 1 for count in pair_counts.values() if count > 1)

    for error in json_errors:
        anomalies.append({"page_id": "unknown", "reason": error})
    for page_id in duplicate_page_ids:
        anomalies.append({"page_id": page_id, "reason": "duplicate_page_id"})

    status_counts: Counter[str] = Counter()
    block_counts: list[int] = []
    text_lengths: list[int] = []
    clean_block_counts: list[int] = []
    clean_text_lengths: list[int] = []

    invalid_bbox_count = 0
    empty_block_text_count = 0
    pages_with_zero_blocks = 0
    pages_with_empty_text = 0
    schema_missing_count = 0
    invalid_schema_count = 0
    image_path_missing_count = 0
    image_path_mismatch_count = 0
    invalid_status_count = 0

    for record in records:
        page_id = str(record.get("page_id", ""))
        status = str(record.get("parser_status", ""))
        status_counts[status] += 1

        missing = [field for field in REQUIRED_TOP_LEVEL if field not in record]
        if missing:
            schema_missing_count += len(missing)
            anomalies.append({"page_id": page_id, "reason": f"missing_fields:{missing}"})

        if page_id not in manifest_ids:
            anomalies.append({"page_id": page_id, "reason": "page_id_not_in_debug_manifest"})
        if record.get("condition") != condition:
            anomalies.append({"page_id": page_id, "reason": "condition_mismatch"})
        if record.get("pipeline") != pipeline:
            anomalies.append({"page_id": page_id, "reason": "pipeline_mismatch"})
        if status not in VALID_STATUSES:
            invalid_status_count += 1
            anomalies.append({"page_id": page_id, "reason": "invalid_parser_status"})

        width = _as_int(record.get("width"))
        height = _as_int(record.get("height"))
        if width is None or width <= 0:
            invalid_schema_count += 1
            width = 0
            anomalies.append({"page_id": page_id, "reason": "invalid_width"})
        if height is None or height <= 0:
            invalid_schema_count += 1
            height = 0
            anomalies.append({"page_id": page_id, "reason": "invalid_height"})

        page_text = record.get("page_text")
        blocks = record.get("blocks")
        if not isinstance(page_text, str):
            invalid_schema_count += 1
            page_text = ""
            anomalies.append({"page_id": page_id, "reason": "page_text_not_string"})
        if not isinstance(blocks, list):
            invalid_schema_count += 1
            blocks = []
            anomalies.append({"page_id": page_id, "reason": "blocks_not_list"})

        image_path = str(record.get("image_path", ""))
        resolved_image = _resolve_path(image_path, pm.project_root)
        expected_image = _expected_perturbed_image(pm, page_id, condition)
        if not resolved_image.exists():
            image_path_missing_count += 1
            anomalies.append({"page_id": page_id, "reason": "image_path_missing"})
        if resolved_image.resolve() != expected_image.resolve():
            image_path_mismatch_count += 1
            anomalies.append({"page_id": page_id, "reason": "image_path_not_expected_perturbed_image"})

        block_counts.append(len(blocks))
        text_lengths.append(len(page_text))
        if len(blocks) == 0:
            pages_with_zero_blocks += 1
        if not page_text.strip():
            pages_with_empty_text += 1

        clean = clean_reference.get(page_id, {})
        clean_blocks = clean.get("blocks", [])
        clean_text = clean.get("page_text", "")
        clean_block_counts.append(len(clean_blocks) if isinstance(clean_blocks, list) else 0)
        clean_text_lengths.append(len(clean_text) if isinstance(clean_text, str) else 0)

        if status == "success":
            if not page_text.strip():
                anomalies.append({"page_id": page_id, "reason": "success_empty_page_text"})
            if not blocks:
                anomalies.append({"page_id": page_id, "reason": "success_zero_blocks"})
            seen_orders: set[int] = set()
            duplicate_order = False
            for idx, block in enumerate(blocks):
                if not isinstance(block, dict):
                    anomalies.append({"page_id": page_id, "reason": f"block_{idx}_not_object"})
                    continue
                block_missing = [field for field in REQUIRED_BLOCK_FIELDS if field not in block]
                if block_missing:
                    schema_missing_count += len(block_missing)
                    anomalies.append({"page_id": page_id, "reason": f"block_{idx}_missing:{block_missing}"})
                ok_bbox, reason = _valid_bbox(block.get("bbox"), width, height)
                if not ok_bbox:
                    invalid_bbox_count += 1
                    anomalies.append({"page_id": page_id, "reason": f"block_{idx}_{reason}"})
                if not isinstance(block.get("text", ""), str):
                    invalid_schema_count += 1
                    anomalies.append({"page_id": page_id, "reason": f"block_{idx}_text_not_string"})
                elif not block.get("text", "").strip():
                    empty_block_text_count += 1
                if not isinstance(block.get("reading_order"), int):
                    invalid_schema_count += 1
                    anomalies.append({"page_id": page_id, "reason": f"block_{idx}_reading_order_not_int"})
                else:
                    reading_order = int(block["reading_order"])
                    if reading_order in seen_orders:
                        duplicate_order = True
                    seen_orders.add(reading_order)
            if duplicate_order:
                anomalies.append({"page_id": page_id, "reason": "duplicate_reading_order"})

        if status == "empty":
            meta = metadata.get((page_id, condition), {})
            empty_page_details.append(
                {
                    "page_id": page_id,
                    "pipeline": pipeline,
                    "condition": condition,
                    "image_path": image_path,
                    "TOR": meta.get("TOR"),
                    "support_bbox": meta.get("support_bbox"),
                    "reason": record.get("error_message"),
                    "qa_pair_count": int(qa_page_counts.get(page_id, 0)),
                    "has_qa_pairs": bool(qa_page_counts.get(page_id, 0)),
                }
            )

    avg_text = _average(text_lengths)
    avg_blocks = _average(block_counts)
    avg_clean_text = _average(clean_text_lengths)
    avg_clean_blocks = _average(clean_block_counts)

    return {
        "pipeline": pipeline,
        "condition": condition,
        "path": str(path),
        "jsonl_exists": path.exists(),
        "json_errors": json_errors,
        "row_count": len(records),
        "expected_rows": len(manifest_ids),
        "unique_page_ids": len(set(page_ids)),
        "duplicate_page_ids": duplicate_page_ids,
        "duplicate_page_pipeline_condition_count": duplicate_page_pipeline_condition,
        "success_pages": status_counts.get("success", 0),
        "empty_pages": status_counts.get("empty", 0),
        "failed_pages": status_counts.get("failed", 0),
        "invalid_status_count": invalid_status_count,
        "average_blocks": avg_blocks,
        "median_blocks": _median(block_counts),
        "average_page_text_length": avg_text,
        "median_page_text_length": _median(text_lengths),
        "min_page_text_length": min(text_lengths) if text_lengths else 0,
        "max_page_text_length": max(text_lengths) if text_lengths else 0,
        "average_clean_text_length": avg_clean_text,
        "average_clean_block_count": avg_clean_blocks,
        "mean_text_length_ratio": _ratio(avg_text, avg_clean_text),
        "mean_block_count_ratio": _ratio(avg_blocks, avg_clean_blocks),
        "invalid_bbox_count": invalid_bbox_count,
        "empty_block_text_count": empty_block_text_count,
        "pages_with_zero_blocks": pages_with_zero_blocks,
        "pages_with_empty_text": pages_with_empty_text,
        "schema_missing_count": schema_missing_count,
        "invalid_schema_count": invalid_schema_count,
        "image_path_missing_count": image_path_missing_count,
        "image_path_mismatch_count": image_path_mismatch_count,
        "empty_page_details": empty_page_details,
        "anomalies": anomalies,
    }


def _readiness(audits: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for key, audit in audits.items():
        if audit["row_count"] != audit["expected_rows"]:
            reasons.append(f"{key}: row_count {audit['row_count']} != expected {audit['expected_rows']}")
        if audit["json_errors"]:
            reasons.append(f"{key}: json_errors={len(audit['json_errors'])}")
        if audit["schema_missing_count"] != 0:
            reasons.append(f"{key}: schema_missing_count={audit['schema_missing_count']}")
        if audit["invalid_schema_count"] != 0:
            reasons.append(f"{key}: invalid_schema_count={audit['invalid_schema_count']}")
        if audit["success_pages"] < 15:
            reasons.append(f"{key}: success_pages={audit['success_pages']} < 15")
        if audit["failed_pages"] > 2:
            reasons.append(f"{key}: failed_pages={audit['failed_pages']} > 2")
        if audit["empty_pages"] > 5:
            reasons.append(f"{key}: empty_pages={audit['empty_pages']} > 5")
        if audit["pages_with_empty_text"] >= audit["expected_rows"]:
            reasons.append(f"{key}: all pages have empty text")
        if audit["invalid_bbox_count"] > max(20, audit["success_pages"] * 5):
            reasons.append(f"{key}: invalid_bbox_count is high ({audit['invalid_bbox_count']})")
        if audit["image_path_missing_count"] != 0 or audit["image_path_mismatch_count"] != 0:
            reasons.append(
                f"{key}: image path issue missing={audit['image_path_missing_count']} mismatch={audit['image_path_mismatch_count']}"
            )
        if audit["duplicate_page_pipeline_condition_count"] != 0:
            reasons.append(f"{key}: duplicate page/pipeline/condition rows={audit['duplicate_page_pipeline_condition_count']}")
    return not reasons, reasons


def _slug(pipeline: str, condition: str) -> str:
    cond = {"area_matched_erasure": "area", "structural_probe": "structural", "large_area_erasure": "large"}[condition]
    return f"{pipeline}_{cond}"


def _write_report(path: Path, audits: dict[str, dict[str, Any]], ready: bool, reasons: list[str]) -> None:
    lines = [
        "# Perturbed Parse Debug20 Audit Report",
        "",
        "## Required Summary",
        "",
    ]
    for pipeline in PIPELINES:
        for condition in CONDITIONS:
            key = _slug(pipeline, condition)
            audit = audits[key]
            lines.extend(
                [
                    f"- {key}_success: `{audit['success_pages']}`",
                    f"- {key}_empty: `{audit['empty_pages']}`",
                    f"- {key}_failed: `{audit['failed_pages']}`",
                ]
            )
    lines.extend(["", "## Average Text Length By Pipeline Condition", ""])
    for key, audit in audits.items():
        lines.append(f"- {key}: `{audit['average_page_text_length']:.3f}`")
    lines.extend(["", "## Average Blocks By Pipeline Condition", ""])
    for key, audit in audits.items():
        lines.append(f"- {key}: `{audit['average_blocks']:.3f}`")
    lines.extend(["", "## Invalid BBox Count By Pipeline Condition", ""])
    for key, audit in audits.items():
        lines.append(f"- {key}: `{audit['invalid_bbox_count']}`")
    lines.extend(["", "## Text Length Ratio Vs Clean By Pipeline Condition", ""])
    for key, audit in audits.items():
        lines.append(f"- {key}: `{audit['mean_text_length_ratio']:.6f}`")
    lines.extend(["", "## Block Count Ratio Vs Clean By Pipeline Condition", ""])
    for key, audit in audits.items():
        lines.append(f"- {key}: `{audit['mean_block_count_ratio']:.6f}`")
    lines.extend(["", "## Condition-Level Detail", ""])
    for key, audit in audits.items():
        lines.extend(
            [
                f"### {key}",
                "",
                f"- merged_jsonl_exists: `{audit['jsonl_exists']}`",
                f"- row_count: `{audit['row_count']}`",
                f"- expected_rows: `{audit['expected_rows']}`",
                f"- unique_page_ids: `{audit['unique_page_ids']}`",
                f"- duplicate_page_pipeline_condition_count: `{audit['duplicate_page_pipeline_condition_count']}`",
                f"- median_blocks: `{audit['median_blocks']:.3f}`",
                f"- median_page_text_length: `{audit['median_page_text_length']:.3f}`",
                f"- min_page_text_length: `{audit['min_page_text_length']}`",
                f"- max_page_text_length: `{audit['max_page_text_length']}`",
                f"- empty_block_text_count: `{audit['empty_block_text_count']}`",
                f"- pages_with_zero_blocks: `{audit['pages_with_zero_blocks']}`",
                f"- pages_with_empty_text: `{audit['pages_with_empty_text']}`",
                f"- schema_missing_count: `{audit['schema_missing_count']}`",
                f"- invalid_schema_count: `{audit['invalid_schema_count']}`",
                f"- image_path_missing_count: `{audit['image_path_missing_count']}`",
                f"- image_path_mismatch_count: `{audit['image_path_mismatch_count']}`",
                f"- average_clean_text_length: `{audit['average_clean_text_length']:.3f}`",
                f"- average_perturbed_text_length: `{audit['average_page_text_length']:.3f}`",
                f"- average_clean_block_count: `{audit['average_clean_block_count']:.3f}`",
                f"- average_perturbed_block_count: `{audit['average_blocks']:.3f}`",
                f"- merged_jsonl_path: `{audit['path']}`",
                "",
            ]
        )
    lines.extend(["## Empty Page Details", ""])
    empty_details = [detail for audit in audits.values() for detail in audit["empty_page_details"]]
    if empty_details:
        for detail in empty_details:
            lines.extend(
                [
                    f"- page_id: `{detail['page_id']}`",
                    f"  pipeline: `{detail['pipeline']}`",
                    f"  condition: `{detail['condition']}`",
                    f"  image_path: `{detail['image_path']}`",
                    f"  TOR: `{detail['TOR']}`",
                    f"  support_bbox: `{detail['support_bbox']}`",
                    f"  reason / error_message: `{detail['reason']}`",
                    f"  empty_page_has_qa_pairs: `{'YES' if detail['has_qa_pairs'] else 'NO'}`",
                    f"  qa_pair_count: `{detail['qa_pair_count']}`",
                    "",
                ]
            )
    else:
        lines.append("- None.")
        lines.append("")
    lines.extend(["## Empty Page Has QA Pairs", ""])
    if empty_details:
        for detail in empty_details:
            lines.append(
                f"- {detail['pipeline']} / {detail['condition']} / {detail['page_id']}: "
                f"`{'YES' if detail['has_qa_pairs'] else 'NO'}` ({detail['qa_pair_count']})"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Readiness", ""])
    if reasons:
        lines.extend(f"- {reason}" for reason in reasons)
    else:
        lines.append("- All QA answering readiness checks passed for debug20.")
    lines.extend(
        [
            "",
            f"- ready_for_qa_answering: `{'YES' if ready else 'NO'}`",
            "",
            "No DeepSeek, QA answering, QA evaluation, or full500 code was called.",
            "",
            f"Ready for Prompt 8: {'YES' if ready else 'NO'}",
        ]
    )
    ensure_dir(path.parent)
    write_text(path, "\n".join(lines) + "\n")


def audit_perturbed_parse(config_path: str | Path, debug: bool = False) -> dict[str, Any]:
    pm = PathManager(config_path, create_dirs=True)
    manifest_path = pm.page_manifest_debug20 if debug else pm.page_manifest_500
    manifest = load_manifest(manifest_path)
    manifest_ids = {row["page_id"] for row in manifest}
    clean_refs = {pipeline: _read_clean_reference(pm, pipeline) for pipeline in PIPELINES}
    metadata = _read_metadata(pm)
    qa_page_counts = _read_qa_page_counts(pm)

    audits: dict[str, dict[str, Any]] = {}
    for pipeline in PIPELINES:
        for condition in CONDITIONS:
            key = _slug(pipeline, condition)
            audits[key] = _audit_pipeline_condition(
                pipeline=pipeline,
                condition=condition,
                path=_perturbed_merged_path(pm, pipeline, condition),
                manifest_ids=manifest_ids,
                pm=pm,
                clean_reference=clean_refs[pipeline],
                metadata=metadata,
                qa_page_counts=qa_page_counts,
            )

    ready, reasons = _readiness(audits)
    report_path = pm.shared_log_root / ("perturbed_parse_debug20_audit_report.md" if debug else "perturbed_parse_audit_report.md")
    _write_report(report_path, audits, ready, reasons)

    empty_details = [detail for audit in audits.values() for detail in audit["empty_page_details"]]
    return {
        "report_path": str(report_path),
        "ready_for_qa_answering": "YES" if ready else "NO",
        "ready_prompt8": "YES" if ready else "NO",
        "reasons": reasons,
        "empty_page_details": empty_details,
        "empty_page_has_qa_pairs": {
            f"{d['pipeline']}:{d['condition']}:{d['page_id']}": bool(d["has_qa_pairs"]) for d in empty_details
        },
        "audits": {
            key: {
                "success_pages": audit["success_pages"],
                "empty_pages": audit["empty_pages"],
                "failed_pages": audit["failed_pages"],
                "average_page_text_length": audit["average_page_text_length"],
                "mean_text_length_ratio": audit["mean_text_length_ratio"],
                "average_blocks": audit["average_blocks"],
                "mean_block_count_ratio": audit["mean_block_count_ratio"],
                "invalid_bbox_count": audit["invalid_bbox_count"],
                "schema_missing_count": audit["schema_missing_count"],
                "row_count": audit["row_count"],
            }
            for key, audit in audits.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit perturbed parse outputs.")
    parser.add_argument("--config", default="experiment_add/configs/base.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    result = audit_perturbed_parse(args.config, debug=args.debug)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ready_for_qa_answering"] == "YES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
