from __future__ import annotations

import argparse
import json
import re
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
from experiment_add.shared.text.answer_matching import contains_answer
from experiment_add.shared.text.normalize_text import normalize_answer
from experiment_add.shared.utils.io import ensure_dir, read_jsonl, write_text
from experiment_add.shared.utils.path_manager import PathManager


REQUIRED_QA_FIELDS = [
    "qa_id",
    "page_id",
    "question",
    "gold_answer",
    "gold_answer_normalized",
    "evidence_text",
    "answer_type",
    "generation_source_pipeline",
    "answer_occurrences_clean_mineru",
    "answer_occurrences_clean_ppstructure",
    "evidence_block_id_mineru",
    "evidence_block_id_ppstructure",
    "evidence_bbox_mineru",
    "evidence_bbox_ppstructure",
]
YES_NO_PREFIX = re.compile(r"^(is|are|was|were|do|does|did|can|could|should|would|will|has|have|had)\\b", re.I)
SUMMARY_STYLE = re.compile(r"\\b(summary|summarize|main idea|overall|purpose of (the|this) (document|paper|study))\\b", re.I)
TEMPLATE_STYLE = re.compile(r"\\b(according to the document|in this paper|according to this document)\\b", re.I)


def _token_len(text: str) -> int:
    return len(normalize_answer(text).split())


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _valid_bbox(bbox: Any, width: int, height: int) -> tuple[bool, bool]:
    """Return (is_valid, is_out_of_image)."""
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False, False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False, False
    if x2 <= x1 or y2 <= y1:
        return False, False
    tol_x = max(10.0, width * 0.02)
    tol_y = max(10.0, height * 0.02)
    out = x1 < -tol_x or y1 < -tol_y or x2 > width + tol_x or y2 > height + tol_y
    return True, out


def _read_jsonl_with_errors(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return records, [f"missing:{path}"]
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    errors.append(f"line_{line_no}:not_object")
            except json.JSONDecodeError as exc:
                errors.append(f"line_{line_no}:json_error:{exc}")
    return records, errors


def audit_qa_pairs(
    config_path: str | Path,
    debug: bool = False,
    qa_pairs_suffix: str | None = None,
) -> dict[str, Any]:
    """Audit filtered shared QA pairs against clean parse outputs.

    When ``qa_pairs_suffix`` is provided, audits the shared QA file under
    ``experiment_add/outputs/shared/qa_pairs_<suffix>/`` instead of the
    canonical ``qa_pairs/`` directory.
    """
    pm = PathManager(config_path, create_dirs=True, qa_pairs_suffix=qa_pairs_suffix)
    manifest = load_manifest(pm.page_manifest_debug20 if debug else pm.page_manifest_500)
    manifest_ids = {row["page_id"] for row in manifest}
    manifest_by_id = {row["page_id"]: row for row in manifest}
    qa_path = pm.qa_pairs_shared_path
    qa_rows, json_errors = _read_jsonl_with_errors(qa_path)
    mineru = {row["page_id"]: row for row in read_jsonl(pm.clean_parse_merged_path("mineru"))}
    pp = {row["page_id"]: row for row in read_jsonl(pm.clean_parse_merged_path("ppstructure"))}

    schema_invalid_rows: list[str] = []
    schema_missing_count = 0
    answer_presence_failures_mineru = 0
    answer_presence_failures_ppstructure = 0
    answer_too_frequent_mineru = 0
    answer_too_frequent_ppstructure = 0
    answer_length_invalid_count = 0
    empty_gold_answer_count = 0
    empty_normalized_answer_count = 0
    empty_question_count = 0
    too_short_question_count = 0
    yes_no_question_count = 0
    summary_style_question_count = 0
    template_style_question_count = 0
    invalid_bbox_count_mineru = 0
    invalid_bbox_count_ppstructure = 0
    missing_evidence_block_mineru = 0
    missing_evidence_block_ppstructure = 0
    bbox_out_of_image_count_mineru = 0
    bbox_out_of_image_count_ppstructure = 0
    answer_lengths: list[int] = []
    abnormal: list[dict[str, str]] = []

    qa_ids = [str(row.get("qa_id", "")) for row in qa_rows]
    duplicate_question_count = 0
    question_seen: set[str] = set()
    page_counts: Counter[str] = Counter()
    for row in qa_rows:
        qa_id = str(row.get("qa_id", "unknown"))
        page_id = str(row.get("page_id", ""))
        page_counts[page_id] += 1
        missing = [field for field in REQUIRED_QA_FIELDS if field not in row]
        if missing:
            schema_missing_count += len(missing)
            schema_invalid_rows.append(qa_id)
            abnormal.append({"page_id": page_id, "qa_id": qa_id, "reason": f"schema_missing:{missing}"})
            continue
        if page_id not in manifest_ids:
            abnormal.append({"page_id": page_id, "qa_id": qa_id, "reason": "page_not_in_manifest"})
        raw_question = row.get("question")
        raw_answer = row.get("gold_answer")
        question = "" if raw_question is None else str(raw_question).strip()
        answer = "" if raw_answer is None else str(raw_answer).strip()
        answer_norm = normalize_answer(raw_answer)
        if raw_answer is None or not answer:
            empty_gold_answer_count += 1
            abnormal.append({"page_id": page_id, "qa_id": qa_id, "reason": "empty_gold_answer"})
        if not answer_norm:
            empty_normalized_answer_count += 1
            abnormal.append({"page_id": page_id, "qa_id": qa_id, "reason": "empty_normalized_answer"})
        if normalize_answer(row.get("gold_answer_normalized")) != answer_norm:
            abnormal.append({"page_id": page_id, "qa_id": qa_id, "reason": "gold_answer_normalized_mismatch"})
        if not question:
            empty_question_count += 1
        if _token_len(question) < 4:
            too_short_question_count += 1
        nq = normalize_answer(question)
        if nq in question_seen:
            duplicate_question_count += 1
        question_seen.add(nq)
        if YES_NO_PREFIX.search(question):
            yes_no_question_count += 1
        if SUMMARY_STYLE.search(question):
            summary_style_question_count += 1
        if TEMPLATE_STYLE.search(question):
            template_style_question_count += 1
        alen = _token_len(answer)
        answer_lengths.append(alen)
        if alen < 2 or alen > 15:
            answer_length_invalid_count += 1

        mineru_text = str(mineru.get(page_id, {}).get("page_text", ""))
        pp_text = str(pp.get(page_id, {}).get("page_text", ""))
        if not contains_answer(mineru_text, answer):
            answer_presence_failures_mineru += 1
            abnormal.append({"page_id": page_id, "qa_id": qa_id, "reason": "answer_missing_mineru"})
        if not contains_answer(pp_text, answer):
            answer_presence_failures_ppstructure += 1
            abnormal.append({"page_id": page_id, "qa_id": qa_id, "reason": "answer_missing_ppstructure"})
        if _to_int(row.get("answer_occurrences_clean_mineru")) > 3:
            answer_too_frequent_mineru += 1
        if _to_int(row.get("answer_occurrences_clean_ppstructure")) > 3:
            answer_too_frequent_ppstructure += 1

        dims = manifest_by_id.get(page_id, {})
        width = _to_int(dims.get("width"))
        height = _to_int(dims.get("height"))
        ok_m, out_m = _valid_bbox(row.get("evidence_bbox_mineru"), width, height)
        ok_p, out_p = _valid_bbox(row.get("evidence_bbox_ppstructure"), width, height)
        if not row.get("evidence_block_id_mineru"):
            missing_evidence_block_mineru += 1
        if not row.get("evidence_block_id_ppstructure"):
            missing_evidence_block_ppstructure += 1
        if not ok_m:
            invalid_bbox_count_mineru += 1
            abnormal.append({"page_id": page_id, "qa_id": qa_id, "reason": "invalid_bbox_mineru"})
        if not ok_p:
            invalid_bbox_count_ppstructure += 1
            abnormal.append({"page_id": page_id, "qa_id": qa_id, "reason": "invalid_bbox_ppstructure"})
        if out_m:
            bbox_out_of_image_count_mineru += 1
        if out_p:
            bbox_out_of_image_count_ppstructure += 1

    duplicate_qa_ids = [qa_id for qa_id, n in Counter(qa_ids).items() if qa_id and n > 1]
    total_pages = len(manifest)
    pages_with_qa = len(set(page_counts))
    pages_with_0_qa = max(0, total_pages - pages_with_qa)
    per_page_distribution = Counter(page_counts.values())
    answer_type_distribution = dict(Counter(row.get("answer_type", "unknown") for row in qa_rows))
    source_distribution = dict(Counter(row.get("generation_source_pipeline", "unknown") for row in qa_rows))
    mean_answer_tokens = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0
    min_answer_tokens = min(answer_lengths) if answer_lengths else 0
    max_answer_tokens = max(answer_lengths) if answer_lengths else 0

    severe_quality_fail = schema_missing_count > 0 or empty_gold_answer_count > 0 or empty_normalized_answer_count > 0 or empty_question_count > 0 or answer_presence_failures_mineru > 0 or answer_presence_failures_ppstructure > 0
    bbox_serious = invalid_bbox_count_mineru > 5 or invalid_bbox_count_ppstructure > 5
    pass_quantity = len(qa_rows) >= 40 and (len(qa_rows) / max(1, total_pages)) >= 2 and pages_with_qa >= 15
    borderline_quantity = len(qa_rows) >= 35 and (len(qa_rows) / max(1, total_pages)) >= 1.5 and pages_with_qa >= 10
    if severe_quality_fail or bbox_serious or len(qa_rows) < 20 or pages_with_qa < 8:
        readiness = "NO"
    elif pass_quantity:
        readiness = "YES"
    elif borderline_quantity:
        readiness = "BORDERLINE"
    else:
        readiness = "NO"

    causes = []
    if answer_presence_failures_mineru or answer_presence_failures_ppstructure:
        causes.append("generation quality problem or cross-parser text mismatch")
    if schema_missing_count:
        causes.append("schema problem")
    if empty_gold_answer_count or empty_normalized_answer_count or empty_question_count:
        causes.append("empty QA question/answer problem")
    if invalid_bbox_count_mineru or invalid_bbox_count_ppstructure:
        causes.append("evidence locating problem")
    if len(qa_rows) < 40 or pages_with_qa < 15:
        causes.append("filtering too strict or insufficient candidate count")

    report_path = pm.qa_pairs_dir / "qa_pairs_debug20_audit_report.md"
    lines = [
        "# QA Pairs Debug20 Audit Report",
        "",
        f"- qa_pairs_suffix: `{pm.qa_pairs_suffix or 'canonical'}`",
        f"- qa_pairs_shared_path: `{qa_path}`",
        f"- total_shared_qa: `{len(qa_rows)}`",
        f"- unique_qa_id_count: `{len(set(qa_ids))}`",
        f"- pages_with_qa: `{pages_with_qa}`",
        f"- pages_with_0_qa: `{pages_with_0_qa}`",
        f"- average_qa_per_page: `{len(qa_rows) / max(1, total_pages):.3f}`",
        f"- schema_missing_count: `{schema_missing_count}`",
        f"- schema_invalid_rows: `{schema_invalid_rows}`",
        f"- answer_presence_failures_mineru: `{answer_presence_failures_mineru}`",
        f"- answer_presence_failures_ppstructure: `{answer_presence_failures_ppstructure}`",
        f"- answer_too_frequent_mineru: `{answer_too_frequent_mineru}`",
        f"- answer_too_frequent_ppstructure: `{answer_too_frequent_ppstructure}`",
        f"- answer_length_invalid_count: `{answer_length_invalid_count}`",
        f"- empty_gold_answer_count: `{empty_gold_answer_count}`",
        f"- empty_normalized_answer_count: `{empty_normalized_answer_count}`",
        f"- min_answer_tokens: `{min_answer_tokens}`",
        f"- max_answer_tokens: `{max_answer_tokens}`",
        f"- mean_answer_tokens: `{mean_answer_tokens:.3f}`",
        f"- empty_question_count: `{empty_question_count}`",
        f"- too_short_question_count: `{too_short_question_count}`",
        f"- duplicate_question_count: `{duplicate_question_count}`",
        f"- yes_no_question_count: `{yes_no_question_count}`",
        f"- summary_style_question_count: `{summary_style_question_count}`",
        f"- template_style_question_count: `{template_style_question_count}`",
        f"- invalid_bbox_count_mineru: `{invalid_bbox_count_mineru}`",
        f"- invalid_bbox_count_ppstructure: `{invalid_bbox_count_ppstructure}`",
        f"- missing_evidence_block_mineru: `{missing_evidence_block_mineru}`",
        f"- missing_evidence_block_ppstructure: `{missing_evidence_block_ppstructure}`",
        f"- bbox_out_of_image_count_mineru: `{bbox_out_of_image_count_mineru}`",
        f"- bbox_out_of_image_count_ppstructure: `{bbox_out_of_image_count_ppstructure}`",
        f"- answer_type_distribution: `{answer_type_distribution}`",
        f"- generation_source_distribution: `{source_distribution}`",
        f"- qa_per_page_distribution: `{dict(per_page_distribution)}`",
        f"- duplicate_qa_id_count: `{len(duplicate_qa_ids)}`",
        f"- json_error_count: `{len(json_errors)}`",
        f"- ready_for_perturbation: `{readiness}`",
        "",
        "## Diagnosis",
        "",
    ]
    lines.extend(f"- {cause}" for cause in causes) if causes else lines.append("- No quality blockers found.")
    lines.extend(["", "## Abnormal Rows", ""])
    if abnormal:
        lines.append("| page_id | qa_id | reason |")
        lines.append("| --- | --- | --- |")
        for item in abnormal[:100]:
            lines.append(f"| `{item['page_id']}` | `{item['qa_id']}` | `{item['reason']}` |")
    else:
        lines.append("- None.")
    lines.extend(["", "## Recommendation", ""])
    if readiness == "YES":
        lines.append("Ready for Prompt 6: YES")
    elif readiness == "BORDERLINE":
        lines.append("Ready for Prompt 6 debug-only: YES")
        lines.append("Recommended before full500: improve QA coverage")
    else:
        lines.append("Ready for Prompt 6: NO")
    lines.append("")
    lines.append("No DeepSeek, perturbation, QA answering, QA evaluation, or full500 run was called by this audit.")
    ensure_dir(report_path.parent)
    write_text(report_path, "\n".join(lines) + "\n")
    return {
        "qa_pairs_suffix": pm.qa_pairs_suffix or "canonical",
        "qa_pairs_shared_path": str(qa_path),
        "total_shared_qa": len(qa_rows),
        "pages_with_qa": pages_with_qa,
        "pages_with_0_qa": pages_with_0_qa,
        "average_qa_per_page": len(qa_rows) / max(1, total_pages),
        "schema_missing_count": schema_missing_count,
        "answer_presence_failures_mineru": answer_presence_failures_mineru,
        "answer_presence_failures_ppstructure": answer_presence_failures_ppstructure,
        "empty_gold_answer_count": empty_gold_answer_count,
        "empty_normalized_answer_count": empty_normalized_answer_count,
        "empty_question_count": empty_question_count,
        "answer_length_invalid_count": answer_length_invalid_count,
        "invalid_bbox_count_mineru": invalid_bbox_count_mineru,
        "invalid_bbox_count_ppstructure": invalid_bbox_count_ppstructure,
        "ready_for_perturbation": readiness,
        "report_path": str(report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit shared QA pairs.")
    parser.add_argument("--config", default="experiment_add/configs/base.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--input-suffix",
        default=None,
        help=(
            "If set, audit QA pair artifacts under "
            "experiment_add/outputs/shared/qa_pairs_<suffix>/ instead of "
            "the canonical qa_pairs/ directory."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Alias for --input-suffix; if both are given they must match.",
    )
    args = parser.parse_args()
    if args.input_suffix and args.output_suffix and args.input_suffix != args.output_suffix:
        parser.error("--input-suffix and --output-suffix must match if both are given")
    suffix = args.input_suffix or args.output_suffix
    result = audit_qa_pairs(args.config, debug=args.debug, qa_pairs_suffix=suffix)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ready_for_perturbation"] in {"YES", "BORDERLINE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
