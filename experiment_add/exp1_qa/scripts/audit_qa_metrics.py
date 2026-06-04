from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiment_add.shared.utils.io import ensure_dir, write_text
from experiment_add.shared.utils.path_manager import PathManager


METRIC_FILES = {
    "pipeline_condition": "qa_metrics_by_pipeline_condition.csv",
    "page": "qa_metrics_by_page.csv",
    "question_type": "qa_metrics_by_question_type.csv",
    "non_overlap": "qa_metrics_non_overlap_subset.csv",
    "correlations": "qa_metrics_correlations.csv",
    "failure": "qa_failure_decomposition.csv",
}
RATE_FIELDS = ("answer_missing_rate", "not_found_rate", "qa_em", "qa_f1")


def _base_config_path(config_path: Path) -> Path:
    return config_path.parent / "base.yaml"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _count_status(rows: list[dict[str, Any]], status: str) -> int:
    return sum(1 for row in rows if str(row.get("answer_status", "")) == status)


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# QA Metrics Debug Audit Report",
        "",
        f"- metrics_files_exist: `{result['metrics_files_exist']}`",
        f"- pipeline_condition_rows: `{result['pipeline_condition_rows']}`",
        f"- page_rows: `{result['page_rows']}`",
        f"- question_type_rows: `{result['question_type_rows']}`",
        f"- failure_decomposition_rows: `{result['failure_decomposition_rows']}`",
        f"- api_failed_count: `{result['api_failed_count']}`",
        f"- parser_empty_count: `{result['parser_empty_count']}`",
        f"- clean_answer_missing_mineru: `{result['clean_answer_missing_mineru']}`",
        f"- clean_answer_missing_ppstructure: `{result['clean_answer_missing_ppstructure']}`",
        f"- clean_qa_em_mineru: `{result['clean_qa_em_mineru']}`",
        f"- clean_qa_em_ppstructure: `{result['clean_qa_em_ppstructure']}`",
        f"- negative_qa_drop_detected: `{result['negative_qa_drop_detected']}`",
        f"- correlation_available: `{result['correlation_available']}`",
        f"- metrics_ready_for_tables: `{result['metrics_ready_for_tables']}`",
        "",
        "## Negative QA-Drop Details",
        "",
    ]
    if result["negative_qa_drop_details"]:
        for item in result["negative_qa_drop_details"]:
            lines.append(f"- `{item['pipeline']}` / `{item['condition']}`: qa_drop=`{item['qa_drop']}`")
        lines.extend(
            [
                "",
                "Negative QA-Drop in MinerU debug20 likely reflects small-sample variance or LLM answering variability; full500 is required before drawing conclusions.",
            ]
        )
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    if result["warnings"]:
        lines.extend(f"- {warning}" for warning in result["warnings"])
    else:
        lines.append("- None.")
    ensure_dir(path.parent)
    write_text(path, "\n".join(lines) + "\n")


def audit_metrics(config_path: str | Path, debug: bool = False) -> dict[str, Any]:
    pm = PathManager(_base_config_path(Path(config_path)), create_dirs=True)
    metrics_dir = pm.exp1_metrics_dir
    paths = {key: metrics_dir / name for key, name in METRIC_FILES.items()}
    rows = {key: _read_csv(path) for key, path in paths.items()}
    metrics_files_exist = all(path.exists() and path.stat().st_size > 0 and rows[key] for key, path in paths.items())
    pc_rows = rows["pipeline_condition"]
    page_rows = rows["page"]
    qt_rows = rows["question_type"]
    failure_rows = rows["failure"]
    corr_rows = rows["correlations"]
    non_overlap_rows = rows["non_overlap"]

    warnings: list[str] = []
    ready = True
    expected_counts = {"pipeline_condition": 8, "page": 96, "question_type": 32, "failure": 304}
    if not metrics_files_exist:
        ready = False
        warnings.append("One or more required metrics CSV files are missing or empty.")
    for key, expected in expected_counts.items():
        if len(rows[key]) != expected:
            ready = False
            warnings.append(f"{key} rows {len(rows[key])} != expected {expected}.")

    pc_by_key = {(row.get("pipeline"), row.get("condition")): row for row in pc_rows}
    for pipeline in ("mineru", "ppstructure"):
        clean = pc_by_key.get((pipeline, "clean"), {})
        if _float(clean.get("qa_drop", 0.0)) not in (0.0,):
            ready = False
            warnings.append(f"{pipeline} clean qa_drop is not 0.")
        if str(clean.get("qa_drop_per_tor", "")).strip() not in {"", "nan", "NaN"}:
            ready = False
            warnings.append(f"{pipeline} clean qa_drop_per_tor should be blank/NaN.")

    for row in pc_rows:
        condition = str(row.get("condition", ""))
        if condition != "clean" and str(row.get("mean_tor", "")).strip() in {"", "nan", "NaN"}:
            ready = False
            warnings.append(f"{row.get('pipeline')} {condition} mean_tor missing.")
        for field in RATE_FIELDS:
            value = _float(row.get(field))
            if not (0.0 <= value <= 1.0):
                ready = False
                warnings.append(f"{field} out of [0,1] for {row.get('pipeline')} {condition}: {row.get(field)}")

    negative = [
        {"pipeline": row.get("pipeline"), "condition": row.get("condition"), "qa_drop": _float(row.get("qa_drop"))}
        for row in pc_rows
        if str(row.get("condition")) != "clean" and _float(row.get("qa_drop")) < 0
    ]

    parser_empty = _count_status(failure_rows, "parser_empty")
    api_failed = _count_status(failure_rows, "api_failed")
    clean_missing_mineru = _float(pc_by_key.get(("mineru", "clean"), {}).get("answer_missing_rate", "nan"))
    clean_missing_pp = _float(pc_by_key.get(("ppstructure", "clean"), {}).get("answer_missing_rate", "nan"))
    clean_em_mineru = _float(pc_by_key.get(("mineru", "clean"), {}).get("qa_em", "nan"))
    clean_em_pp = _float(pc_by_key.get(("ppstructure", "clean"), {}).get("qa_em", "nan"))
    if api_failed != 0:
        ready = False
        warnings.append(f"api_failed_count is {api_failed}; expected 0.")
    if parser_empty != 4:
        ready = False
        warnings.append(f"parser_empty_count is {parser_empty}; expected 4.")
    if clean_missing_mineru != 0.0 or clean_missing_pp != 0.0:
        warnings.append("Clean AnswerMissing is non-zero.")

    corr_nan = sum(1 for row in corr_rows if _is_nan(row.get("spearman")))
    correlation_available = "YES" if corr_rows and corr_nan == 0 else ("PARTIAL" if corr_rows else "NO")
    if corr_nan:
        warnings.append("Some correlations are NaN because parser metrics such as B-SLR / SLR_topo / CER are not fully available.")

    result = {
        "metrics_files_exist": "YES" if metrics_files_exist else "NO",
        "pipeline_condition_rows": len(pc_rows),
        "page_rows": len(page_rows),
        "question_type_rows": len(qt_rows),
        "failure_decomposition_rows": len(failure_rows),
        "api_failed_count": api_failed,
        "parser_empty_count": parser_empty,
        "clean_answer_missing_mineru": clean_missing_mineru,
        "clean_answer_missing_ppstructure": clean_missing_pp,
        "clean_qa_em_mineru": clean_em_mineru,
        "clean_qa_em_ppstructure": clean_em_pp,
        "negative_qa_drop_detected": "YES" if negative else "NO",
        "negative_qa_drop_details": negative,
        "correlation_available": correlation_available,
        "metrics_ready_for_tables": "YES" if ready else "NO",
        "warnings": warnings,
        "report_path": str(pm.exp1_log_root / "qa_metrics_debug_audit_report.md"),
        "non_overlap_rows": len(non_overlap_rows),
    }
    _write_report(Path(result["report_path"]), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit debug QA metrics before table generation.")
    parser.add_argument("--config", default="experiment_add/configs/exp1_qa.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    result = audit_metrics(args.config, debug=args.debug)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["metrics_ready_for_tables"] == "YES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
