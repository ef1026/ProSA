from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiment_add.exp1_qa.metrics.qa_failure_decomposition import classify_failure, failure_row
from experiment_add.exp1_qa.metrics.qa_metrics import answer_missing, qa_em, qa_f1, summarize_instances
from experiment_add.exp1_qa.metrics.qa_non_overlap_analysis import evidence_bbox_for_pipeline, is_non_overlap_subset
from experiment_add.shared.metrics.aggregation import safe_divide
from experiment_add.shared.metrics.correlation_utils import spearman
from experiment_add.shared.metrics.parser_metrics_loader import load_parser_metrics
from experiment_add.shared.utils.io import ensure_dir, read_jsonl, write_text
from experiment_add.shared.utils.path_manager import PathManager


PIPELINES = ("mineru", "ppstructure")
CONDITIONS = ("clean", "area_matched_erasure", "structural_probe", "large_area_erasure")
PERTURBED_CONDITIONS = ("area_matched_erasure", "structural_probe", "large_area_erasure")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _base_config_path(config_path: Path) -> Path:
    return config_path.parent / "base.yaml"


def _answer_path(pm: PathManager, pipeline: str, condition: str) -> Path:
    return pm.exp1_answers_path(pipeline, condition)


def _parse_path(pm: PathManager, pipeline: str, condition: str) -> Path:
    if condition == "clean":
        return pm.clean_parse_merged_path(pipeline)
    return pm.perturbed_parse_dir(pipeline, condition) / "merged.jsonl"


def _load_parse_texts(pm: PathManager) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for pipeline in PIPELINES:
        for condition in CONDITIONS:
            for row in read_jsonl(_parse_path(pm, pipeline, condition)):
                out[(pipeline, condition, str(row.get("page_id", "")))] = row
    return out


def _load_qa_pairs(pm: PathManager) -> dict[str, dict[str, Any]]:
    return {str(row.get("qa_id", "")): row for row in read_jsonl(pm.qa_pairs_shared_path)}


def _load_metadata(pm: PathManager) -> dict[tuple[str, str], dict[str, Any]]:
    primary = pm.project_root / "experiment_add/outputs/shared/perturbed_pages/merged_perturb_metadata.jsonl"
    path = primary if primary.exists() else pm.merged_perturb_metadata_path
    return {(str(row.get("page_id", "")), str(row.get("condition", ""))): row for row in read_jsonl(path)}


def _load_mean_tor(pm: PathManager, metadata: dict[tuple[str, str], dict[str, Any]]) -> dict[str, float]:
    means = {condition: 0.0 for condition in CONDITIONS}
    summary = pm.project_root / "experiment_add/outputs/shared/perturbed_pages/perturb_summary.csv"
    for row in _read_csv(summary):
        condition = str(row.get("condition", ""))
        if condition in means:
            try:
                means[condition] = float(row.get("mean_TOR", 0.0))
            except (TypeError, ValueError):
                pass
    for condition in PERTURBED_CONDITIONS:
        if means.get(condition, 0.0) == 0.0:
            tors = [float(row.get("TOR", 0.0)) for (page_id, cond), row in metadata.items() if cond == condition]
            means[condition] = sum(tors) / len(tors) if tors else 0.0
    return means


def _flatten_instances(pm: PathManager) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, float]]:
    qa_by_id = _load_qa_pairs(pm)
    parse_by_key = _load_parse_texts(pm)
    metadata = _load_metadata(pm)
    mean_tor = _load_mean_tor(pm, metadata)
    rows: list[dict[str, Any]] = []
    for pipeline in PIPELINES:
        for condition in CONDITIONS:
            for batch in read_jsonl(_answer_path(pm, pipeline, condition)):
                page_id = str(batch.get("page_id", ""))
                parser_status = str(batch.get("parser_status", ""))
                parse = parse_by_key.get((pipeline, condition, page_id), {})
                page_text = str(parse.get("page_text", ""))
                if condition == "clean":
                    page_text = str(parse_by_key.get((pipeline, "clean", page_id), {}).get("page_text", ""))
                for answer in batch.get("answers", []):
                    qa_id = str(answer.get("qa_id", ""))
                    qa = qa_by_id.get(qa_id, {})
                    gold = str(qa.get("gold_answer", ""))
                    status = str(answer.get("status", ""))
                    pred = str(answer.get("pred_answer", ""))
                    missing = answer_missing(page_text, gold, parser_status=parser_status)
                    em = qa_em(pred, gold, status)
                    f1 = qa_f1(pred, gold, status)
                    row = {
                        "pipeline": pipeline,
                        "condition": condition,
                        "page_id": page_id,
                        "qa_id": qa_id,
                        "question": answer.get("question", qa.get("question", "")),
                        "answer_type": qa.get("answer_type", "unknown"),
                        "gold_answer": gold,
                        "pred_answer": pred,
                        "answer_status": status,
                        "api_status": batch.get("api_status"),
                        "parser_status": parser_status,
                        "answer_missing": missing,
                        "em": em,
                        "f1": f1,
                        "TOR": metadata.get((page_id, condition), {}).get("TOR", 0.0 if condition == "clean" else mean_tor.get(condition, "")),
                    }
                    row["failure_type"] = classify_failure(row)
                    rows.append(row)
    return rows, qa_by_id, metadata, mean_tor


def _metrics_by_pipeline_condition(instances: list[dict[str, Any]], mean_tor: dict[str, float]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in instances:
        grouped[(row["pipeline"], row["condition"])].append(row)
    clean_em = {
        pipeline: summarize_instances(grouped.get((pipeline, "clean"), []))["qa_em"]
        for pipeline in PIPELINES
    }
    rows: list[dict[str, Any]] = []
    for pipeline in PIPELINES:
        for condition in CONDITIONS:
            items = grouped.get((pipeline, condition), [])
            summary = summarize_instances(items)
            qa_drop = 0.0 if condition == "clean" else clean_em[pipeline] - summary["qa_em"]
            tor = 0.0 if condition == "clean" else mean_tor.get(condition, float("nan"))
            rows.append(
                {
                    "pipeline": pipeline,
                    "condition": condition,
                    **summary,
                    "qa_drop": qa_drop,
                    "mean_tor": "" if condition == "clean" else tor,
                    "qa_drop_per_tor": "" if condition == "clean" else safe_divide(qa_drop, tor),
                }
            )
    return rows


def _metrics_by_page(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in instances:
        grouped[(row["pipeline"], row["condition"], row["page_id"])].append(row)
    rows: list[dict[str, Any]] = []
    for (pipeline, condition, page_id), items in sorted(grouped.items()):
        summary = summarize_instances(items)
        failures = Counter(str(row.get("failure_type", "")) for row in items)
        rows.append(
            {
                "pipeline": pipeline,
                "condition": condition,
                "page_id": page_id,
                "num_qa": len(items),
                "parser_status": items[0].get("parser_status", ""),
                "answer_missing_rate": summary["answer_missing_rate"],
                "not_found_rate": summary["not_found_rate"],
                "qa_em": summary["qa_em"],
                "qa_f1": summary["qa_f1"],
                "num_correct": failures["Correct"],
                "num_answer_lost": failures["Answer Lost"],
                "num_not_found": failures["Answer Present but NOT_FOUND"],
                "num_wrong": failures["Answer Present but Wrong"],
                "num_parser_empty": failures["Parser Empty / Invalid"],
                "TOR": items[0].get("TOR", ""),
            }
        )
    return rows


def _metrics_by_question_type(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in instances:
        grouped[(row["pipeline"], row["condition"], str(row.get("answer_type", "unknown")))].append(row)
    clean_em: dict[tuple[str, str], float] = {}
    for pipeline in PIPELINES:
        for answer_type in {str(row.get("answer_type", "unknown")) for row in instances}:
            clean_em[(pipeline, answer_type)] = summarize_instances(grouped.get((pipeline, "clean", answer_type), []))["qa_em"]
    rows: list[dict[str, Any]] = []
    for (pipeline, condition, answer_type), items in sorted(grouped.items()):
        summary = summarize_instances(items)
        drop = 0.0 if condition == "clean" else clean_em.get((pipeline, answer_type), 0.0) - summary["qa_em"]
        rows.append(
            {
                "pipeline": pipeline,
                "condition": condition,
                "answer_type": answer_type,
                "num_qa": len(items),
                "answer_missing_rate": summary["answer_missing_rate"],
                "not_found_rate": summary["not_found_rate"],
                "qa_em": summary["qa_em"],
                "qa_f1": summary["qa_f1"],
                "qa_drop": drop,
            }
        )
    return rows


def _failure_decomposition(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [failure_row(row) for row in instances]


def _non_overlap_metrics(
    instances: list[dict[str, Any]],
    qa_by_id: dict[str, dict[str, Any]],
    metadata: dict[tuple[str, str], dict[str, Any]],
    mean_tor: dict[str, float],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {
        (row["pipeline"], row["condition"], row["qa_id"]): row for row in instances
    }
    out: list[dict[str, Any]] = []
    for pipeline in PIPELINES:
        clean_rows = [row for row in instances if row["pipeline"] == pipeline and row["condition"] == "clean"]
        clean_summary = summarize_instances(clean_rows)
        out.append(
            {
                "pipeline": pipeline,
                "condition": "clean",
                "num_qa_non_overlap": len(clean_rows),
                "answer_missing": clean_summary["answer_missing_rate"],
                "not_found_rate": clean_summary["not_found_rate"],
                "qa_em": clean_summary["qa_em"],
                "qa_f1": clean_summary["qa_f1"],
                "qa_drop": 0.0,
                "mean_tor": "",
                "qa_drop_per_tor": "",
            }
        )
        for condition in PERTURBED_CONDITIONS:
            subset: list[dict[str, Any]] = []
            clean_subset: list[dict[str, Any]] = []
            for row in instances:
                if row["pipeline"] != pipeline or row["condition"] != condition:
                    continue
                qa = qa_by_id.get(row["qa_id"], {})
                support = metadata.get((row["page_id"], condition), {}).get("support_bbox")
                evidence = evidence_bbox_for_pipeline(qa, pipeline)
                if is_non_overlap_subset(evidence, support):
                    subset.append(row)
                    clean_row = by_key.get((pipeline, "clean", row["qa_id"]))
                    if clean_row:
                        clean_subset.append(clean_row)
            summary = summarize_instances(subset)
            clean_subset_summary = summarize_instances(clean_subset)
            qa_drop = clean_subset_summary["qa_em"] - summary["qa_em"] if subset else 0.0
            tor = mean_tor.get(condition, float("nan"))
            out.append(
                {
                    "pipeline": pipeline,
                    "condition": condition,
                    "num_qa_non_overlap": len(subset),
                    "answer_missing": summary["answer_missing_rate"],
                    "not_found_rate": summary["not_found_rate"],
                    "qa_em": summary["qa_em"],
                    "qa_f1": summary["qa_f1"],
                    "qa_drop": qa_drop,
                    "mean_tor": tor,
                    "qa_drop_per_tor": safe_divide(qa_drop, tor),
                }
            )
    return out


def _correlations(
    pc_rows: list[dict[str, Any]],
    page_rows: list[dict[str, Any]],
    parser_metric_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    # TOR is always available from the QA metrics rows.
    for target in ("qa_drop", "answer_missing_rate"):
        pairs = [(row.get("mean_tor"), row.get(target)) for row in pc_rows if row.get("condition") != "clean"]
        rows.append({"x_metric": "TOR", "y_metric": target, "spearman": spearman(pairs), "n": len(pairs), "source": "perturb_metadata"})
    available = "PARTIAL"
    # Best-effort parser metrics; do not block evaluation.
    for metric in ("B_SLR", "SLR_topo", "delta_CER", "CER_matched_mean"):
        metric_pairs = []
        for row in parser_metric_rows:
            y = row.get("QA-Drop") or row.get("qa_drop")
            if y is not None:
                metric_pairs.append((row.get(metric), y))
        rows.append({"x_metric": metric, "y_metric": "qa_drop", "spearman": spearman(metric_pairs), "n": len(metric_pairs), "source": "parser_metrics_best_effort"})
    if not parser_metric_rows:
        available = "PARTIAL"
    return rows, available


def _write_summary(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# QA Evaluation Debug Summary",
        "",
        f"- total_answer_items: `{result['total_answer_items']}`",
        f"- api_failed_count: `{result['api_failed_count']}`",
        f"- parser_empty_count: `{result['parser_empty_count']}`",
        f"- clean_answer_missing_by_pipeline: `{result['clean_answer_missing_by_pipeline']}`",
        f"- clean_qa_em_by_pipeline: `{result['clean_qa_em_by_pipeline']}`",
        f"- qa_drop_by_pipeline_condition: `{result['qa_drop_by_pipeline_condition']}`",
        f"- qa_drop_per_tor_by_pipeline_condition: `{result['qa_drop_per_tor_by_pipeline_condition']}`",
        f"- non_overlap_subset_size_by_pipeline_condition: `{result['non_overlap_subset_size_by_pipeline_condition']}`",
        f"- correlation_available: `{result['correlation_available']}`",
        f"- warnings: `{result['warnings']}`",
        f"- ready_for_table_generation: `{result['ready_for_table_generation']}`",
        "",
    ]
    write_text(path, "\n".join(lines))


def evaluate(config_path: str | Path, debug: bool = False) -> dict[str, Any]:
    config_path = Path(config_path)
    pm = PathManager(_base_config_path(config_path), create_dirs=True)
    metrics_dir = pm.exp1_metrics_dir
    ensure_dir(metrics_dir)
    instances, qa_by_id, metadata, mean_tor = _flatten_instances(pm)
    pc_rows = _metrics_by_pipeline_condition(instances, mean_tor)
    page_rows = _metrics_by_page(instances)
    qt_rows = _metrics_by_question_type(instances)
    failure_rows = _failure_decomposition(instances)
    non_overlap_rows = _non_overlap_metrics(instances, qa_by_id, metadata, mean_tor)
    parser_metric_rows, parser_metric_sources = load_parser_metrics(pm.project_root)
    corr_rows, corr_available = _correlations(pc_rows, page_rows, parser_metric_rows)

    _write_csv(
        metrics_dir / "qa_metrics_by_pipeline_condition.csv",
        pc_rows,
        ["pipeline", "condition", "num_pages", "num_qa", "num_answer_items", "parser_empty_count", "api_failed_count", "answer_missing_rate", "not_found_rate", "qa_em", "qa_f1", "qa_drop", "mean_tor", "qa_drop_per_tor"],
    )
    _write_csv(
        metrics_dir / "qa_metrics_by_page.csv",
        page_rows,
        ["pipeline", "condition", "page_id", "num_qa", "parser_status", "answer_missing_rate", "not_found_rate", "qa_em", "qa_f1", "num_correct", "num_answer_lost", "num_not_found", "num_wrong", "num_parser_empty", "TOR"],
    )
    _write_csv(
        metrics_dir / "qa_metrics_by_question_type.csv",
        qt_rows,
        ["pipeline", "condition", "answer_type", "num_qa", "answer_missing_rate", "not_found_rate", "qa_em", "qa_f1", "qa_drop"],
    )
    _write_csv(
        metrics_dir / "qa_metrics_non_overlap_subset.csv",
        non_overlap_rows,
        ["pipeline", "condition", "num_qa_non_overlap", "answer_missing", "not_found_rate", "qa_em", "qa_f1", "qa_drop", "mean_tor", "qa_drop_per_tor"],
    )
    _write_csv(
        metrics_dir / "qa_metrics_correlations.csv",
        corr_rows,
        ["x_metric", "y_metric", "spearman", "n", "source"],
    )
    _write_csv(
        metrics_dir / "qa_failure_decomposition.csv",
        failure_rows,
        ["pipeline", "condition", "page_id", "qa_id", "question", "gold_answer", "pred_answer", "answer_status", "parser_status", "answer_missing", "em", "f1", "failure_type"],
    )

    pc_by_key = {(row["pipeline"], row["condition"]): row for row in pc_rows}
    clean_answer_missing = {pipeline: pc_by_key[(pipeline, "clean")]["answer_missing_rate"] for pipeline in PIPELINES}
    clean_qa_em = {pipeline: pc_by_key[(pipeline, "clean")]["qa_em"] for pipeline in PIPELINES}
    qa_drop = {f"{row['pipeline']}_{row['condition']}": row["qa_drop"] for row in pc_rows if row["condition"] != "clean"}
    qa_drop_per_tor = {f"{row['pipeline']}_{row['condition']}": row["qa_drop_per_tor"] for row in pc_rows if row["condition"] != "clean"}
    non_overlap_size = {f"{row['pipeline']}_{row['condition']}": row["num_qa_non_overlap"] for row in non_overlap_rows if row["condition"] != "clean"}

    api_failed_count = sum(1 for row in instances if row["answer_status"] == "api_failed")
    parser_empty_count = sum(1 for row in instances if row["answer_status"] == "parser_empty")
    warnings = []
    for pipeline, value in clean_answer_missing.items():
        if float(value) > 0.05:
            warnings.append(f"{pipeline} clean AnswerMissing > 0.05: {value}")
    if parser_empty_count != 4:
        warnings.append(f"parser_empty_count expected 4, got {parser_empty_count}")
    if api_failed_count != 0:
        warnings.append(f"api_failed_count expected 0, got {api_failed_count}")
    if not parser_metric_sources:
        warnings.append("Parser metrics beyond TOR were not found; non-TOR correlations are NaN.")
    ready = "YES" if len(instances) == 304 and api_failed_count == 0 and parser_empty_count == 4 else "NO"
    summary = {
        "total_answer_items": len(instances),
        "api_failed_count": api_failed_count,
        "parser_empty_count": parser_empty_count,
        "clean_answer_missing_by_pipeline": clean_answer_missing,
        "clean_qa_em_by_pipeline": clean_qa_em,
        "qa_drop_by_pipeline_condition": qa_drop,
        "qa_drop_per_tor_by_pipeline_condition": qa_drop_per_tor,
        "non_overlap_subset_size_by_pipeline_condition": non_overlap_size,
        "correlation_available": corr_available,
        "warnings": warnings,
        "ready_for_table_generation": ready,
    }
    summary_path = pm.exp1_log_root / ("qa_evaluation_debug_summary.md" if debug else "qa_evaluation_summary.md")
    _write_summary(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    summary["metrics_dir"] = str(metrics_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate debug QA robustness metrics.")
    parser.add_argument("--config", default="experiment_add/configs/exp1_qa.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.config, debug=args.debug)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ready_for_table_generation"] == "YES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
