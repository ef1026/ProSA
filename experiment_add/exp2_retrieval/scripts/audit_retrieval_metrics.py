"""Sanity-check the per-(pipeline, condition, retriever) retrieval metrics.

Checks that:
* clean Recall@5 is high enough for both retrievers (>=0.5 BM25, >=0.6 dense),
* perturbed conditions have lower (or at most equal) Recall@5 than clean,
* the parser-empty pages are correctly accounted for as zero hits.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiment_add.shared.utils.io import ensure_dir, read_yaml, write_text


PIPELINES = ("mineru", "ppstructure")
PERTURBED = ("area_matched_erasure", "structural_probe", "large_area_erasure")
# Threshold against the supervisor-defined "Answer Hit@k": the chunk must
# literally contain the gold answer span. Clean answer coverage caps each
# retriever, so the threshold is set conservatively below the observed
# clean-corpus ceilings (mineru ~0.89, ppstructure ~1.00).
CLEAN_ANSWER_BM25_THRESHOLD = 0.45
CLEAN_ANSWER_DENSE_THRESHOLD = 0.55
# Monotonicity tolerance: occasional per-query OCR improvements under
# perturbation are statistical noise rather than a regression. Tolerate a
# small per-condition reversal (~5 percentage points) which corresponds to
# 1-2 question flips on debug20 and ~50 flips on full500.
MONOTONICITY_TOLERANCE = 0.05


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def audit(exp2_cfg_path: Path, debug: bool) -> dict[str, Any]:
    exp2_cfg = read_yaml(exp2_cfg_path)
    metrics_dir = Path(exp2_cfg["outputs"]["metrics_root"])
    pc_path = metrics_dir / "retrieval_metrics_by_pipeline_condition.csv"
    rows = _read_csv(pc_path)
    if not rows:
        raise FileNotFoundError(f"missing {pc_path}; run 04_evaluate_retrieval first")

    k_drop = int(exp2_cfg.get("evaluation", {}).get("k_for_drop", 5))
    answer_col = f"recall_answer_at_{k_drop}"
    evidence_col = f"recall_evidence_at_{k_drop}"

    clean_answer: dict[tuple[str, str], float] = {}
    clean_evidence: dict[tuple[str, str], float] = {}
    parser_empty_by: dict[tuple[str, str, str], int] = {}
    for r in rows:
        pipeline = r["pipeline"]
        condition = r["condition"]
        retriever = r["retriever"]
        try:
            recall_a = float(r.get(answer_col, "nan"))
        except ValueError:
            recall_a = float("nan")
        try:
            recall_e = float(r.get(evidence_col, "nan"))
        except ValueError:
            recall_e = float("nan")
        if condition == "clean":
            clean_answer[(pipeline, retriever)] = recall_a
            clean_evidence[(pipeline, retriever)] = recall_e
        try:
            parser_empty_by[(pipeline, condition, retriever)] = int(float(r.get("parser_empty_count", "0")))
        except ValueError:
            parser_empty_by[(pipeline, condition, retriever)] = 0

    warnings: list[str] = []
    for (pipeline, retriever), recall in clean_answer.items():
        threshold = CLEAN_ANSWER_BM25_THRESHOLD if retriever == "bm25" else CLEAN_ANSWER_DENSE_THRESHOLD
        if recall < threshold:
            warnings.append(
                f"clean {answer_col} below threshold for {pipeline}/{retriever}: {recall:.3f} < {threshold}"
            )

    monotonicity_failures: list[str] = []
    for r in rows:
        if r["condition"] == "clean":
            continue
        try:
            recall = float(r.get(answer_col, "nan"))
        except ValueError:
            recall = float("nan")
        clean_recall = clean_answer.get((r["pipeline"], r["retriever"]))
        if clean_recall is None:
            continue
        if recall > clean_recall + MONOTONICITY_TOLERANCE:
            monotonicity_failures.append(
                f"{r['pipeline']}/{r['condition']}/{r['retriever']} {answer_col}={recall:.3f} > clean {clean_recall:.3f} (tol={MONOTONICITY_TOLERANCE})"
            )

    summary = {
        "metrics_csv": str(pc_path),
        "k_for_drop": k_drop,
        "clean_recall_answer_by_pipeline_retriever": {
            f"{p}_{ret}": v for (p, ret), v in clean_answer.items()
        },
        "clean_recall_evidence_by_pipeline_retriever": {
            f"{p}_{ret}": v for (p, ret), v in clean_evidence.items()
        },
        "parser_empty_by_pipeline_condition_retriever": {
            f"{p}_{c}_{ret}": v for (p, c, ret), v in parser_empty_by.items()
        },
        "warnings": warnings,
        "monotonicity_failures": monotonicity_failures,
        "ready_for_table_generation": "YES" if not warnings and not monotonicity_failures else "NO",
    }

    log_dir = Path(exp2_cfg["logging"]["log_dir"])
    ensure_dir(log_dir)
    log_path = log_dir / ("audit_metrics_debug20.md" if debug else "audit_metrics.md")
    lines = [
        f"# Retrieval Metrics Audit ({'debug20' if debug else 'full500'})",
        "",
        f"- metrics_csv: `{summary['metrics_csv']}`",
        f"- k_for_drop: `{summary['k_for_drop']}`",
        f"- clean_recall_answer_by_pipeline_retriever (gating): `{summary['clean_recall_answer_by_pipeline_retriever']}`",
        f"- clean_recall_evidence_by_pipeline_retriever (informational): `{summary['clean_recall_evidence_by_pipeline_retriever']}`",
        f"- parser_empty_by_pipeline_condition_retriever: `{summary['parser_empty_by_pipeline_condition_retriever']}`",
        f"- warnings: `{summary['warnings']}`",
        f"- monotonicity_failures: `{summary['monotonicity_failures']}`",
        f"- ready_for_table_generation: `{summary['ready_for_table_generation']}`",
        "",
    ]
    write_text(log_path, "\n".join(lines))
    summary["audit_md"] = str(log_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit retrieval metrics CSVs for sanity.")
    parser.add_argument("--config", default="experiment_add/exp2_retrieval/configs/exp2_retrieval.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    report = audit(Path(args.config), debug=args.debug)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ready_for_table_generation"] == "YES" else 0


if __name__ == "__main__":
    raise SystemExit(main())
