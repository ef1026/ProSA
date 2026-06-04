"""Aggregate retrieval runs into evaluation tables.

Outputs (all CSV, plus a small markdown summary for fast triage):

* ``retrieval_metrics_by_pipeline_condition.csv``
* ``retrieval_metrics_by_page.csv``
* ``retrieval_metrics_by_question_type.csv``
* ``retrieval_metrics_non_overlap_subset.csv``
* ``retrieval_correlations.csv``
* ``retrieval_failure_decomposition.csv``

The metric definitions match the plan: page-internal Recall@k, MRR@k, and
EvidenceHit@k / AnswerHit@k. Drops are computed against the same pipeline's
clean condition; per-TOR efficiency divides drop by mean TOR.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiment_add.exp2_retrieval.metrics.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from experiment_add.exp2_retrieval.metrics.retrieval_non_overlap_analysis import (
    select_non_overlap_qa_ids,
)
from experiment_add.shared.metrics.aggregation import safe_divide
from experiment_add.shared.metrics.correlation_utils import spearman
from experiment_add.shared.metrics.parser_metrics_loader import load_parser_metrics
from experiment_add.shared.utils.io import ensure_dir, read_jsonl, read_yaml, write_text
from experiment_add.shared.utils.path_manager import PathManager


PIPELINES = ("mineru", "ppstructure")
CONDITIONS = ("clean", "area_matched_erasure", "structural_probe", "large_area_erasure")
PERTURBED_CONDITIONS = ("area_matched_erasure", "structural_probe", "large_area_erasure")
KS_DEFAULT = (1, 3, 5, 10)


def _resolve_base_config(exp2_cfg_path: Path, exp2_cfg: dict[str, Any]) -> Path:
    base = exp2_cfg.get("inputs", {}).get("base_config")
    if base:
        candidate = Path(base)
        if not candidate.is_absolute():
            for ancestor in [exp2_cfg_path.parent, *exp2_cfg_path.parents]:
                guess = ancestor / candidate
                if guess.exists():
                    return guess.resolve()
        if candidate.exists():
            return candidate.resolve()
    for ancestor in exp2_cfg_path.parents:
        guess = ancestor / "experiment_add" / "configs" / "base.yaml"
        if guess.exists():
            return guess.resolve()
    raise FileNotFoundError("Could not locate experiment_add/configs/base.yaml")


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


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _flatten_runs(
    runs_root: Path,
    qa_by_id: dict[str, dict[str, Any]],
    metadata: dict[tuple[str, str], dict[str, Any]],
    mean_tor: dict[str, float],
    retrievers: list[str],
    ks: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Compute per-instance metrics for every (pipeline, condition, retriever, qa)."""
    rows: list[dict[str, Any]] = []
    for pipeline in PIPELINES:
        for condition in CONDITIONS:
            for retriever in retrievers:
                run_path = runs_root / f"{pipeline}_{condition}_{retriever}" / "run.jsonl"
                run_rows = read_jsonl(run_path)
                for r in run_rows:
                    qa_id = str(r.get("qa_id", ""))
                    qa = qa_by_id.get(qa_id, {})
                    hits = r.get("hits") or []
                    rel_evidence = [int(h.get("evidence_hit", 0)) for h in hits]
                    rel_answer = [int(h.get("answer_hit", 0)) for h in hits]

                    row = {
                        "pipeline": pipeline,
                        "condition": condition,
                        "retriever": retriever,
                        "page_id": str(r.get("page_id", "")),
                        "qa_id": qa_id,
                        "answer_type": qa.get("answer_type", "unknown"),
                        "parser_status": str(r.get("parser_status", "")),
                        "num_chunks_for_page": int(r.get("num_chunks_for_page", 0)),
                        "num_hits": len(hits),
                        "TOR": metadata.get((str(r.get("page_id", "")), condition), {}).get(
                            "TOR", 0.0 if condition == "clean" else mean_tor.get(condition, "")
                        ),
                        "rr_evidence_at_10": reciprocal_rank(rel_evidence, cap_k=10),
                        "rr_answer_at_10": reciprocal_rank(rel_answer, cap_k=10),
                        "ndcg_evidence_at_10": ndcg_at_k(rel_evidence, 10),
                    }
                    for k in ks:
                        row[f"recall_evidence_at_{k}"] = recall_at_k(rel_evidence, k)
                        row[f"recall_answer_at_{k}"] = recall_at_k(rel_answer, k)
                    rows.append(row)
    return rows


def _by_pipeline_condition(
    instances: list[dict[str, Any]],
    mean_tor: dict[str, float],
    ks: tuple[int, ...],
    k_drop: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in instances:
        grouped[(r["pipeline"], r["condition"], r["retriever"])].append(r)

    base_clean: dict[tuple[str, str], float] = {}
    for (p, c, ret), rows in grouped.items():
        if c == "clean":
            base_clean[(p, ret)] = _mean([float(x[f"recall_evidence_at_{k_drop}"]) for x in rows])

    out: list[dict[str, Any]] = []
    for (p, c, ret), rows in sorted(grouped.items()):
        clean_val = base_clean.get((p, ret), 0.0)
        cur_recall = _mean([float(x[f"recall_evidence_at_{k_drop}"]) for x in rows])
        recall_drop = 0.0 if c == "clean" else clean_val - cur_recall
        tor = 0.0 if c == "clean" else mean_tor.get(c, float("nan"))

        record: dict[str, Any] = {
            "pipeline": p,
            "condition": c,
            "retriever": ret,
            "num_queries": len(rows),
            "num_pages": len({x["page_id"] for x in rows}),
            "parser_empty_count": sum(1 for x in rows if x["parser_status"] == "empty"),
            "mean_num_chunks_for_page": _mean([float(x["num_chunks_for_page"]) for x in rows]),
            "mrr_evidence_at_10": _mean([float(x["rr_evidence_at_10"]) for x in rows]),
            "mrr_answer_at_10": _mean([float(x["rr_answer_at_10"]) for x in rows]),
            "ndcg_evidence_at_10": _mean([float(x["ndcg_evidence_at_10"]) for x in rows]),
            "mean_tor": "" if c == "clean" else tor,
        }
        for k in ks:
            record[f"recall_evidence_at_{k}"] = _mean([float(x[f"recall_evidence_at_{k}"]) for x in rows])
            record[f"recall_answer_at_{k}"] = _mean([float(x[f"recall_answer_at_{k}"]) for x in rows])
        record[f"recall_evidence_drop_at_{k_drop}"] = recall_drop
        record[f"recall_evidence_drop_per_tor_at_{k_drop}"] = "" if c == "clean" else safe_divide(recall_drop, tor)
        out.append(record)
    return out


def _by_page(instances: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in instances:
        grouped[(r["pipeline"], r["condition"], r["retriever"], r["page_id"])].append(r)
    out: list[dict[str, Any]] = []
    for (p, c, ret, pid), rows in sorted(grouped.items()):
        out.append(
            {
                "pipeline": p,
                "condition": c,
                "retriever": ret,
                "page_id": pid,
                "num_qa": len(rows),
                "parser_status": rows[0]["parser_status"],
                "num_chunks_for_page": rows[0]["num_chunks_for_page"],
                f"recall_evidence_at_{k}": _mean([float(x[f"recall_evidence_at_{k}"]) for x in rows]),
                f"recall_answer_at_{k}": _mean([float(x[f"recall_answer_at_{k}"]) for x in rows]),
                "mrr_evidence_at_10": _mean([float(x["rr_evidence_at_10"]) for x in rows]),
                "TOR": rows[0]["TOR"],
            }
        )
    return out


def _by_question_type(instances: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in instances:
        grouped[(r["pipeline"], r["condition"], r["retriever"], str(r.get("answer_type", "unknown")))].append(r)
    out: list[dict[str, Any]] = []
    for (p, c, ret, at), rows in sorted(grouped.items()):
        out.append(
            {
                "pipeline": p,
                "condition": c,
                "retriever": ret,
                "answer_type": at,
                "num_qa": len(rows),
                f"recall_evidence_at_{k}": _mean([float(x[f"recall_evidence_at_{k}"]) for x in rows]),
                f"recall_answer_at_{k}": _mean([float(x[f"recall_answer_at_{k}"]) for x in rows]),
                "mrr_evidence_at_10": _mean([float(x["rr_evidence_at_10"]) for x in rows]),
            }
        )
    return out


def _non_overlap_subset(
    instances: list[dict[str, Any]],
    qa_by_id: dict[str, dict[str, Any]],
    metadata: dict[tuple[str, str], dict[str, Any]],
    mean_tor: dict[str, float],
    k: int,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {
        (r["pipeline"], r["condition"], r["retriever"], r["qa_id"]): r for r in instances
    }
    out: list[dict[str, Any]] = []
    for pipeline in PIPELINES:
        for retriever in {r["retriever"] for r in instances}:
            clean_rows = [
                r for r in instances
                if r["pipeline"] == pipeline and r["condition"] == "clean" and r["retriever"] == retriever
            ]
            clean_recall = _mean([float(x[f"recall_evidence_at_{k}"]) for x in clean_rows])
            out.append(
                {
                    "pipeline": pipeline,
                    "condition": "clean",
                    "retriever": retriever,
                    "num_qa_non_overlap": len(clean_rows),
                    f"recall_evidence_at_{k}": clean_recall,
                    f"recall_evidence_drop_at_{k}": 0.0,
                    "mean_tor": "",
                    f"recall_evidence_drop_per_tor_at_{k}": "",
                }
            )
            for cond in PERTURBED_CONDITIONS:
                qa_ids = select_non_overlap_qa_ids(instances, qa_by_id, metadata, pipeline, cond)
                cond_rows = [
                    r for r in instances
                    if r["pipeline"] == pipeline and r["condition"] == cond and r["retriever"] == retriever and r["qa_id"] in qa_ids
                ]
                clean_subset = [
                    by_key.get((pipeline, "clean", retriever, qid))
                    for qid in qa_ids
                    if (pipeline, "clean", retriever, qid) in by_key
                ]
                clean_subset_recall = _mean([float(x[f"recall_evidence_at_{k}"]) for x in clean_subset])
                cur_recall = _mean([float(x[f"recall_evidence_at_{k}"]) for x in cond_rows])
                drop = clean_subset_recall - cur_recall if cond_rows else 0.0
                tor = mean_tor.get(cond, float("nan"))
                out.append(
                    {
                        "pipeline": pipeline,
                        "condition": cond,
                        "retriever": retriever,
                        "num_qa_non_overlap": len(cond_rows),
                        f"recall_evidence_at_{k}": cur_recall,
                        f"recall_evidence_drop_at_{k}": drop,
                        "mean_tor": tor,
                        f"recall_evidence_drop_per_tor_at_{k}": safe_divide(drop, tor),
                    }
                )
    return out


def _correlations(
    pc_rows: list[dict[str, Any]],
    parser_metric_rows: list[dict[str, Any]],
    k: int,
) -> tuple[list[dict[str, Any]], str]:
    out: list[dict[str, Any]] = []
    for retriever in sorted({r["retriever"] for r in pc_rows}):
        target_col = f"recall_evidence_drop_at_{k}"
        pairs = [
            (r.get("mean_tor"), r.get(target_col))
            for r in pc_rows
            if r["retriever"] == retriever and r["condition"] != "clean"
        ]
        out.append(
            {
                "x_metric": "TOR",
                "y_metric": target_col,
                "retriever": retriever,
                "spearman": spearman(pairs),
                "n": len(pairs),
                "source": "perturb_metadata",
            }
        )
    available = "PARTIAL"
    for retriever in sorted({r["retriever"] for r in pc_rows}):
        for metric in ("B_SLR", "delta_CER", "CER_matched_mean"):
            pm_pairs = []
            for r in parser_metric_rows:
                y = r.get(f"recall_evidence_drop_at_{k}_{retriever}")
                if y is not None:
                    pm_pairs.append((r.get(metric), y))
            out.append(
                {
                    "x_metric": metric,
                    "y_metric": f"recall_evidence_drop_at_{k}",
                    "retriever": retriever,
                    "spearman": spearman(pm_pairs),
                    "n": len(pm_pairs),
                    "source": "parser_metrics_best_effort",
                }
            )
    if not parser_metric_rows:
        available = "PARTIAL"
    return out, available


def _failure_decomposition(instances: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in instances:
        evidence_at_k = float(r.get(f"recall_evidence_at_{k}", 0.0))
        answer_at_k = float(r.get(f"recall_answer_at_{k}", 0.0))
        if r["parser_status"] != "success":
            ftype = "Parser Empty / Invalid"
        elif evidence_at_k > 0:
            ftype = "Evidence Retrieved"
        elif answer_at_k > 0:
            ftype = "Answer Present, Evidence Missed"
        else:
            ftype = "Both Missed"
        out.append(
            {
                "pipeline": r["pipeline"],
                "condition": r["condition"],
                "retriever": r["retriever"],
                "page_id": r["page_id"],
                "qa_id": r["qa_id"],
                "answer_type": r.get("answer_type", "unknown"),
                "parser_status": r["parser_status"],
                "num_hits": r["num_hits"],
                f"recall_evidence_at_{k}": evidence_at_k,
                f"recall_answer_at_{k}": answer_at_k,
                "failure_type": ftype,
            }
        )
    return out


def _load_metadata(pm: PathManager) -> dict[tuple[str, str], dict[str, Any]]:
    primary = pm.project_root / "experiment_add/outputs/shared/perturbed_pages/merged_perturb_metadata.jsonl"
    path = primary if primary.exists() else pm.merged_perturb_metadata_path
    return {(str(row.get("page_id", "")), str(row.get("condition", ""))): row for row in read_jsonl(path)}


def _load_mean_tor(pm: PathManager, metadata: dict[tuple[str, str], dict[str, Any]]) -> dict[str, float]:
    means = {c: 0.0 for c in CONDITIONS}
    summary_path = pm.project_root / "experiment_add/outputs/shared/perturbed_pages/perturb_summary.csv"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                cond = str(row.get("condition", ""))
                if cond in means:
                    try:
                        means[cond] = float(row.get("mean_TOR", 0.0))
                    except (TypeError, ValueError):
                        pass
    for cond in PERTURBED_CONDITIONS:
        if means.get(cond, 0.0) == 0.0:
            tors = [float(r.get("TOR", 0.0)) for (pid, c), r in metadata.items() if c == cond]
            means[cond] = sum(tors) / len(tors) if tors else 0.0
    return means


def evaluate(exp2_cfg_path: Path, debug: bool) -> dict[str, Any]:
    exp2_cfg = read_yaml(exp2_cfg_path)
    base_cfg = _resolve_base_config(exp2_cfg_path, exp2_cfg)
    pm = PathManager(base_cfg, create_dirs=True)

    runs_root = Path(exp2_cfg["outputs"]["retrieval_runs_root"])
    metrics_dir = Path(exp2_cfg["outputs"]["metrics_root"])
    ensure_dir(metrics_dir)

    qa_rows = read_jsonl(pm.qa_pairs_shared_path)
    qa_by_id = {str(r.get("qa_id", "")): r for r in qa_rows}
    metadata = _load_metadata(pm)
    mean_tor = _load_mean_tor(pm, metadata)

    retrievers = list(exp2_cfg.get("retrieval", {}).get("retrievers", ["bm25", "dense"]))
    ks = tuple(int(k) for k in exp2_cfg.get("retrieval", {}).get("ks", KS_DEFAULT))
    k_drop = int(exp2_cfg.get("evaluation", {}).get("k_for_drop", 5))

    instances = _flatten_runs(runs_root, qa_by_id, metadata, mean_tor, retrievers, ks)
    pc_rows = _by_pipeline_condition(instances, mean_tor, ks, k_drop)
    page_rows = _by_page(instances, k_drop)
    qt_rows = _by_question_type(instances, k_drop)
    non_overlap_rows = _non_overlap_subset(instances, qa_by_id, metadata, mean_tor, k_drop)
    parser_metric_rows, parser_metric_sources = load_parser_metrics(pm.project_root)
    corr_rows, corr_available = _correlations(pc_rows, parser_metric_rows, k_drop)
    failure_rows = _failure_decomposition(instances, k_drop)

    pc_fields = [
        "pipeline", "condition", "retriever",
        "num_queries", "num_pages", "parser_empty_count", "mean_num_chunks_for_page",
    ] + [f"recall_evidence_at_{k}" for k in ks] + [f"recall_answer_at_{k}" for k in ks] + [
        "mrr_evidence_at_10", "mrr_answer_at_10", "ndcg_evidence_at_10", "mean_tor",
        f"recall_evidence_drop_at_{k_drop}", f"recall_evidence_drop_per_tor_at_{k_drop}",
    ]
    _write_csv(metrics_dir / "retrieval_metrics_by_pipeline_condition.csv", pc_rows, pc_fields)
    _write_csv(
        metrics_dir / "retrieval_metrics_by_page.csv",
        page_rows,
        ["pipeline", "condition", "retriever", "page_id", "num_qa", "parser_status", "num_chunks_for_page",
         f"recall_evidence_at_{k_drop}", f"recall_answer_at_{k_drop}", "mrr_evidence_at_10", "TOR"],
    )
    _write_csv(
        metrics_dir / "retrieval_metrics_by_question_type.csv",
        qt_rows,
        ["pipeline", "condition", "retriever", "answer_type", "num_qa",
         f"recall_evidence_at_{k_drop}", f"recall_answer_at_{k_drop}", "mrr_evidence_at_10"],
    )
    _write_csv(
        metrics_dir / "retrieval_metrics_non_overlap_subset.csv",
        non_overlap_rows,
        ["pipeline", "condition", "retriever", "num_qa_non_overlap",
         f"recall_evidence_at_{k_drop}", f"recall_evidence_drop_at_{k_drop}",
         "mean_tor", f"recall_evidence_drop_per_tor_at_{k_drop}"],
    )
    _write_csv(
        metrics_dir / "retrieval_correlations.csv",
        corr_rows,
        ["x_metric", "y_metric", "retriever", "spearman", "n", "source"],
    )
    _write_csv(
        metrics_dir / "retrieval_failure_decomposition.csv",
        failure_rows,
        ["pipeline", "condition", "retriever", "page_id", "qa_id", "answer_type",
         "parser_status", "num_hits",
         f"recall_evidence_at_{k_drop}", f"recall_answer_at_{k_drop}", "failure_type"],
    )

    by_retriever_drop = {
        f"{r['pipeline']}_{r['condition']}_{r['retriever']}": r[f"recall_evidence_drop_at_{k_drop}"]
        for r in pc_rows if r["condition"] != "clean"
    }
    clean_evidence = {
        f"{r['pipeline']}_{r['retriever']}": r[f"recall_evidence_at_{k_drop}"]
        for r in pc_rows if r["condition"] == "clean"
    }
    clean_answer = {
        f"{r['pipeline']}_{r['retriever']}": r[f"recall_answer_at_{k_drop}"]
        for r in pc_rows if r["condition"] == "clean"
    }
    # Gating uses the supervisor-named "Answer Hit@k". Evidence-text recall is
    # informational: with block-aware char-window chunking it caps below 1 even
    # on clean because evidence spans frequently cross chunk boundaries.
    warnings = []
    for key, val in clean_answer.items():
        if val < 0.45:
            warnings.append(f"clean Answer-Hit@{k_drop} unexpectedly low for {key}: {val:.3f}")
    if not parser_metric_rows:
        warnings.append("Parser metrics beyond TOR were not found; parser-metric correlations are NaN.")

    summary = {
        "total_instances": len(instances),
        "retrievers": retrievers,
        "ks": list(ks),
        "k_for_drop": k_drop,
        "clean_recall_answer_by_pipeline_retriever": clean_answer,
        "clean_recall_evidence_by_pipeline_retriever": clean_evidence,
        "drop_recall_evidence_by_pipeline_condition_retriever": by_retriever_drop,
        "correlation_available": corr_available,
        "warnings": warnings,
        "metrics_dir": str(metrics_dir),
        "parser_metric_sources": parser_metric_sources,
    }
    summary_path = pm.exp2_log_root / ("retrieval_evaluation_debug20_summary.md" if debug else "retrieval_evaluation_summary.md")
    write_text(summary_path, _render_summary(summary))
    summary["summary_path"] = str(summary_path)
    return summary


def _render_summary(result: dict[str, Any]) -> str:
    lines = [
        "# Retrieval Evaluation Summary",
        "",
        f"- total_instances: `{result['total_instances']}`",
        f"- retrievers: `{result['retrievers']}`",
        f"- ks: `{result['ks']}`",
        f"- k_for_drop: `{result['k_for_drop']}`",
        f"- clean_recall_answer_by_pipeline_retriever (gating): `{result['clean_recall_answer_by_pipeline_retriever']}`",
        f"- clean_recall_evidence_by_pipeline_retriever (informational): `{result['clean_recall_evidence_by_pipeline_retriever']}`",
        f"- drop_recall_evidence_by_pipeline_condition_retriever: `{result['drop_recall_evidence_by_pipeline_condition_retriever']}`",
        f"- correlation_available: `{result['correlation_available']}`",
        f"- warnings: `{result['warnings']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval runs into CSV tables.")
    parser.add_argument("--config", default="experiment_add/exp2_retrieval/configs/exp2_retrieval.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    result = evaluate(Path(args.config), debug=args.debug)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not result["warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
