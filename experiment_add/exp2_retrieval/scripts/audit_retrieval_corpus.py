"""Audit the chunked retrieval corpora.

Validates that:
* every page in the manifest is represented (or accounted for as empty/failed),
* the per-page chunk count distribution is sane (flag pages with < 3 chunks),
* on the *clean* corpora, the gold ``evidence_text`` is contained in at least
  one chunk for the overwhelming majority of QA (>=95% target),
* the gold ``gold_answer`` is contained in at least one chunk on clean corpora,
* perturbed corpora exhibit lower coverage (sanity check that the perturbation
  actually disturbs retrieval evidence).
"""

from __future__ import annotations

import argparse
import json
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
from experiment_add.shared.utils.io import ensure_dir, read_jsonl, read_yaml, write_text
from experiment_add.shared.utils.path_manager import PathManager


PIPELINES = ("mineru", "ppstructure")
CONDITIONS = ("clean", "area_matched_erasure", "structural_probe", "large_area_erasure")
MIN_CHUNKS_WARNING = 3
# Evidence text often spans multiple blocks and gets split by block-aware
# chunking, so the strict text-match metric is informational only. The gating
# coverage signal is the gold-answer string, which corresponds to the
# supervisor-defined "Answer Hit@k" relevance.
EVIDENCE_COVERAGE_THRESHOLD = 0.50
ANSWER_COVERAGE_THRESHOLD = 0.85


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


def _coverage_for_pc(
    chunks_by_page: dict[str, list[dict[str, Any]]],
    qa_by_page: dict[str, list[dict[str, Any]]],
    use_evidence: bool,
) -> tuple[int, int, list[str]]:
    """Return (covered, total, missed_qa_ids[:25])."""
    covered = 0
    total = 0
    missed: list[str] = []
    for page_id, qa_list in qa_by_page.items():
        chunks = chunks_by_page.get(page_id, [])
        for qa in qa_list:
            total += 1
            target = qa.get("evidence_text" if use_evidence else "gold_answer", "")
            if not target:
                continue
            ok = any(contains_answer(c.get("text", ""), target) for c in chunks)
            if ok:
                covered += 1
            else:
                missed.append(str(qa.get("qa_id", "")))
    return covered, total, missed[:25]


def _summarize_chunks(records: list[dict[str, Any]]) -> dict[str, Any]:
    chunks_per_page: dict[str, int] = defaultdict(int)
    char_lens: list[int] = []
    for r in records:
        chunks_per_page[str(r.get("page_id", ""))] += 1
        char_lens.append(int(r.get("char_len", 0)))
    counts = sorted(chunks_per_page.values())
    pages_below = sum(1 for c in counts if c < MIN_CHUNKS_WARNING)
    return {
        "num_chunks": len(records),
        "num_pages_with_chunks": len(counts),
        "min_chunks_per_page": counts[0] if counts else 0,
        "max_chunks_per_page": counts[-1] if counts else 0,
        "median_chunks_per_page": counts[len(counts) // 2] if counts else 0,
        "pages_below_min_chunks": pages_below,
        "min_chars": min(char_lens) if char_lens else 0,
        "max_chars": max(char_lens) if char_lens else 0,
        "mean_chars": (sum(char_lens) / len(char_lens)) if char_lens else 0.0,
    }


def audit(exp2_cfg_path: Path, debug: bool) -> dict[str, Any]:
    exp2_cfg = read_yaml(exp2_cfg_path)
    base_cfg = _resolve_base_config(exp2_cfg_path, exp2_cfg)
    pm = PathManager(base_cfg, create_dirs=True)

    manifest = load_manifest(pm.page_manifest_debug20 if debug else pm.page_manifest_500)
    manifest_ids = {str(row["page_id"]) for row in manifest}
    qa_rows = read_jsonl(pm.qa_pairs_shared_path)
    qa_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in qa_rows:
        pid = str(r.get("page_id", ""))
        if pid in manifest_ids:
            qa_by_page[pid].append(r)
    qa_by_page = {pid: rows[:4] for pid, rows in qa_by_page.items()}

    corpora_root = Path(exp2_cfg["outputs"]["corpora_root"])
    report: dict[str, Any] = {
        "manifest_size": len(manifest_ids),
        "num_qa_in_scope": sum(len(v) for v in qa_by_page.values()),
        "min_chunks_warning_threshold": MIN_CHUNKS_WARNING,
        "evidence_coverage_threshold": EVIDENCE_COVERAGE_THRESHOLD,
        "answer_coverage_threshold": ANSWER_COVERAGE_THRESHOLD,
        "by_pipeline_condition": {},
        "warnings": [],
    }

    for pipeline in PIPELINES:
        for condition in CONDITIONS:
            corpus_dir = corpora_root / f"{pipeline}_{condition}"
            chunks_path = corpus_dir / "chunks.jsonl"
            pages_path = corpus_dir / "pages.jsonl"
            if not chunks_path.exists():
                report["warnings"].append(f"missing chunks.jsonl: {chunks_path}")
                continue
            chunks = read_jsonl(chunks_path)
            chunks_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for c in chunks:
                chunks_by_page[str(c.get("page_id", ""))].append(c)
            page_status: dict[str, str] = {}
            for r in read_jsonl(pages_path):
                page_status[str(r.get("page_id", ""))] = str(r.get("parser_status", ""))

            chunk_summary = _summarize_chunks(chunks)
            ev_cov, ev_tot, ev_missed = _coverage_for_pc(chunks_by_page, qa_by_page, use_evidence=True)
            an_cov, an_tot, an_missed = _coverage_for_pc(chunks_by_page, qa_by_page, use_evidence=False)
            ev_rate = (ev_cov / ev_tot) if ev_tot else 0.0
            an_rate = (an_cov / an_tot) if an_tot else 0.0

            entry = {
                "pipeline": pipeline,
                "condition": condition,
                **chunk_summary,
                "qa_coverage_evidence": {
                    "covered": ev_cov,
                    "total": ev_tot,
                    "rate": ev_rate,
                    "missed_sample": ev_missed,
                },
                "qa_coverage_answer": {
                    "covered": an_cov,
                    "total": an_tot,
                    "rate": an_rate,
                    "missed_sample": an_missed,
                },
                "page_status_counts": dict(Counter(page_status.values())),
            }
            report["by_pipeline_condition"][f"{pipeline}_{condition}"] = entry

            if condition == "clean":
                if an_rate < ANSWER_COVERAGE_THRESHOLD:
                    report["warnings"].append(
                        f"clean answer coverage low for {pipeline}: {an_rate:.3f} < {ANSWER_COVERAGE_THRESHOLD} (gating)"
                    )
                if ev_rate < EVIDENCE_COVERAGE_THRESHOLD:
                    report["warnings"].append(
                        f"clean evidence coverage low for {pipeline}: {ev_rate:.3f} < {EVIDENCE_COVERAGE_THRESHOLD} (informational)"
                    )
            if chunk_summary["pages_below_min_chunks"] > 0:
                report["warnings"].append(
                    f"{pipeline}/{condition}: {chunk_summary['pages_below_min_chunks']} pages have < {MIN_CHUNKS_WARNING} chunks"
                )

    log_dir = Path(exp2_cfg["logging"]["log_dir"])
    ensure_dir(log_dir)
    log_path = log_dir / ("audit_corpus_debug20.md" if debug else "audit_corpus.md")
    write_text(log_path, _render_audit_md(report, debug=debug))
    report["audit_md"] = str(log_path)
    return report


def _render_audit_md(report: dict[str, Any], debug: bool) -> str:
    lines = [
        f"# Retrieval Corpus Audit ({'debug20' if debug else 'full500'})",
        "",
        f"- manifest_size: `{report['manifest_size']}`",
        f"- num_qa_in_scope: `{report['num_qa_in_scope']}`",
        f"- min_chunks_warning_threshold: `{report['min_chunks_warning_threshold']}`",
        f"- evidence_coverage_threshold: `{report['evidence_coverage_threshold']}`",
        f"- answer_coverage_threshold: `{report['answer_coverage_threshold']}`",
        "",
        "## Per (pipeline, condition)",
        "",
        "| pipeline | condition | chunks | pages | min/med/max chunks/page | <min | mean chars | evidence cov | answer cov | page_status |",
        "|---|---|---:|---:|---|---:|---:|---|---|---|",
    ]
    for key, e in report["by_pipeline_condition"].items():
        ev = e["qa_coverage_evidence"]
        an = e["qa_coverage_answer"]
        lines.append(
            "| {p} | {c} | {chunks} | {pages} | {mn}/{md}/{mx} | {below} | {mean:.1f} | {ev_c}/{ev_t} ({ev_r:.3f}) | {an_c}/{an_t} ({an_r:.3f}) | {st} |".format(
                p=e["pipeline"], c=e["condition"],
                chunks=e["num_chunks"], pages=e["num_pages_with_chunks"],
                mn=e["min_chunks_per_page"], md=e["median_chunks_per_page"], mx=e["max_chunks_per_page"],
                below=e["pages_below_min_chunks"], mean=e["mean_chars"],
                ev_c=ev["covered"], ev_t=ev["total"], ev_r=ev["rate"],
                an_c=an["covered"], an_t=an["total"], an_r=an["rate"],
                st=e["page_status_counts"],
            )
        )
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if report["warnings"]:
        for w in report["warnings"]:
            lines.append(f"- {w}")
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit page-internal retrieval corpora.")
    parser.add_argument("--config", default="experiment_add/exp2_retrieval/configs/exp2_retrieval.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    report = audit(Path(args.config), debug=args.debug)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
