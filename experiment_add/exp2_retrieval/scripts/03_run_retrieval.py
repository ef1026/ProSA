"""Run BM25 + dense retrieval over the page-internal indexes.

For every (qa, pipeline, condition, retriever) tuple this writes one row to
``retrieval_runs/{pipeline}_{condition}_{retriever}/run.jsonl`` containing
the top-K hits with both ``evidence_hit`` and ``answer_hit`` flags so that
downstream evaluation never has to re-read the chunk corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import numpy as np

from experiment_add.exp2_retrieval.metrics.answer_hit import is_answer_chunk
from experiment_add.exp2_retrieval.metrics.evidence_recall import is_evidence_chunk
from experiment_add.exp2_retrieval.retrievers.bm25_retriever import (
    BM25PageIndex,
    query_page as bm25_query_page,
)
from experiment_add.exp2_retrieval.retrievers.dense_retriever import (
    DenseEncoder,
    encode_queries,
    topk_for_page,
)
from experiment_add.exp2_retrieval.retrievers.index_io import (
    index_dir_for,
    load_bm25_payload,
    load_dense_payload,
    read_manifest,
)
from experiment_add.shared.data.load_manifest import load_manifest
from experiment_add.shared.utils.io import (
    atomic_write_jsonl,
    ensure_dir,
    read_jsonl,
    read_yaml,
    write_text,
)
from experiment_add.shared.utils.path_manager import PathManager


PIPELINES = ("mineru", "ppstructure")
CONDITIONS = ("clean", "area_matched_erasure", "structural_probe", "large_area_erasure")
MAX_QA_PER_PAGE = 4


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


def _group_qa_by_page(qa_rows: list[dict[str, Any]], manifest_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qa_rows:
        page_id = str(row.get("page_id", ""))
        if page_id in manifest_ids:
            grouped[page_id].append(row)
    return {page_id: rows[:MAX_QA_PER_PAGE] for page_id, rows in grouped.items() if rows}


def _load_chunk_text_map(corpora_root: Path, pipeline: str, condition: str) -> tuple[dict[str, str], dict[str, str]]:
    chunks_path = corpora_root / f"{pipeline}_{condition}" / "chunks.jsonl"
    chunk_text: dict[str, str] = {}
    chunk_page: dict[str, str] = {}
    for r in read_jsonl(chunks_path):
        cid = str(r.get("chunk_id", ""))
        chunk_text[cid] = str(r.get("text", "") or "")
        chunk_page[cid] = str(r.get("page_id", ""))
    return chunk_text, chunk_page


def _load_page_status(corpora_root: Path, pipeline: str, condition: str) -> dict[str, str]:
    pages_path = corpora_root / f"{pipeline}_{condition}" / "pages.jsonl"
    out: dict[str, str] = {}
    for r in read_jsonl(pages_path):
        out[str(r.get("page_id", ""))] = str(r.get("parser_status", ""))
    return out


def _empty_run_row(qa: dict[str, Any], pipeline: str, condition: str, retriever: str, parser_status: str) -> dict[str, Any]:
    return {
        "qa_id": str(qa.get("qa_id", "")),
        "page_id": str(qa.get("page_id", "")),
        "pipeline": pipeline,
        "condition": condition,
        "retriever": retriever,
        "parser_status": parser_status or "missing",
        "num_chunks_for_page": 0,
        "hits": [],
    }


def _hit_row(rank: int, chunk_id: str, score: float, evidence_hit: int, answer_hit: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "score": float(score),
        "evidence_hit": int(evidence_hit),
        "answer_hit": int(answer_hit),
    }


def _run_bm25_for_pc(
    pipeline: str,
    condition: str,
    qa_by_page: dict[str, list[dict[str, Any]]],
    indexes_root: Path,
    chunk_text: dict[str, str],
    page_status: dict[str, str],
    k: int,
) -> list[dict[str, Any]]:
    index_dir = index_dir_for(indexes_root, pipeline, condition)
    payload = load_bm25_payload(index_dir)
    rows: list[dict[str, Any]] = []
    for page_id, qa_list in qa_by_page.items():
        bundle = payload.get(page_id)
        parser_status = page_status.get(page_id, "")
        if not bundle:
            for qa in qa_list:
                rows.append(_empty_run_row(qa, pipeline, condition, "bm25", parser_status))
            continue
        idx = BM25PageIndex(chunk_ids=bundle["chunk_ids"], bm25=bundle["bm25"])
        for qa in qa_list:
            top = bm25_query_page(idx, str(qa.get("question", "")), k=k)
            row = _empty_run_row(qa, pipeline, condition, "bm25", parser_status)
            row["num_chunks_for_page"] = len(bundle["chunk_ids"])
            for rank, (chunk_id, score) in enumerate(top, start=1):
                ctext = chunk_text.get(chunk_id, "")
                row["hits"].append(
                    _hit_row(
                        rank=rank,
                        chunk_id=chunk_id,
                        score=score,
                        evidence_hit=is_evidence_chunk(ctext, qa.get("evidence_text")),
                        answer_hit=is_answer_chunk(ctext, qa.get("gold_answer")),
                    )
                )
            rows.append(row)
    return rows


def _run_dense_for_pc(
    pipeline: str,
    condition: str,
    qa_by_page: dict[str, list[dict[str, Any]]],
    indexes_root: Path,
    chunk_text: dict[str, str],
    page_status: dict[str, str],
    encoder: DenseEncoder,
    k: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    index_dir = index_dir_for(indexes_root, pipeline, condition)
    manifest = read_manifest(index_dir)
    if not manifest.get("dense", {}).get("enabled"):
        return []
    payload, _meta = load_dense_payload(index_dir)
    page_ids_arr = np.array(payload.page_ids, dtype=object)
    chunk_ids = payload.chunk_ids
    embeddings = payload.embeddings

    flat_qa: list[tuple[str, dict[str, Any]]] = []
    for page_id, qa_list in qa_by_page.items():
        for qa in qa_list:
            flat_qa.append((page_id, qa))
    if not flat_qa:
        return []
    questions = [str(qa.get("question", "")) for _, qa in flat_qa]
    query_vecs = encode_queries(encoder, questions, batch_size=batch_size)

    rows: list[dict[str, Any]] = []
    for (page_id, qa), q_vec in zip(flat_qa, query_vecs):
        parser_status = page_status.get(page_id, "")
        mask = page_ids_arr == page_id
        num_chunks_for_page = int(mask.sum())
        row = _empty_run_row(qa, pipeline, condition, "dense", parser_status)
        row["num_chunks_for_page"] = num_chunks_for_page
        if num_chunks_for_page == 0:
            rows.append(row)
            continue
        top = topk_for_page(
            page_id=page_id,
            page_ids=page_ids_arr,
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            query_vec=q_vec,
            k=k,
        )
        for rank, (chunk_id, score) in enumerate(top, start=1):
            ctext = chunk_text.get(chunk_id, "")
            row["hits"].append(
                _hit_row(
                    rank=rank,
                    chunk_id=chunk_id,
                    score=score,
                    evidence_hit=is_evidence_chunk(ctext, qa.get("evidence_text")),
                    answer_hit=is_answer_chunk(ctext, qa.get("gold_answer")),
                )
            )
        rows.append(row)
    return rows


def run_all(exp2_cfg_path: Path, debug: bool) -> dict[str, Any]:
    exp2_cfg = read_yaml(exp2_cfg_path)
    base_cfg = _resolve_base_config(exp2_cfg_path, exp2_cfg)
    pm = PathManager(base_cfg, create_dirs=True)

    manifest = load_manifest(pm.page_manifest_debug20 if debug else pm.page_manifest_500)
    manifest_ids = {str(row["page_id"]) for row in manifest}
    qa_rows = read_jsonl(pm.qa_pairs_shared_path)
    qa_by_page = _group_qa_by_page(qa_rows, manifest_ids)

    corpora_root = Path(exp2_cfg["outputs"]["corpora_root"])
    indexes_root = Path(exp2_cfg["outputs"]["indexes_root"])
    runs_root = Path(exp2_cfg["outputs"]["retrieval_runs_root"])

    k_default = int(exp2_cfg.get("retrieval", {}).get("default_k", 10))
    retrievers = list(exp2_cfg.get("retrieval", {}).get("retrievers", ["bm25", "dense"]))

    encoder: DenseEncoder | None = None
    if "dense" in retrievers and bool(exp2_cfg.get("dense", {}).get("enabled", True)):
        d = exp2_cfg.get("dense", {}) or {}
        encoder = DenseEncoder(
            model_name=str(d.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
            device=str(d.get("device", "cpu")),
            normalize=bool(d.get("normalize_embeddings", True)),
            cache_dir=str(d.get("cache_dir") or "") or None,
        )

    summaries: dict[str, Any] = {}
    started = time.time()
    for pipeline in PIPELINES:
        for condition in CONDITIONS:
            t0 = time.time()
            chunk_text, _chunk_page = _load_chunk_text_map(corpora_root, pipeline, condition)
            page_status = _load_page_status(corpora_root, pipeline, condition)

            for retriever in retrievers:
                run_dir = runs_root / f"{pipeline}_{condition}_{retriever}"
                ensure_dir(run_dir)
                if retriever == "bm25":
                    rows = _run_bm25_for_pc(
                        pipeline=pipeline,
                        condition=condition,
                        qa_by_page=qa_by_page,
                        indexes_root=indexes_root,
                        chunk_text=chunk_text,
                        page_status=page_status,
                        k=k_default,
                    )
                elif retriever == "dense":
                    if encoder is None:
                        rows = []
                    else:
                        rows = _run_dense_for_pc(
                            pipeline=pipeline,
                            condition=condition,
                            qa_by_page=qa_by_page,
                            indexes_root=indexes_root,
                            chunk_text=chunk_text,
                            page_status=page_status,
                            encoder=encoder,
                            k=k_default,
                            batch_size=int(exp2_cfg.get("dense", {}).get("batch_size", 64)),
                        )
                else:
                    raise ValueError(f"Unknown retriever: {retriever}")

                run_path = run_dir / "run.jsonl"
                atomic_write_jsonl(run_path, rows)
                key = f"{pipeline}_{condition}_{retriever}"
                summaries[key] = {
                    "pipeline": pipeline,
                    "condition": condition,
                    "retriever": retriever,
                    "num_queries": len(rows),
                    "num_with_hits": sum(1 for r in rows if r["hits"]),
                    "run_path": str(run_path),
                    "elapsed_seconds": round(time.time() - t0, 2),
                }
                print(
                    f"[run] {key} queries={len(rows)} with_hits={summaries[key]['num_with_hits']} "
                    f"elapsed={summaries[key]['elapsed_seconds']}s",
                    flush=True,
                )

    log_dir = Path(exp2_cfg["logging"]["log_dir"])
    ensure_dir(log_dir)
    log_path = log_dir / ("retrieval_run_debug20_summary.md" if debug else "retrieval_run_summary.md")
    write_text(log_path, _render_run_md(summaries, total_elapsed=time.time() - started, debug=debug))
    return {"summaries": summaries, "summary_md": str(log_path), "total_elapsed_seconds": round(time.time() - started, 2)}


def _render_run_md(summaries: dict[str, Any], total_elapsed: float, debug: bool) -> str:
    lines = [
        f"# Retrieval Run Summary ({'debug20' if debug else 'full500'})",
        "",
        f"- total_elapsed_seconds: `{round(total_elapsed, 2)}`",
        "",
        "| pipeline | condition | retriever | queries | with_hits | elapsed_s |",
        "|---|---|---|---:|---:|---:|",
    ]
    for s in summaries.values():
        lines.append(
            "| {p} | {c} | {r} | {q} | {wh} | {e} |".format(
                p=s["pipeline"], c=s["condition"], r=s["retriever"],
                q=s["num_queries"], wh=s["num_with_hits"], e=s["elapsed_seconds"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BM25 + dense retrieval per (pipeline, condition).")
    parser.add_argument("--config", default="experiment_add/exp2_retrieval/configs/exp2_retrieval.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    result = run_all(Path(args.config), debug=args.debug)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
