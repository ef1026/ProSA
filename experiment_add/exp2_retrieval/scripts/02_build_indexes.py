"""Build BM25 + dense indexes for each (pipeline, condition) corpus.

The dense encoder is loaded once and reused across all corpora to amortize the
sentence-transformers warm-up. Indexes are persisted under
``experiment_add/outputs/exp2_retrieval/indexes/{pipeline}_{condition}/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiment_add.exp2_retrieval.retrievers.bm25_retriever import (
    build_page_indexes,
    group_chunks_by_page,
)
from experiment_add.exp2_retrieval.retrievers.dense_retriever import (
    DenseEncoder,
    build_dense_payload,
)
from experiment_add.exp2_retrieval.retrievers.index_io import (
    DensePayload,
    index_dir_for,
    save_bm25_payload,
    save_dense_payload,
    write_manifest,
)
from experiment_add.shared.utils.io import (
    ensure_dir,
    read_jsonl,
    read_yaml,
    write_text,
)


PIPELINES = ("mineru", "ppstructure")
CONDITIONS = ("clean", "area_matched_erasure", "structural_probe", "large_area_erasure")


def _load_chunks(corpora_root: Path, pipeline: str, condition: str) -> list[dict[str, Any]]:
    chunks_path = corpora_root / f"{pipeline}_{condition}" / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl missing at {chunks_path}; run 01_build_retrieval_corpus first")
    return read_jsonl(chunks_path)


def build_for_pipeline_condition(
    indexes_root: Path,
    corpora_root: Path,
    pipeline: str,
    condition: str,
    bm25_cfg: dict[str, Any],
    encoder: DenseEncoder | None,
    dense_batch_size: int,
) -> dict[str, Any]:
    chunks = _load_chunks(corpora_root, pipeline, condition)
    index_dir = index_dir_for(indexes_root, pipeline, condition)
    ensure_dir(index_dir)

    if not chunks:
        manifest = {
            "pipeline": pipeline,
            "condition": condition,
            "num_chunks": 0,
            "bm25": {"enabled": True, "pages": 0, "k1": float(bm25_cfg.get("k1", 1.5)), "b": float(bm25_cfg.get("b", 0.75))},
            "dense": {"enabled": encoder is not None, "embeddings": 0},
        }
        write_manifest(index_dir, manifest)
        return manifest

    chunks_by_page = group_chunks_by_page(chunks)
    page_indexes = build_page_indexes(
        chunks_by_page,
        k1=float(bm25_cfg.get("k1", 1.5)),
        b=float(bm25_cfg.get("b", 0.75)),
    )
    bm25_payload = {
        page_id: {"chunk_ids": idx.chunk_ids, "bm25": idx.bm25}
        for page_id, idx in page_indexes.items()
    }
    save_bm25_payload(index_dir, bm25_payload)

    dense_meta: dict[str, Any] = {"enabled": False}
    if encoder is not None:
        chunk_ids, page_ids, embeddings = build_dense_payload(chunks, encoder, batch_size=dense_batch_size)
        dense_payload = DensePayload(chunk_ids=chunk_ids, page_ids=page_ids, embeddings=embeddings)
        save_dense_payload(
            index_dir,
            dense_payload,
            meta={
                "model_name": encoder.model_name,
                "device": encoder.device,
                "normalize_embeddings": encoder.normalize,
                "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
                "num_chunks": int(embeddings.shape[0]) if embeddings.ndim == 2 else 0,
            },
        )
        dense_meta = {
            "enabled": True,
            "model_name": encoder.model_name,
            "device": encoder.device,
            "normalize_embeddings": encoder.normalize,
            "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "embeddings": int(embeddings.shape[0]) if embeddings.ndim == 2 else 0,
        }

    manifest = {
        "pipeline": pipeline,
        "condition": condition,
        "num_chunks": len(chunks),
        "num_pages_indexed": len(page_indexes),
        "bm25": {
            "enabled": True,
            "pages": len(page_indexes),
            "k1": float(bm25_cfg.get("k1", 1.5)),
            "b": float(bm25_cfg.get("b", 0.75)),
        },
        "dense": dense_meta,
    }
    write_manifest(index_dir, manifest)
    return manifest


def build_indexes(exp2_cfg_path: Path, debug: bool) -> dict[str, Any]:
    exp2_cfg = read_yaml(exp2_cfg_path)
    corpora_root = Path(exp2_cfg["outputs"]["corpora_root"])
    indexes_root = Path(exp2_cfg["outputs"]["indexes_root"])
    ensure_dir(indexes_root)

    bm25_cfg = exp2_cfg.get("bm25", {})
    dense_cfg = exp2_cfg.get("dense", {}) or {}
    encoder: DenseEncoder | None = None
    if bool(dense_cfg.get("enabled", True)):
        if dense_cfg.get("cache_dir"):
            os.environ.setdefault("HF_HOME", str(dense_cfg["cache_dir"]))
        encoder = DenseEncoder(
            model_name=str(dense_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
            device=str(dense_cfg.get("device", "cpu")),
            normalize=bool(dense_cfg.get("normalize_embeddings", True)),
            cache_dir=str(dense_cfg.get("cache_dir") or "") or None,
        )

    summaries: dict[str, Any] = {}
    started = time.time()
    for pipeline in PIPELINES:
        for condition in CONDITIONS:
            t0 = time.time()
            manifest = build_for_pipeline_condition(
                indexes_root=indexes_root,
                corpora_root=corpora_root,
                pipeline=pipeline,
                condition=condition,
                bm25_cfg=bm25_cfg,
                encoder=encoder,
                dense_batch_size=int(dense_cfg.get("batch_size", 64)),
            )
            manifest["elapsed_seconds"] = round(time.time() - t0, 2)
            summaries[f"{pipeline}_{condition}"] = manifest
            print(
                f"[index] {pipeline}/{condition} chunks={manifest.get('num_chunks',0)} "
                f"pages={manifest.get('num_pages_indexed',0)} dim={manifest.get('dense',{}).get('dim',0)} "
                f"elapsed={manifest['elapsed_seconds']}s",
                flush=True,
            )

    log_dir = Path(exp2_cfg["logging"]["log_dir"])
    ensure_dir(log_dir)
    log_path = log_dir / ("index_debug20_summary.md" if debug else "index_summary.md")
    write_text(log_path, _render_index_md(summaries, total_elapsed=time.time() - started, debug=debug))
    return {"summaries": summaries, "summary_md": str(log_path), "total_elapsed_seconds": round(time.time() - started, 2)}


def _render_index_md(summaries: dict[str, Any], total_elapsed: float, debug: bool) -> str:
    lines = [
        f"# Retrieval Index Summary ({'debug20' if debug else 'full500'})",
        "",
        f"- total_elapsed_seconds: `{round(total_elapsed, 2)}`",
        "",
        "| pipeline | condition | chunks | pages_indexed | dense_dim | dense_n | elapsed_s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, m in summaries.items():
        d = m.get("dense", {}) or {}
        lines.append(
            "| {p} | {c} | {chunks} | {pages} | {dim} | {n} | {e} |".format(
                p=m.get("pipeline", ""),
                c=m.get("condition", ""),
                chunks=m.get("num_chunks", 0),
                pages=m.get("num_pages_indexed", 0),
                dim=d.get("dim", 0),
                n=d.get("embeddings", 0),
                e=m.get("elapsed_seconds", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BM25 + dense indexes per (pipeline, condition).")
    parser.add_argument("--config", default="experiment_add/exp2_retrieval/configs/exp2_retrieval.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    result = build_indexes(Path(args.config), debug=args.debug)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
