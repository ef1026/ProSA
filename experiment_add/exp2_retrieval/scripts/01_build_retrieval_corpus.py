"""Build the per (pipeline, condition) page-internal retrieval corpora.

Reads the merged parser outputs already produced by ``shared/`` (clean and
perturbed), turns each page's blocks into block-aware ~400 char chunks, and
writes one ``chunks.jsonl`` per (pipeline, condition) along with a small
``corpus_summary.json`` for auditability.

The script never modifies parser output and never re-runs parsers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiment_add.shared.data.load_manifest import load_manifest
from experiment_add.shared.utils.io import (
    atomic_write_json,
    atomic_write_jsonl,
    ensure_dir,
    read_jsonl,
    read_yaml,
    write_text,
)
from experiment_add.shared.utils.path_manager import PathManager


PIPELINES = ("mineru", "ppstructure")
CONDITIONS = ("clean", "area_matched_erasure", "structural_probe", "large_area_erasure")


def _resolve_base_config(exp2_cfg_path: Path, exp2_cfg: dict[str, Any]) -> Path:
    """Locate ``base.yaml`` from the exp2 config or by walking upward."""
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


def _load_parse_path(pm: PathManager, pipeline: str, condition: str) -> Path:
    if condition == "clean":
        return pm.clean_parse_merged_path(pipeline)
    return pm.perturbed_parse_dir(pipeline, condition) / "merged.jsonl"


def _block_text(block: dict[str, Any]) -> str:
    return str(block.get("text", "") or "").strip()


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[tuple[int, int, str]]:
    """Char-window split for blocks longer than ``max_chars``."""
    if len(text) <= max_chars:
        return [(0, len(text), text)]
    out: list[tuple[int, int, str]] = []
    step = max(1, max_chars - overlap)
    pos = 0
    while pos < len(text):
        end = min(len(text), pos + max_chars)
        out.append((pos, end, text[pos:end]))
        if end == len(text):
            break
        pos += step
    return out


def _greedy_block_chunks(
    blocks: list[dict[str, Any]],
    target_chars: int,
    min_chars: int,
    max_chars: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """Group blocks in reading order into ~target_chars chunks, splitting overlong blocks."""
    sorted_blocks = sorted(
        [b for b in blocks if _block_text(b)],
        key=lambda b: (int(b.get("reading_order", 0)), str(b.get("block_id", ""))),
    )

    chunks: list[dict[str, Any]] = []
    cur_block_ids: list[str] = []
    cur_bboxes: list[list[float]] = []
    cur_text: list[str] = []
    cur_len = 0

    def _flush() -> None:
        nonlocal cur_block_ids, cur_bboxes, cur_text, cur_len
        if not cur_block_ids:
            return
        chunks.append(
            {
                "text": " ".join(cur_text).strip(),
                "source_block_ids": list(cur_block_ids),
                "source_bboxes": [list(b) for b in cur_bboxes],
            }
        )
        cur_block_ids, cur_bboxes, cur_text, cur_len = [], [], [], 0

    for block in sorted_blocks:
        text = _block_text(block)
        block_id = str(block.get("block_id", ""))
        bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]

        if len(text) > max_chars:
            if cur_block_ids:
                _flush()
            for _, _, sub_text in _split_long_text(text, max_chars=max_chars, overlap=overlap):
                chunks.append(
                    {
                        "text": sub_text.strip(),
                        "source_block_ids": [block_id],
                        "source_bboxes": [list(bbox)],
                    }
                )
            continue

        if cur_len > 0 and cur_len + len(text) + 1 > int(target_chars * 1.4) and cur_len >= min_chars:
            _flush()

        cur_block_ids.append(block_id)
        cur_bboxes.append(list(bbox))
        cur_text.append(text)
        cur_len += len(text) + 1

        if cur_len >= int(target_chars * 1.4):
            _flush()

    _flush()
    return [c for c in chunks if c["text"]]


def _make_chunk_records(
    page_id: str,
    pipeline: str,
    condition: str,
    parser_status: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{page_id}__{pipeline}__{condition}__c{idx:04d}"
        out.append(
            {
                "chunk_id": chunk_id,
                "page_id": page_id,
                "pipeline": pipeline,
                "condition": condition,
                "parser_status": parser_status,
                "chunk_index": idx,
                "text": chunk["text"],
                "char_len": len(chunk["text"]),
                "source_block_ids": chunk["source_block_ids"],
                "source_bboxes": chunk["source_bboxes"],
            }
        )
    return out


def _summarize_corpus(records: list[dict[str, Any]], page_status: dict[str, str]) -> dict[str, Any]:
    pages_with_chunks = Counter()
    chunks_per_page: dict[str, int] = Counter()
    chars_per_chunk: list[int] = []
    for r in records:
        pages_with_chunks[r["page_id"]] = 1
        chunks_per_page[r["page_id"]] += 1
        chars_per_chunk.append(int(r["char_len"]))

    counts_status = Counter(page_status.values())
    pages_with_chunks_set = set(pages_with_chunks.keys())
    pages_zero_chunks = [pid for pid, st in page_status.items() if pid not in pages_with_chunks_set]
    return {
        "num_pages_total": len(page_status),
        "num_pages_success": int(counts_status.get("success", 0)),
        "num_pages_empty": int(counts_status.get("empty", 0)),
        "num_pages_failed": int(counts_status.get("failed", 0)),
        "num_pages_with_chunks": len(pages_with_chunks_set),
        "num_pages_zero_chunks": len(pages_zero_chunks),
        "pages_zero_chunks_sample": sorted(pages_zero_chunks)[:20],
        "num_chunks_total": len(records),
        "chunks_per_page_min": min(chunks_per_page.values()) if chunks_per_page else 0,
        "chunks_per_page_max": max(chunks_per_page.values()) if chunks_per_page else 0,
        "chunks_per_page_mean": (sum(chunks_per_page.values()) / len(chunks_per_page)) if chunks_per_page else 0.0,
        "chunks_per_page_median": _median(list(chunks_per_page.values())),
        "chunk_chars_mean": (sum(chars_per_chunk) / len(chars_per_chunk)) if chars_per_chunk else 0.0,
        "chunk_chars_min": min(chars_per_chunk) if chars_per_chunk else 0,
        "chunk_chars_max": max(chars_per_chunk) if chars_per_chunk else 0,
    }


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 0:
        return float((values[mid - 1] + values[mid]) / 2.0)
    return float(values[mid])


def build_for_pipeline_condition(
    pm: PathManager,
    exp2_cfg: dict[str, Any],
    pipeline: str,
    condition: str,
    manifest_ids: set[str],
) -> dict[str, Any]:
    parse_path = _load_parse_path(pm, pipeline, condition)
    parse_rows = read_jsonl(parse_path)
    if not parse_rows:
        raise FileNotFoundError(f"No parser output rows at {parse_path}")

    corpus_cfg = exp2_cfg.get("corpus", {})
    target = int(corpus_cfg.get("target_chars", 400))
    min_c = int(corpus_cfg.get("min_chars", 80))
    max_c = int(corpus_cfg.get("max_chars", 700))
    overlap = int(corpus_cfg.get("overlap_chars", 80))

    records: list[dict[str, Any]] = []
    page_status: dict[str, str] = {}

    for row in parse_rows:
        page_id = str(row.get("page_id", ""))
        if page_id not in manifest_ids:
            continue
        parser_status = str(row.get("parser_status", "") or "")
        page_status[page_id] = parser_status
        if parser_status != "success":
            continue
        blocks = row.get("blocks") or []
        chunks = _greedy_block_chunks(
            blocks,
            target_chars=target,
            min_chars=min_c,
            max_chars=max_c,
            overlap=overlap,
        )
        records.extend(_make_chunk_records(page_id, pipeline, condition, parser_status, chunks))

    corpus_dir = Path(exp2_cfg["outputs"]["corpora_root"]) / f"{pipeline}_{condition}"
    chunks_path = corpus_dir / "chunks.jsonl"
    summary_path = corpus_dir / "corpus_summary.json"
    pages_path = corpus_dir / "pages.jsonl"

    atomic_write_jsonl(chunks_path, records)
    atomic_write_jsonl(
        pages_path,
        [
            {
                "page_id": pid,
                "parser_status": status,
                "num_chunks": sum(1 for r in records if r["page_id"] == pid),
            }
            for pid, status in sorted(page_status.items())
        ],
    )

    summary = _summarize_corpus(records, page_status)
    summary.update(
        {
            "pipeline": pipeline,
            "condition": condition,
            "chunks_path": str(chunks_path),
            "pages_path": str(pages_path),
            "parse_path": str(parse_path),
            "config": {
                "strategy": "block_aware",
                "target_chars": target,
                "min_chars": min_c,
                "max_chars": max_c,
                "overlap_chars": overlap,
            },
        }
    )
    atomic_write_json(summary_path, summary)
    return summary


def build_corpora(exp2_cfg_path: Path, debug: bool) -> dict[str, Any]:
    exp2_cfg = read_yaml(exp2_cfg_path)
    base_cfg = _resolve_base_config(exp2_cfg_path, exp2_cfg)
    pm = PathManager(base_cfg, create_dirs=True)
    manifest = load_manifest(pm.page_manifest_debug20 if debug else pm.page_manifest_500)
    manifest_ids = {str(row["page_id"]) for row in manifest}

    out: dict[str, Any] = {
        "manifest": str(pm.page_manifest_debug20 if debug else pm.page_manifest_500),
        "manifest_size": len(manifest_ids),
        "summaries": {},
    }
    for pipeline in PIPELINES:
        for condition in CONDITIONS:
            summary = build_for_pipeline_condition(pm, exp2_cfg, pipeline, condition, manifest_ids)
            out["summaries"][f"{pipeline}_{condition}"] = summary

    log_dir = Path(exp2_cfg["logging"]["log_dir"])
    ensure_dir(log_dir)
    log_path = log_dir / ("corpus_debug20_summary.md" if debug else "corpus_summary.md")
    write_text(log_path, _render_summary_md(out, debug=debug))
    out["summary_md"] = str(log_path)
    return out


def _render_summary_md(result: dict[str, Any], debug: bool) -> str:
    lines = [
        f"# Retrieval Corpus Summary ({'debug20' if debug else 'full500'})",
        "",
        f"- manifest: `{result['manifest']}`",
        f"- manifest_size: `{result['manifest_size']}`",
        "",
        "## Per (pipeline, condition)",
        "",
        "| pipeline | condition | pages | success | empty | with_chunks | zero_chunks | chunks | chunks/page (min/med/mean/max) | chars (mean/min/max) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for key, s in result["summaries"].items():
        lines.append(
            "| {pipeline} | {condition} | {n} | {ok} | {emp} | {wc} | {zc} | {chunks} | "
            "{cmin}/{cmed}/{cmean:.1f}/{cmax} | {chmean:.1f}/{chmin}/{chmax} |".format(
                pipeline=s["pipeline"],
                condition=s["condition"],
                n=s["num_pages_total"],
                ok=s["num_pages_success"],
                emp=s["num_pages_empty"],
                wc=s["num_pages_with_chunks"],
                zc=s["num_pages_zero_chunks"],
                chunks=s["num_chunks_total"],
                cmin=s["chunks_per_page_min"],
                cmed=s["chunks_per_page_median"],
                cmean=s["chunks_per_page_mean"],
                cmax=s["chunks_per_page_max"],
                chmean=s["chunk_chars_mean"],
                chmin=s["chunk_chars_min"],
                chmax=s["chunk_chars_max"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build page-internal retrieval corpora from shared parser outputs.")
    parser.add_argument("--config", default="experiment_add/exp2_retrieval/configs/exp2_retrieval.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    result = build_corpora(Path(args.config), debug=args.debug)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
