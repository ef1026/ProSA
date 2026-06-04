"""Persistence helpers for BM25 and dense retrieval indexes.

Indexes are stored per (pipeline, condition) under
``experiment_add/outputs/exp2_retrieval/indexes/{pipeline}_{condition}/``.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiment_add.shared.utils.io import ensure_dir


BM25_FILE = "bm25.pkl"
DENSE_EMBED_FILE = "dense_embeddings.npz"
DENSE_META_FILE = "dense_meta.json"
INDEX_MANIFEST_FILE = "manifest.json"


@dataclass
class DensePayload:
    """In-memory dense index payload for a single (pipeline, condition)."""

    chunk_ids: list[str]
    page_ids: list[str]
    embeddings: np.ndarray

    def __post_init__(self) -> None:
        if len(self.chunk_ids) != len(self.page_ids):
            raise ValueError("chunk_ids and page_ids must have equal length")
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.chunk_ids):
            raise ValueError("embeddings shape inconsistent with chunk_ids")


def index_dir_for(indexes_root: Path, pipeline: str, condition: str) -> Path:
    """Return the canonical index directory for one (pipeline, condition)."""
    return Path(indexes_root) / f"{pipeline}_{condition}"


def write_manifest(index_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write ``manifest.json`` next to the persisted index files."""
    ensure_dir(index_dir)
    path = index_dir / INDEX_MANIFEST_FILE
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_manifest(index_dir: Path) -> dict[str, Any]:
    """Load ``manifest.json`` if present; otherwise return an empty dict."""
    path = index_dir / INDEX_MANIFEST_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_bm25_payload(index_dir: Path, page_to_index: dict[str, dict[str, Any]]) -> Path:
    """Persist a ``{page_id: {chunk_ids, bm25}}`` mapping as a pickle file."""
    ensure_dir(index_dir)
    path = index_dir / BM25_FILE
    with path.open("wb") as f:
        pickle.dump(page_to_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_bm25_payload(index_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the ``{page_id: {chunk_ids, bm25}}`` pickle written by :func:`save_bm25_payload`."""
    path = index_dir / BM25_FILE
    if not path.exists():
        raise FileNotFoundError(f"BM25 index missing at {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def save_dense_payload(index_dir: Path, payload: DensePayload, meta: dict[str, Any]) -> tuple[Path, Path]:
    """Persist dense embeddings (npz) plus a small JSON metadata sidecar."""
    ensure_dir(index_dir)
    npz_path = index_dir / DENSE_EMBED_FILE
    meta_path = index_dir / DENSE_META_FILE
    np.savez_compressed(
        npz_path,
        embeddings=payload.embeddings.astype(np.float32, copy=False),
        chunk_ids=np.array(payload.chunk_ids, dtype=object),
        page_ids=np.array(payload.page_ids, dtype=object),
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return npz_path, meta_path


def load_dense_payload(index_dir: Path) -> tuple[DensePayload, dict[str, Any]]:
    """Load the dense embeddings + metadata produced by :func:`save_dense_payload`."""
    npz_path = index_dir / DENSE_EMBED_FILE
    meta_path = index_dir / DENSE_META_FILE
    if not npz_path.exists():
        raise FileNotFoundError(f"Dense index missing at {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    payload = DensePayload(
        chunk_ids=[str(x) for x in data["chunk_ids"].tolist()],
        page_ids=[str(x) for x in data["page_ids"].tolist()],
        embeddings=np.asarray(data["embeddings"], dtype=np.float32),
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return payload, meta
