"""Dense retrieval wrapper around sentence-transformers.

The encoder is loaded lazily so that scripts which only need BM25 (e.g.,
audits) do not pay the import cost. Embeddings are stored row-aligned with
``chunk_ids`` and ``page_ids`` so page-internal retrieval is a simple boolean
mask + top-k over cosine similarities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np


_ENCODER_CACHE: dict[tuple[str, str, bool], object] = {}


@dataclass
class DenseEncoder:
    """Thin wrapper that hides sentence-transformers from callers."""

    model_name: str
    device: str = "cpu"
    normalize: bool = True
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        cache_key = (self.model_name, self.device, self.normalize)
        if cache_key in _ENCODER_CACHE:
            self._model = _ENCODER_CACHE[cache_key]
            return
        if self.cache_dir:
            os.environ.setdefault("HF_HOME", str(self.cache_dir))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(self.cache_dir))
            os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(self.cache_dir))
        self._model = SentenceTransformer(self.model_name, device=self.device, cache_folder=self.cache_dir)
        _ENCODER_CACHE[cache_key] = self._model

    @property
    def dim(self) -> int:
        try:
            return int(self._model.get_sentence_embedding_dimension())
        except Exception:
            return int(self.encode(["_"]).shape[1])

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of strings into a float32 ndarray of shape (N, D)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self._model.encode(
            texts,
            batch_size=int(batch_size),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)


def build_dense_payload(
    chunks: Iterable[dict[str, str]],
    encoder: DenseEncoder,
    batch_size: int = 64,
) -> tuple[list[str], list[str], np.ndarray]:
    """Encode an iterable of chunks; returns ``(chunk_ids, page_ids, embeddings)``."""
    chunk_list = list(chunks)
    chunk_ids = [str(c["chunk_id"]) for c in chunk_list]
    page_ids = [str(c.get("page_id", "")) for c in chunk_list]
    texts = [str(c.get("text", "") or " ") for c in chunk_list]
    embeddings = encoder.encode(texts, batch_size=batch_size)
    return chunk_ids, page_ids, embeddings


def encode_queries(encoder: DenseEncoder, queries: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode a list of question strings into a float32 ndarray."""
    return encoder.encode(queries, batch_size=batch_size)


def topk_for_page(
    page_id: str,
    page_ids: np.ndarray,
    chunk_ids: list[str],
    embeddings: np.ndarray,
    query_vec: np.ndarray,
    k: int,
) -> list[tuple[str, float]]:
    """Return up to ``k`` ``(chunk_id, score)`` hits restricted to ``page_id``."""
    mask = page_ids == page_id
    if not mask.any():
        return []
    sub_emb = embeddings[mask]
    sub_ids = [chunk_ids[i] for i, keep in enumerate(mask) if keep]
    scores = sub_emb @ query_vec.astype(np.float32)
    order = np.argsort(-scores)
    out: list[tuple[str, float]] = []
    for i in order[:k]:
        out.append((sub_ids[int(i)], float(scores[int(i)])))
    return out
