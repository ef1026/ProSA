"""BM25 page-internal retriever wrapper.

Each page becomes its own mini-corpus. Tokenization reuses the same
``normalize_answer`` pipeline as exp1's QA scoring so that retrieval and
answer-hit comparisons share an identical token space.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from rank_bm25 import BM25Okapi

from experiment_add.shared.text.normalize_text import normalize_answer


@dataclass
class BM25PageIndex:
    """Holds chunk ids + the underlying ``BM25Okapi`` model for one page."""

    chunk_ids: list[str]
    bm25: BM25Okapi


def tokenize(text: str | None) -> list[str]:
    """Normalize and whitespace-tokenize a string for BM25."""
    return [tok for tok in normalize_answer(text).split() if tok]


def _safe_bm25(tokens_per_chunk: list[list[str]], k1: float, b: float) -> BM25Okapi:
    """Construct ``BM25Okapi`` while substituting empty token lists."""
    safe_tokens = [toks if toks else ["<empty>"] for toks in tokens_per_chunk]
    return BM25Okapi(safe_tokens, k1=k1, b=b)


def build_page_indexes(
    chunks_by_page: dict[str, list[dict[str, str]]],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, BM25PageIndex]:
    """Create one BM25 index per page from a ``{page_id: [chunk dict]}`` map."""
    out: dict[str, BM25PageIndex] = {}
    for page_id, chunks in chunks_by_page.items():
        if not chunks:
            continue
        chunk_ids = [str(c["chunk_id"]) for c in chunks]
        tokens = [tokenize(c.get("text", "")) for c in chunks]
        out[page_id] = BM25PageIndex(chunk_ids=chunk_ids, bm25=_safe_bm25(tokens, k1=k1, b=b))
    return out


def query_page(index: BM25PageIndex, question: str, k: int) -> list[tuple[str, float]]:
    """Return up to ``k`` ``(chunk_id, score)`` hits for the question."""
    tokens = tokenize(question)
    if not tokens:
        return []
    scores = index.bm25.get_scores(tokens)
    if not len(scores):
        return []
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    out: list[tuple[str, float]] = []
    for i in order[:k]:
        out.append((index.chunk_ids[i], float(scores[i])))
    return out


def group_chunks_by_page(chunks: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group a flat chunk iterable by ``page_id`` while preserving order."""
    by_page: dict[str, list[dict[str, str]]] = defaultdict(list)
    for chunk in chunks:
        by_page[str(chunk.get("page_id", ""))].append(chunk)
    return dict(by_page)
