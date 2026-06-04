"""Generic retrieval metrics over per-query relevance lists.

A "hit" is a list of dicts with at least ``rank`` (1-indexed) and a relevance
flag. Two relevance flavours are computed independently:

* ``evidence_hit``: chunk text contains the gold evidence span.
* ``answer_hit``:   chunk text contains the gold answer span.

The metrics are the standard intra-document retrieval set; ``ndcg_at_k`` is
included for completeness even though the experiment is binary-relevance.
"""

from __future__ import annotations

import math
from typing import Iterable


def recall_at_k(rel_list: Iterable[int], k: int) -> float:
    """Return ``1.0`` if any of the first ``k`` items has relevance > 0."""
    rel_list = list(rel_list)[:k]
    return 1.0 if any(r > 0 for r in rel_list) else 0.0


def precision_at_k(rel_list: Iterable[int], k: int) -> float:
    """Fraction of top-``k`` items that are relevant."""
    rel_list = list(rel_list)[:k]
    if not rel_list:
        return 0.0
    return float(sum(1 for r in rel_list if r > 0)) / float(k)


def reciprocal_rank(rel_list: Iterable[int], cap_k: int | None = None) -> float:
    """Return ``1/rank`` of the first relevant item, or 0 if none seen.

    ``cap_k`` truncates the search horizon (matches the reported ``MRR@k``).
    """
    for idx, rel in enumerate(rel_list, start=1):
        if cap_k is not None and idx > cap_k:
            break
        if rel > 0:
            return 1.0 / float(idx)
    return 0.0


def ndcg_at_k(rel_list: Iterable[int], k: int) -> float:
    """Binary-relevance NDCG@k. Ideal ranking puts a single 1 at rank 1."""
    rel_list = list(rel_list)[:k]
    dcg = 0.0
    for idx, rel in enumerate(rel_list, start=1):
        if rel > 0:
            dcg += 1.0 / math.log2(idx + 1)
    idcg = 1.0 / math.log2(2)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def per_k_recall(rel_list: Iterable[int], ks: Iterable[int]) -> dict[int, float]:
    """Convenience wrapper returning ``{k: recall@k}`` for each requested k."""
    rel_list = list(rel_list)
    return {int(k): recall_at_k(rel_list, int(k)) for k in ks}
