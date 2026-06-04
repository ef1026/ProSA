"""Answer-level relevance: does a chunk contain the gold answer span?

Pairs with :mod:`evidence_recall` to give two distinct retrieval-quality
signals: evidence (the parser preserved the surrounding sentence) vs answer
(the parser preserved at least the literal answer string).
"""

from __future__ import annotations

from experiment_add.shared.text.answer_matching import contains_answer


def is_answer_chunk(chunk_text: str | None, gold_answer: str | None) -> int:
    """Return 1 when normalized gold answer is contained in the chunk."""
    if not gold_answer:
        return 0
    return 1 if contains_answer(chunk_text, gold_answer) else 0


def label_hits_with_answer(
    hits: list[dict],
    chunk_text_by_id: dict[str, str],
    gold_answer: str | None,
) -> list[int]:
    """Return a parallel ``[0/1]`` list flagging which hits contain the gold answer."""
    out: list[int] = []
    for h in hits:
        chunk_text = chunk_text_by_id.get(str(h.get("chunk_id", "")), "") or ""
        out.append(is_answer_chunk(chunk_text, gold_answer))
    return out


def answer_hit_in_topk(rel_answer_list: list[int], k: int) -> int:
    """1 if any of top-k chunks contains the gold answer; 0 otherwise."""
    return 1 if any(r > 0 for r in rel_answer_list[:k]) else 0
