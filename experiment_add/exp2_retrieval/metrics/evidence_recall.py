"""Evidence-level relevance: does a retrieved chunk contain the gold evidence span?

The check reuses ``contains_answer`` from the shared text utilities so that
evidence relevance and exp1's ``answer_missing`` use exactly the same
normalization (NFKC, lowercase, punctuation strip, whitespace collapse).
"""

from __future__ import annotations

from experiment_add.shared.text.answer_matching import contains_answer


def is_evidence_chunk(chunk_text: str | None, evidence_text: str | None) -> int:
    """Return 1 when normalized evidence text is contained in the chunk."""
    if not evidence_text:
        return 0
    return 1 if contains_answer(chunk_text, evidence_text) else 0


def label_hits_with_evidence(
    hits: list[dict],
    chunk_text_by_id: dict[str, str],
    evidence_text: str | None,
) -> list[int]:
    """Return a parallel ``[0/1]`` list flagging which hits contain evidence."""
    out: list[int] = []
    for h in hits:
        chunk_text = chunk_text_by_id.get(str(h.get("chunk_id", "")), "") or ""
        out.append(is_evidence_chunk(chunk_text, evidence_text))
    return out
