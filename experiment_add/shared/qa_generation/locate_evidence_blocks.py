from __future__ import annotations

from typing import Any

from experiment_add.shared.text.normalize_text import normalize_answer


def _token_overlap_score(a: str, b: str) -> float:
    """Return a simple token overlap score."""
    ta = set(normalize_answer(a).split())
    tb = set(normalize_answer(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta))


def _find_block(blocks: list[dict[str, Any]], answer: str, evidence_text: str) -> dict[str, Any] | None:
    """Find best evidence block by exact answer containment, then overlap."""
    norm_answer = normalize_answer(answer)
    if norm_answer:
        for block in blocks:
            if norm_answer in normalize_answer(block.get("text", "")):
                return block
    scored = [(_token_overlap_score(evidence_text, block.get("text", "")), block) for block in blocks]
    scored = [item for item in scored if item[0] > 0]
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def locate_evidence_blocks(
    answer: str,
    evidence_text: str,
    mineru_record: dict[str, Any],
    ppstructure_record: dict[str, Any],
) -> dict[str, Any]:
    """Locate evidence block ids and bboxes in both clean parser outputs."""
    mineru_block = _find_block(mineru_record.get("blocks", []), answer, evidence_text)
    pp_block = _find_block(ppstructure_record.get("blocks", []), answer, evidence_text)
    return {
        "evidence_block_id_mineru": mineru_block.get("block_id") if mineru_block else None,
        "evidence_bbox_mineru": mineru_block.get("bbox") if mineru_block else None,
        "evidence_block_id_ppstructure": pp_block.get("block_id") if pp_block else None,
        "evidence_bbox_ppstructure": pp_block.get("bbox") if pp_block else None,
    }
