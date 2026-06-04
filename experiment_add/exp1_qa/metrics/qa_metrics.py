from __future__ import annotations

from collections import Counter
from typing import Any

from experiment_add.shared.text.answer_matching import contains_answer, exact_match, token_f1


SPECIAL_ZERO_STATUSES = {"parser_empty", "parser_failed", "api_failed", "context_too_long", "not_found"}


def answer_missing(page_text: str | None, gold_answer: str | None, parser_status: str | None = None) -> int:
    """Return 1 when the gold answer is absent from parser text."""
    if parser_status in {"empty", "failed"}:
        return 1
    return 0 if contains_answer(page_text, gold_answer) else 1


def qa_em(pred_answer: str | None, gold_answer: str | None, answer_status: str | None) -> float:
    """Normalized exact match for answered items; special statuses score 0."""
    if answer_status in SPECIAL_ZERO_STATUSES:
        return 0.0
    return 1.0 if exact_match(pred_answer, gold_answer) else 0.0


def qa_f1(pred_answer: str | None, gold_answer: str | None, answer_status: str | None) -> float:
    """Token F1 for answered items; special statuses score 0."""
    if answer_status in SPECIAL_ZERO_STATUSES:
        return 0.0
    return float(token_f1(pred_answer, gold_answer))


def mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_instances(instances: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize flattened QA answer instances."""
    statuses = Counter(str(row.get("answer_status", "")) for row in instances)
    pages = {str(row.get("page_id", "")) for row in instances}
    return {
        "num_pages": len(pages),
        "num_qa": len({str(row.get("qa_id", "")) for row in instances}),
        "num_answer_items": len(instances),
        "parser_empty_count": statuses["parser_empty"],
        "api_failed_count": statuses["api_failed"],
        "answer_missing_rate": mean([float(row.get("answer_missing", 0)) for row in instances]),
        "not_found_rate": statuses["not_found"] / len(instances) if instances else 0.0,
        "qa_em": mean([float(row.get("em", 0.0)) for row in instances]),
        "qa_f1": mean([float(row.get("f1", 0.0)) for row in instances]),
    }
