from __future__ import annotations

from typing import Any


def classify_failure(instance: dict[str, Any]) -> str:
    """Classify a QA instance using the required ordering."""
    parser_status = str(instance.get("parser_status", ""))
    answer_status = str(instance.get("answer_status", ""))
    if parser_status in {"empty", "failed"} or answer_status in {"parser_empty", "parser_failed"}:
        return "Parser Empty / Invalid"
    if answer_status == "api_failed":
        return "API Failed"
    if answer_status == "context_too_long":
        return "Context Too Long"
    if float(instance.get("em", 0.0)) >= 1.0:
        return "Correct"
    if int(instance.get("answer_missing", 0)) == 1:
        return "Answer Lost"
    if answer_status == "not_found":
        return "Answer Present but NOT_FOUND"
    return "Answer Present but Wrong"


def failure_row(instance: dict[str, Any]) -> dict[str, Any]:
    """Convert a flattened instance to the failure decomposition CSV row."""
    return {
        "pipeline": instance.get("pipeline"),
        "condition": instance.get("condition"),
        "page_id": instance.get("page_id"),
        "qa_id": instance.get("qa_id"),
        "question": instance.get("question"),
        "gold_answer": instance.get("gold_answer"),
        "pred_answer": instance.get("pred_answer"),
        "answer_status": instance.get("answer_status"),
        "parser_status": instance.get("parser_status"),
        "answer_missing": instance.get("answer_missing"),
        "em": instance.get("em"),
        "f1": instance.get("f1"),
        "failure_type": classify_failure(instance),
    }
