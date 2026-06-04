from __future__ import annotations

from collections import Counter

from experiment_add.shared.text.normalize_text import normalize_answer


def contains_answer(text: str | None, answer: str | None) -> bool:
    """Return True when normalized answer is a substring of normalized text."""
    norm_text = normalize_answer(text)
    norm_answer = normalize_answer(answer)
    return bool(norm_answer) and norm_answer in norm_text


def count_answer_occurrences(text: str | None, answer: str | None) -> int:
    """Count non-overlapping normalized answer occurrences in normalized text."""
    norm_text = normalize_answer(text)
    norm_answer = normalize_answer(answer)
    if not norm_answer:
        return 0
    return norm_text.count(norm_answer)


def exact_match(prediction: str | None, gold_answer: str | None) -> bool:
    """Return normalized exact match between prediction and gold answer."""
    return normalize_answer(prediction) == normalize_answer(gold_answer)


def token_f1(prediction: str | None, gold_answer: str | None) -> float:
    """Compute token-level F1 after answer normalization.

    Empty prediction and empty gold returns 1.0. If only one side is empty,
    returns 0.0.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
