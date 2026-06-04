from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def _is_valid_number(value: Any) -> bool:
    """Return True for int/float values that are not NaN."""
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(numeric)


def group_records(records: Iterable[dict[str, Any]], keys: list[str] | tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """Group records by one or more keys; missing fields group as None."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_key = tuple(record.get(key) for key in keys)
        grouped[group_key].append(record)
    return dict(grouped)


def mean_or_nan(values: Iterable[Any]) -> float:
    """Return the mean of numeric non-NaN values, or NaN when none exist."""
    numeric_values = [float(value) for value in values if _is_valid_number(value)]
    if not numeric_values:
        return float("nan")
    return sum(numeric_values) / len(numeric_values)


def safe_divide(numerator: Any, denominator: Any) -> float:
    """Divide two values, returning NaN for invalid input or zero denominator."""
    if not _is_valid_number(numerator) or not _is_valid_number(denominator):
        return float("nan")
    denom = float(denominator)
    if denom == 0:
        return float("nan")
    return float(numerator) / denom


def _aggregate_group(
    group_key: tuple[Any, ...],
    key_names: list[str],
    group: list[dict[str, Any]],
    metric_fields: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    row = {name: value for name, value in zip(key_names, group_key, strict=False)}
    row["n_records"] = len(group)
    for field in metric_fields:
        row[f"{field}_mean"] = mean_or_nan(record.get(field) for record in group)
    return row


def aggregate_by_pipeline_condition(
    records: Iterable[dict[str, Any]],
    metric_fields: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """Aggregate metric means by `pipeline` and `condition`."""
    key_names = ["pipeline", "condition"]
    grouped = group_records(records, key_names)
    return [
        _aggregate_group(group_key, key_names, group, metric_fields)
        for group_key, group in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0]))
    ]


def aggregate_by_page(
    records: Iterable[dict[str, Any]],
    metric_fields: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """Aggregate metric means by `image_id`."""
    key_names = ["image_id"]
    grouped = group_records(records, key_names)
    return [
        _aggregate_group(group_key, key_names, group, metric_fields)
        for group_key, group in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0]))
    ]
