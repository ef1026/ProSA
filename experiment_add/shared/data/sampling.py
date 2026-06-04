from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


def random_sample(records: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Return a reproducible sample without replacement."""
    if n <= 0 or not records:
        return []
    rng = random.Random(seed)
    pool = list(records)
    if len(pool) <= n:
        rng.shuffle(pool)
        return pool
    return rng.sample(pool, n)


def _should_fallback(records: list[dict[str, Any]], stratify_key: str) -> bool:
    if not records:
        return True
    known = [
        record.get(stratify_key)
        for record in records
        if record.get(stratify_key) not in (None, "", "unknown")
    ]
    return len(known) <= len(records) / 2


def _allocate_counts(n: int, target_distribution: dict[str, float]) -> dict[str, int]:
    total_weight = sum(max(0.0, float(v)) for v in target_distribution.values())
    if n <= 0 or total_weight <= 0:
        return {key: 0 for key in target_distribution}

    raw = {key: n * max(0.0, float(weight)) / total_weight for key, weight in target_distribution.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = n - sum(counts.values())
    ranked = sorted(raw, key=lambda key: (raw[key] - counts[key], raw[key]), reverse=True)
    for key in ranked[:remainder]:
        counts[key] += 1
    return counts


def stratified_sample(
    records: list[dict[str, Any]],
    n: int,
    stratify_key: str,
    target_distribution: dict[str, float],
    seed: int,
) -> list[dict[str, Any]]:
    """Return a reproducible stratified sample without replacement.

    Falls back to random sampling if the stratification key is missing or most
    records have an unknown value. If a stratum is short, remaining slots are
    filled from other strata without duplicates.
    """
    if n <= 0 or not records:
        return []
    if len(records) <= n:
        return random_sample(records, len(records), seed)
    if _should_fallback(records, stratify_key):
        return random_sample(records, n, seed)

    rng = random.Random(seed)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = record.get(stratify_key)
        group = value if value not in (None, "") else "unknown"
        by_group[str(group)].append(record)

    for group_records in by_group.values():
        rng.shuffle(group_records)

    quotas = _allocate_counts(n, target_distribution)
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    for group, quota in quotas.items():
        for record in by_group.get(group, [])[:quota]:
            selected.append(record)
            selected_ids.add(id(record))

    if len(selected) < n:
        leftovers: list[dict[str, Any]] = []
        for group_records in by_group.values():
            leftovers.extend(record for record in group_records if id(record) not in selected_ids)
        rng.shuffle(leftovers)
        for record in leftovers:
            if len(selected) >= n:
                break
            selected.append(record)
            selected_ids.add(id(record))

    rng.shuffle(selected)
    return selected[:n]
