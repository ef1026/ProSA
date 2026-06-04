"""EIR-targeted probe placement (L5 — Element Interference Ratio).

Greedy multi-stamp placement to achieve a desired EIR value.
EIR = |{e ∈ E : bbox(e) ∩ S' ≠ ∅}| / |E|
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def is_element_interfered(element: dict, center: Tuple[int, int], radius: int) -> bool:
    """Rectangle-circle intersection test.

    Returns True when the element bbox overlaps the circle of *radius*
    centred at *center*.
    """
    ex, ey = element["x"], element["y"]
    ew, eh = element["w"], element["h"]
    cx, cy = center

    # closest point on rect to circle centre
    closest_x = max(ex - ew // 2, min(cx, ex + ew // 2))
    closest_y = max(ey - eh // 2, min(cy, ey + eh // 2))

    dist = math.sqrt((cx - closest_x) ** 2 + (cy - closest_y) ** 2)
    return dist < radius


def find_positions_for_eir(
    image_shape: tuple,
    elements: list,
    target_eir: float,
    influence_radius: int = 70,
    max_probes: int = 30,
) -> List[Tuple[int, int]]:
    """Find probe placement positions that achieve *target_eir*.

    Uses greedy set-cover: each iteration picks the candidate position
    (an element centre) that covers the most not-yet-interfered elements.

    Parameters
    ----------
    image_shape : (H, W, ...)
    elements : list of dicts with keys x, y, w, h
    target_eir : desired fraction of interfered elements (0–1)
    influence_radius : geometric interference radius of the probe
    max_probes : maximum number of probes to place

    Returns
    -------
    List of (x, y) positions to place probes at.
    """
    h, w = image_shape[:2]
    total = len(elements)
    if total == 0 or target_eir <= 0:
        return []

    target_count = max(1, int(round(total * target_eir)))

    # Pre-compute candidate positions (element centres, clamped to valid range)
    candidates: List[Tuple[int, int]] = []
    for elem in elements:
        ex, ey = elem["x"], elem["y"]
        cx = max(influence_radius, min(ex, w - influence_radius))
        cy = max(influence_radius, min(ey, h - influence_radius))
        candidates.append((cx, cy))

    # Pre-compute interference matrix for speed
    n_cands = len(candidates)
    n_elems = len(elements)
    interferes = np.zeros((n_cands, n_elems), dtype=bool)
    for c, cand in enumerate(candidates):
        for i, elem in enumerate(elements):
            interferes[c, i] = is_element_interfered(elem, cand, influence_radius)

    interfered = np.zeros(n_elems, dtype=bool)
    positions: List[Tuple[int, int]] = []

    while int(interfered.sum()) < target_count and len(positions) < max_probes:
        gains = np.zeros(n_cands, dtype=int)
        for c in range(n_cands):
            gains[c] = int((interferes[c] & ~interfered).sum())

        best_c = int(gains.argmax())
        if gains[best_c] == 0:
            break

        positions.append(candidates[best_c])
        interfered |= interferes[best_c]

    return positions


# backward compat aliases
find_positions_for_ncsic = find_positions_for_eir
