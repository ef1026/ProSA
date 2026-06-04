from __future__ import annotations

import numpy as np


def _sample_from_mask(mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    idx = int(rng.integers(0, len(xs)))
    return int(xs[idx]), int(ys[idx])


class AnchorTarget:
    def place(self, geometry, layout_info: dict, rng: np.random.Generator) -> tuple[int, int]:
        m_anchor = layout_info["M_anchor"]
        h, w = layout_info["H"], layout_info["W"]
        p = _sample_from_mask(m_anchor, rng)
        return p if p is not None else (w // 2, h // 2)


class ContentTarget:
    def place(self, geometry, layout_info: dict, rng: np.random.Generator) -> tuple[int, int]:
        m_content = layout_info["M_content"]
        h, w = layout_info["H"], layout_info["W"]
        p = _sample_from_mask(m_content, rng)
        return p if p is not None else (w // 2, h // 2)


class RandomTarget:
    def place(self, geometry, layout_info: dict, rng: np.random.Generator) -> tuple[int, int]:
        h, w = layout_info["H"], layout_info["W"]
        return int(rng.integers(0, w)), int(rng.integers(0, h))


class BridgeTarget:
    """Place probe at the midpoint between the two closest annotation blocks.

    If fewer than 2 annotations exist, falls back to image centre and sets
    ``self.last_fallback = True`` so callers can log / filter this case.
    """

    def __init__(self):
        self.last_fallback: bool = False

    def place(self, geometry, layout_info: dict, rng: np.random.Generator) -> tuple[int, int]:
        anns = layout_info.get("annotations", [])
        h, w = layout_info["H"], layout_info["W"]
        if len(anns) < 2:
            self.last_fallback = True
            return w // 2, h // 2

        best_dist = 1e18
        best_center = (w // 2, h // 2)
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                a = anns[i]["bbox"]
                b = anns[j]["bbox"]
                ac = ((a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5)
                bc = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)
                d = (ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2
                if d < best_dist:
                    best_dist = d
                    best_center = (int((ac[0] + bc[0]) / 2), int((ac[1] + bc[1]) / 2))
        self.last_fallback = False
        return best_center
