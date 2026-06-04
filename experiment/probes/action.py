from __future__ import annotations

import numpy as np


class InjectAction:
    def composite(self, image: np.ndarray, mask: np.ndarray, visual: np.ndarray) -> np.ndarray:
        out = image.copy()
        out[mask > 0] = visual[mask > 0]
        return out


class BlendAction:
    def __init__(self, alpha: float = 0.5):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def composite(self, image: np.ndarray, mask: np.ndarray, visual: np.ndarray) -> np.ndarray:
        out = image.copy().astype(np.float32)
        vis = visual.astype(np.float32)
        m = mask > 0
        out[m] = self.alpha * vis[m] + (1 - self.alpha) * out[m]
        return np.clip(out, 0, 255).astype(np.uint8)


class EraseAction:
    def __init__(self, beta: float = 0.5):
        self.beta = float(np.clip(beta, 0.0, 1.0))

    def composite(self, image: np.ndarray, mask: np.ndarray, visual: np.ndarray) -> np.ndarray:
        out = image.copy().astype(np.float32)
        m = mask > 0
        out[m] = (1 - self.beta) * out[m] + self.beta * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)
