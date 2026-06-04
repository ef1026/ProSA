from __future__ import annotations

import cv2
import numpy as np


class SolidVisual:
    def __init__(self, color: tuple[int, int, int]):
        self.color = tuple(int(c) for c in color)

    def render(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[mask > 0] = self.color
        return vis


class GradientVisual:
    def __init__(self, color_start: tuple[int, int, int], color_end: tuple[int, int, int]):
        self.color_start = np.array(color_start, dtype=np.float32)
        self.color_end = np.array(color_end, dtype=np.float32)

    def render(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return vis
        x_min, x_max = xs.min(), xs.max()
        denom = max(1, x_max - x_min)
        for x in np.unique(xs):
            t = (x - x_min) / denom
            c = ((1 - t) * self.color_start + t * self.color_end).astype(np.uint8)
            vis[(mask > 0) & (np.indices(mask.shape)[1] == x)] = c
        return vis


class RingVisual:
    def __init__(self, color: tuple[int, int, int]):
        self.color = tuple(int(c) for c in color)

    def render(self, mask: np.ndarray, center: tuple[int, int], radius: int) -> tuple[np.ndarray, np.ndarray]:
        h, w = mask.shape[:2]
        ring_mask = np.zeros((h, w), dtype=np.uint8)
        thickness = max(2, radius // 6)
        cv2.circle(ring_mask, center, int(radius), 1, thickness=thickness)
        ring_mask = np.logical_and(ring_mask > 0, mask > 0).astype(np.uint8)
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[ring_mask > 0] = self.color
        return vis, ring_mask


class NoiseVisual:
    """Procedural fractal-noise texture for realistic stain simulation (V_tex).

    Generates spatially varying colour using multi-octave Gaussian noise,
    fully parameterized and reproducible via seed.
    """

    def __init__(
        self,
        base_color: tuple[int, int, int],
        noise_scale: float = 0.02,
        color_variation: float = 0.3,
        seed: int | None = None,
    ):
        self.base_color = np.array(base_color, dtype=np.float32)
        self.noise_scale = float(noise_scale)
        self.color_variation = float(np.clip(color_variation, 0.0, 1.0))
        self.seed = seed

    # ── internal helpers ──

    @staticmethod
    def _fractal_noise(
        h: int, w: int, rng: np.random.Generator, noise_scale: float, octaves: int = 4,
    ) -> np.ndarray:
        """Multi-octave Gaussian-blurred white noise → [0, 1] scalar field."""
        result = np.zeros((h, w), dtype=np.float32)
        amplitude = 1.0
        total_amp = 0.0
        for i in range(octaves):
            noise = rng.standard_normal((h, w)).astype(np.float32)
            scale = max(1, int(round(1.0 / (noise_scale * (2 ** i)))))
            ksize = max(3, (scale // 2) * 2 + 1)
            ksize = min(ksize, min(h, w) // 2 * 2 + 1)
            if ksize >= 3:
                noise = cv2.GaussianBlur(noise, (ksize, ksize), 0)
            result += amplitude * noise
            total_amp += amplitude
            amplitude *= 0.5
        if total_amp > 0:
            result /= total_amp
        mn, mx = result.min(), result.max()
        return (result - mn) / max(mx - mn, 1e-6)

    # ── public API ──

    def render(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        rng = np.random.default_rng(self.seed)
        noise = self._fractal_noise(h, w, rng, self.noise_scale)

        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            ch = self.base_color[c] * (1.0 + self.color_variation * (noise - 0.5))
            vis[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        # Edge feathering: darken near mask boundary for realism
        if mask.sum() > 0:
            dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            max_dist = dist.max()
            if max_dist > 0:
                edge_f = np.clip(dist / max(5.0, max_dist * 0.3), 0, 1)
                for c in range(3):
                    vis[:, :, c] = (vis[:, :, c].astype(np.float32) * (0.6 + 0.4 * edge_f)).astype(np.uint8)

        vis[mask == 0] = 0
        return vis
