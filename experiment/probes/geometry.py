from __future__ import annotations

import math

import cv2
import numpy as np


class LineGeometry:
    def __init__(self, theta: float = 0.0, length_ratio: float = 1.0, width: float = 3.0):
        self.theta = float(theta)
        self.length_ratio = float(length_ratio)
        self.width = max(1, int(round(width)))

    def generate(self, shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
        h, w = shape[:2]
        cx, cy = int(center[0]), int(center[1])
        mask = np.zeros((h, w), dtype=np.uint8)
        length = int(round(self.length_ratio * (w if abs(self.theta) < 45 or abs(self.theta) > 135 else h)))
        length = max(10, length)
        rad = math.radians(self.theta)
        dx = int(round((length / 2) * math.cos(rad)))
        dy = int(round((length / 2) * math.sin(rad)))
        x0, y0 = cx - dx, cy - dy
        x1, y1 = cx + dx, cy + dy
        cv2.line(mask, (x0, y0), (x1, y1), 1, thickness=self.width)
        return mask


class DiskGeometry:
    def __init__(self, rx: int, ry: int):
        self.rx = int(rx)
        self.ry = int(ry)

    def generate(self, shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
        h, w = shape[:2]
        cx, cy = int(center[0]), int(center[1])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (self.rx, self.ry), 0, 0, 360, 1, thickness=-1)
        return mask


class RectGeometry:
    def __init__(self, w_rect: int, h_rect: int):
        self.w_rect = int(max(2, w_rect))
        self.h_rect = int(max(2, h_rect))

    def generate(self, shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
        h, w = shape[:2]
        cx, cy = int(center[0]), int(center[1])
        mask = np.zeros((h, w), dtype=np.uint8)
        x1 = max(0, cx - self.w_rect // 2)
        y1 = max(0, cy - self.h_rect // 2)
        x2 = min(w, x1 + self.w_rect)
        y2 = min(h, y1 + self.h_rect)
        mask[y1:y2, x1:x2] = 1
        return mask


class PointGeometry:
    def __init__(self, radius: int = 2, n_points: int = 50, spread_sigma: float = 30.0,
                 rng: np.random.Generator | None = None):
        self.radius = int(max(1, radius))
        self.n_points = int(max(1, n_points))
        self.spread_sigma = float(max(1.0, spread_sigma))
        self._rng = rng if rng is not None else np.random.default_rng()

    def generate(self, shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
        h, w = shape[:2]
        cx, cy = int(center[0]), int(center[1])
        mask = np.zeros((h, w), dtype=np.uint8)
        xs = self._rng.normal(cx, self.spread_sigma, size=self.n_points).astype(int)
        ys = self._rng.normal(cy, self.spread_sigma, size=self.n_points).astype(int)
        for x, y in zip(xs, ys):
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(mask, (x, y), self.radius, 1, thickness=-1)
        return mask


class BlobGeometry:
    def __init__(self, r_base: int = 50, epsilon: float = 0.3,
                 rng: np.random.Generator | None = None):
        self.r_base = int(max(5, r_base))
        self.epsilon = float(max(0.0, epsilon))
        self._rng = rng if rng is not None else np.random.default_rng()

    def generate(self, shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
        h, w = shape[:2]
        cx, cy = int(center[0]), int(center[1])
        mask = np.zeros((h, w), dtype=np.uint8)
        angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        noise = self._rng.uniform(-self.epsilon, self.epsilon, size=len(angles))
        radii = self.r_base * (1.0 + noise)
        pts = []
        for a, r in zip(angles, radii):
            x = int(round(cx + r * np.cos(a)))
            y = int(round(cy + r * np.sin(a)))
            pts.append([max(0, min(w - 1, x)), max(0, min(h - 1, y))])
        contour = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [contour], 1)
        return mask
