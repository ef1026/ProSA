from __future__ import annotations

import cv2
import numpy as np


def compute_content_mask(annotations: list[dict], img_shape: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in annotations:
        x1, y1, x2, y2 = [int(v) for v in ann["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def compute_anchor_mask(annotations: list[dict], img_shape: tuple[int, int] | tuple[int, int, int], delta: int = 5) -> np.ndarray:
    h, w = img_shape[:2]
    m_content = np.zeros((h, w), dtype=np.uint8)
    boundary_mask = np.zeros((h, w), dtype=np.uint8)

    for ann in annotations:
        x1, y1, x2, y2 = [int(v) for v in ann["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        m_content[y1:y2, x1:x2] = 1
        cv2.rectangle(boundary_mask, (x1, y1), (x2, y2), 1, thickness=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * delta + 1, 2 * delta + 1))
    boundary_dilated = cv2.dilate(boundary_mask, kernel)
    m_content_eroded = cv2.erode(m_content, kernel)
    m_anchor = np.clip(boundary_dilated.astype(int) - m_content_eroded.astype(int), 0, 1)
    return m_anchor.astype(np.uint8)
