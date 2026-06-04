from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DocumentContext:
    visual: dict
    layout: dict
    spatial: dict
    image: np.ndarray
    annotations: list
    M_anchor: np.ndarray
    M_content: np.ndarray

    def to_text_description(self) -> str:
        desc = f"""Document Analysis:
- Image size: {self.visual['width']}×{self.visual['height']} px
- Mean intensity: {self.visual['mean_intensity']:.1f}/255
- Edge density: {self.visual['edge_density']:.3f}
- Number of layout blocks: {self.layout['N_blocks']}
- Layout entropy: {self.layout['H_layout']:.2f} bits
- Content area ratio: {self.layout['total_content_ratio']:.1%}
- Categories: {self.layout['category_distribution']}
- Has table: {self.layout['has_table']}
- Has figure: {self.layout['has_figure']}
- Multi-column: {self.spatial['is_multi_column']}
- Gap density: {self.spatial['gap_density']:.4f}

Top vulnerable structural gaps:
"""
        for i, vp in enumerate(self.spatial["vulnerable_points"][:5]):
            desc += (
                f"  {i+1}. Between {vp['class_i']} and {vp['class_j']}: "
                f"gap={vp['width']:.0f}px at ({vp['center'][0]}, {vp['center'][1]}), {vp['direction']}\n"
            )
        return desc

    def to_neutral_description(self) -> str:
        """Neutral document description for LLM — no EIR / gap / area / structural hints."""
        w = self.visual["width"]
        h = self.visual["height"]
        lines = [f"Image Size: {w} x {h} pixels"]
        n = self.layout["N_blocks"]
        lines.append(f"Layout Blocks ({n} total):")
        for i, ann in enumerate(self.annotations):
            x1, y1, x2, y2 = [int(round(v)) for v in ann["bbox"]]
            cat = ann.get("category", "unknown")
            lines.append(f"  Block {i}: {cat} [{x1}, {y1}, {x2}, {y2}]")
        mc = "yes" if self.spatial["is_multi_column"] else "no"
        lines.append(f"Multi-column layout: {mc}")
        ht = "yes" if self.layout["has_table"] else "no"
        hf = "yes" if self.layout["has_figure"] else "no"
        lines.append(f"Has table: {ht}")
        lines.append(f"Has figure: {hf}")
        return "\n".join(lines)


class ContextEncoder:
    def encode(self, image, annotations, m_anchor, m_content):
        c_visual = self._extract_visual(image)
        c_layout = self._extract_layout(annotations, image.shape)
        c_spatial = self._extract_spatial(annotations, m_anchor, m_content, image.shape)
        return DocumentContext(
            visual=c_visual,
            layout=c_layout,
            spatial=c_spatial,
            image=image,
            annotations=annotations,
            M_anchor=m_anchor,
            M_content=m_content,
        )

    def _extract_visual(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return {
            "mean_intensity": float(gray.mean()),
            "std_intensity": float(gray.std()),
            "dark_ratio": float((gray < 50).sum() / gray.size),
            "light_ratio": float((gray > 200).sum() / gray.size),
            "edge_density": float(cv2.Canny(gray, 50, 150).mean() / 255),
            "height": image.shape[0],
            "width": image.shape[1],
        }

    def _extract_layout(self, annotations, img_shape):
        h, w = img_shape[:2]
        if len(annotations) == 0:
            return {
                "N_blocks": 0,
                "H_layout": 0,
                "total_content_ratio": 0,
                "has_table": False,
                "has_figure": False,
                "category_distribution": {},
            }
        areas = []
        cats = []
        for ann in annotations:
            x1, y1, x2, y2 = ann["bbox"]
            areas.append(float((x2 - x1) * (y2 - y1) / (h * w)))
            cats.append(ann["category"])
        cat_counts = Counter(cats)
        cat_probs = {c: n / len(annotations) for c, n in cat_counts.items()}
        return {
            "N_blocks": len(annotations),
            "H_layout": -sum(p * np.log2(p) for p in cat_probs.values() if p > 0),
            "mean_block_area": float(np.mean(areas)),
            "total_content_ratio": float(sum(areas)),
            "has_table": "Table" in cat_counts,
            "has_figure": "Figure" in cat_counts,
            "n_text_blocks": cat_counts.get("Text", 0),
            "n_categories": len(cat_counts),
            "category_distribution": cat_probs,
        }

    def _extract_spatial(self, annotations, m_anchor, m_content, img_shape):
        h, w = img_shape[:2]
        gap_density = float(m_anchor.sum() / (h * w))
        vulnerable_points = self._find_vulnerable_points(annotations)
        is_multi_column = self._detect_multi_column(annotations, w)
        return {
            "gap_density": gap_density,
            "vulnerable_points": vulnerable_points,
            "is_multi_column": is_multi_column,
        }

    def _compute_gap(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        h_overlap = min(ax2, bx2) - max(ax1, bx1)
        v_gap = max(by1 - ay2, ay1 - by2)

        v_overlap = min(ay2, by2) - max(ay1, by1)
        h_gap = max(bx1 - ax2, ax1 - bx2)

        if v_gap > 0 and h_overlap > 0:
            center = ((ax1 + ax2 + bx1 + bx2) // 4, (ay2 + by1) // 2 if by1 > ay2 else (ay1 + by2) // 2)
            return {"center": center, "width": v_gap, "direction": "vertical_gap"}
        if h_gap > 0 and v_overlap > 0:
            center = ((ax2 + bx1) // 2 if bx1 > ax2 else (ax1 + bx2) // 2, (ay1 + ay2 + by1 + by2) // 4)
            return {"center": center, "width": h_gap, "direction": "horizontal_gap"}
        return None

    def _find_vulnerable_points(self, annotations, n=10):
        gaps = []
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                bi = annotations[i]["bbox"]
                bj = annotations[j]["bbox"]
                gap = self._compute_gap(bi, bj)
                if gap is not None and 0 < gap["width"] < 80:
                    gap["block_i"] = i
                    gap["block_j"] = j
                    gap["class_i"] = annotations[i]["category"]
                    gap["class_j"] = annotations[j]["category"]
                    gaps.append(gap)
        gaps.sort(key=lambda x: x["width"])
        return gaps[:n]

    def _detect_multi_column(self, annotations, img_width, threshold=0.3):
        if len(annotations) < 4:
            return False
        x_centers = [(ann["bbox"][0] + ann["bbox"][2]) / 2 for ann in annotations]
        left = sum(1 for x in x_centers if x < img_width * 0.45)
        right = sum(1 for x in x_centers if x > img_width * 0.55)
        return left >= 2 and right >= 2
