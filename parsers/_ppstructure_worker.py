#!/usr/bin/env python
# parsers/_ppstructure_worker.py
# Long-lived subprocess worker for PP-Structure GPU inference.
#
# Runs in the 'paddle' (PaddleOCR 3.x) or legacy 'ppocr' (PaddleOCR 2.x) conda
# environment (isolated from torch/MinerU) to avoid CUDA DLL conflicts on Windows.
# Communicates with the parent process via stdin/stdout using a simple
# line-delimited JSON protocol:
#
#   Parent → Worker (stdin):  JSON line with {"image_b64": "<base64-encoded PNG>"}
#   Worker → Parent (stdout): JSON line with {"ok": true, "result": {...}} or
#                              {"ok": false, "error": "<message>"}
#
# The worker initialises the engine *once* and keeps it alive across many parse
# requests, avoiding repeated model loading.
#
# API auto-detection:
#   - PaddleOCR 3.x (paddleocr >= 3.0): uses PPStructureV3
#   - PaddleOCR 2.x (paddleocr < 3.0):  uses PPStructure + PaddleOCR (legacy)

from __future__ import annotations

# Disable PaddleX's model-hoster connectivity probe BEFORE any paddleocr/paddlex
# import can run. Without this, the probe fires during ``import paddleocr`` at
# module load and can hang for minutes (or forever) on networks where the
# PaddleX endpoints are slow/blocked (e.g. AutoDL), killing the worker before
# it can emit its ready signal. ``setdefault`` preserves any value the parent
# process already injected via ``env=...``. Must precede every paddle-related
# import below (including the transitive import inside _get_paddleocr_major_version).
import os
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import base64
import json
import sys
import traceback

import cv2
import numpy as np


# ── Detect PaddleOCR version ──────────────────────────────────────
def _get_paddleocr_major_version() -> int:
    """Return the major version of the installed paddleocr package."""
    try:
        import paddleocr
        ver = getattr(paddleocr, "__version__", "2.0.0")
        return int(ver.split(".")[0])
    except Exception:
        return 2  # assume legacy


_PADDLEOCR_MAJOR = _get_paddleocr_major_version()


# ── PP-Structure type mapping (shared) ─────────────────────────────
_TYPE_MAP = {
    "text": "text",
    "title": "title",
    "figure": "figure",
    "figure_caption": "figure_caption",
    "figure_title": "figure_caption",   # V3 uses figure_title
    "table": "table",
    "table_caption": "table_caption",
    "table_title": "table_caption",     # V3 uses table_title
    "header": "header",
    "footer": "footer",
    "reference": "reference",
    "equation": "equation",
    "list": "text",
    "seal": "abandon",
    "normal_text": "text",              # V3 order_label
}


# ════════════════════════════════════════════════════════════════════
# PaddleOCR 3.x engine (PPStructureV3)
# ════════════════════════════════════════════════════════════════════

def _init_engines_v3(lang: str, use_gpu: bool, show_log: bool):
    """Initialize PPStructureV3 engine (PaddleOCR >= 3.0)."""
    from paddleocr import PPStructureV3

    engine = PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_seal_recognition=False,
        use_table_recognition=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        lang=lang,
    )
    return engine


def _parse_image_v3(image, engine):
    """Parse image using PPStructureV3.  Returns dict matching legacy format."""
    h_img, w_img = image.shape[:2]

    # PPStructureV3.predict accepts numpy array directly
    results = list(engine.predict(image,
                                  use_doc_orientation_classify=False,
                                  use_doc_unwarping=False))
    if not results:
        return {"text": "", "layout": [], "tables": [],
                "reading_order": [], "content_list": [],
                "middle_json": {"pdf_info": [{"page_idx": 0, "para_blocks": []}]}}

    page = results[0]

    # ── Extract blocks from parsing_res_list ──
    blocks = page.get('parsing_res_list', [])
    ocr_res = page.get('overall_ocr_res', {})
    dt_polys = ocr_res.get('dt_polys', [])
    rec_texts = ocr_res.get('rec_texts', [])
    rec_scores = ocr_res.get('rec_scores', [])

    # Build OCR span list from overall_ocr_res
    ocr_spans = []
    for i in range(len(dt_polys)):
        poly = np.array(dt_polys[i], dtype=np.float32)
        sx1 = float(max(0, poly[:, 0].min()))
        sy1 = float(max(0, poly[:, 1].min()))
        sx2 = float(min(w_img, poly[:, 0].max()))
        sy2 = float(min(h_img, poly[:, 1].max()))
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        text = rec_texts[i] if i < len(rec_texts) else ""
        conf = float(rec_scores[i]) if i < len(rec_scores) else 0.0
        if not text.strip():
            continue
        ocr_spans.append({
            "bbox": [sx1, sy1, sx2, sy2],
            "text": text,
            "confidence": conf,
        })

    layout = []
    content_list = []
    para_blocks = []

    for block in blocks:
        region_type = block.label.lower()
        mapped_type = _TYPE_MAP.get(region_type, region_type)
        if mapped_type in ("abandon", "header", "footer"):
            continue

        bbox = block.bbox
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w_img), x2), min(float(h_img), y2)
        if x2 <= x1 or y2 <= y1:
            continue

        block_text = (block.content or "").strip()

        # Find OCR spans that overlap with this block
        block_spans = []
        for span in ocr_spans:
            sb = span["bbox"]
            # Check overlap: span center inside block bbox
            cx = (sb[0] + sb[2]) / 2
            cy = (sb[1] + sb[3]) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                block_spans.append(span)

        item = {"bbox": [x1, y1, x2, y2], "type": mapped_type, "text": block_text}
        layout.append(item)
        content_list.append(item)

        # Build para_block lines from matched OCR spans
        lines = []
        for span_info in block_spans:
            lines.append({
                "spans": [{"bbox": span_info["bbox"], "content": span_info["text"], "type": mapped_type}]
            })
        if not lines and block_text:
            lines.append({
                "spans": [{"bbox": [x1, y1, x2, y2], "content": block_text, "type": mapped_type}]
            })
        para_blocks.append({"bbox": [x1, y1, x2, y2], "type": mapped_type, "lines": lines})

    middle_json = {"pdf_info": [{"page_idx": 0, "para_blocks": para_blocks}]}
    full_text = " ".join(item["text"] for item in content_list if item["text"])

    return {
        "text": full_text,
        "layout": layout,
        "tables": [it for it in layout if it["type"] == "table"],
        "reading_order": list(range(len(layout))),
        "content_list": content_list,
        "middle_json": middle_json,
    }


# ════════════════════════════════════════════════════════════════════
# PaddleOCR 2.x engine (PPStructure + PaddleOCR)  — legacy
# ════════════════════════════════════════════════════════════════════

def _init_engines_v2(lang: str, use_gpu: bool, show_log: bool):
    """Initialize PPStructure + PaddleOCR engines (PaddleOCR 2.x)."""
    from paddleocr import PPStructure, PaddleOCR

    engine = PPStructure(
        show_log=show_log,
        use_gpu=use_gpu,
        lang=lang,
        layout=True,
        table=False,
        ocr=True,
    )
    ocr_engine = PaddleOCR(
        use_gpu=use_gpu,
        lang=lang,
        show_log=show_log,
    )
    return engine, ocr_engine


def _extract_text_and_spans_v2(region, image, bbox, ocr_engine):
    spans = []
    text_parts = []

    x1_block, y1_block, x2_block, y2_block = bbox
    h_img, w_img = image.shape[:2]

    ocr_results = region.get("res", [])

    if isinstance(ocr_results, list) and len(ocr_results) > 0:
        for item in ocr_results:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            quad_points = item[0]
            text_conf = item[1]

            if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 1:
                text = str(text_conf[0])
                conf = float(text_conf[1]) if len(text_conf) > 1 else 0.0
            elif isinstance(text_conf, str):
                text = text_conf
                conf = 0.0
            else:
                continue

            if not text.strip():
                continue

            if isinstance(quad_points, (list, np.ndarray)) and len(quad_points) >= 4:
                pts = np.array(quad_points, dtype=np.float32)
                sx1, sy1 = float(pts[:, 0].min()), float(pts[:, 1].min())
                sx2, sy2 = float(pts[:, 0].max()), float(pts[:, 1].max())
            else:
                continue

            sx1 = max(0.0, min(sx1, float(w_img)))
            sy1 = max(0.0, min(sy1, float(h_img)))
            sx2 = max(0.0, min(sx2, float(w_img)))
            sy2 = max(0.0, min(sy2, float(h_img)))
            if sx2 <= sx1 or sy2 <= sy1:
                continue

            text_parts.append(text)
            spans.append({"bbox": [sx1, sy1, sx2, sy2], "text": text, "confidence": conf})

    # Fallback OCR on cropped region
    if not spans and ocr_engine is not None:
        region_type = region.get("type", "").lower()
        if region_type in ("text", "title", "list", "reference", "equation"):
            try:
                cx1 = max(0, int(x1_block))
                cy1 = max(0, int(y1_block))
                cx2 = min(w_img, int(x2_block))
                cy2 = min(h_img, int(y2_block))
                if cx2 > cx1 + 5 and cy2 > cy1 + 5:
                    crop = image[cy1:cy2, cx1:cx2]
                    ocr_result = ocr_engine.ocr(crop, cls=True)
                    if ocr_result and ocr_result[0]:
                        for line in ocr_result[0]:
                            if not isinstance(line, (list, tuple)) or len(line) < 2:
                                continue
                            quad = line[0]
                            txt_conf = line[1]
                            text = str(txt_conf[0]) if isinstance(txt_conf, (list, tuple)) else str(txt_conf)
                            if not text.strip():
                                continue
                            pts = np.array(quad, dtype=np.float32)
                            sx1 = float(pts[:, 0].min()) + cx1
                            sy1 = float(pts[:, 1].min()) + cy1
                            sx2 = float(pts[:, 0].max()) + cx1
                            sy2 = float(pts[:, 1].max()) + cy1
                            sx1 = max(0.0, min(sx1, float(w_img)))
                            sy1 = max(0.0, min(sy1, float(h_img)))
                            sx2 = max(0.0, min(sx2, float(w_img)))
                            sy2 = max(0.0, min(sy2, float(h_img)))
                            if sx2 > sx1 and sy2 > sy1:
                                text_parts.append(text)
                                spans.append({"bbox": [sx1, sy1, sx2, sy2], "text": text})
            except Exception:
                pass

    return " ".join(text_parts), spans


def _parse_image_v2(image, engine, ocr_engine):
    h_img, w_img = image.shape[:2]
    result = engine(image)

    layout = []
    content_list = []
    para_blocks = []

    for region in result:
        region_type = region.get("type", "text").lower()
        mapped_type = _TYPE_MAP.get(region_type, region_type)
        if mapped_type in ("abandon", "header", "footer"):
            continue

        bbox_raw = region.get("bbox", [])
        if not bbox_raw or len(bbox_raw) < 4:
            continue

        x1, y1, x2, y2 = float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3])
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w_img), x2), min(float(h_img), y2)
        if x2 <= x1 or y2 <= y1:
            continue

        block_text, spans = _extract_text_and_spans_v2(region, image, [x1, y1, x2, y2], ocr_engine)

        item = {"bbox": [x1, y1, x2, y2], "type": mapped_type, "text": block_text}
        layout.append(item)
        content_list.append(item)

        lines = []
        for span_info in spans:
            lines.append({
                "spans": [{"bbox": span_info["bbox"], "content": span_info["text"], "type": mapped_type}]
            })
        if not lines and block_text:
            lines.append({
                "spans": [{"bbox": [x1, y1, x2, y2], "content": block_text, "type": mapped_type}]
            })
        para_blocks.append({"bbox": [x1, y1, x2, y2], "type": mapped_type, "lines": lines})

    middle_json = {"pdf_info": [{"page_idx": 0, "para_blocks": para_blocks}]}
    full_text = " ".join(item["text"] for item in content_list if item["text"])

    return {
        "text": full_text,
        "layout": layout,
        "tables": [it for it in layout if it["type"] == "table"],
        "reading_order": list(range(len(layout))),
        "content_list": content_list,
        "middle_json": middle_json,
    }


# ════════════════════════════════════════════════════════════════════
# Main worker loop
# ════════════════════════════════════════════════════════════════════

def main():
    import logging
    logging.disable(logging.WARNING)  # suppress paddle/ppocr warnings

    # Read init params from first line
    init_line = sys.stdin.readline().strip()
    if not init_line:
        sys.exit(1)
    init_cfg = json.loads(init_line)
    lang = init_cfg.get("lang", "en")
    use_gpu = init_cfg.get("use_gpu", True)
    show_log = init_cfg.get("show_log", False)

    try:
        if _PADDLEOCR_MAJOR >= 3:
            engine = _init_engines_v3(lang, use_gpu, show_log)
            ocr_engine = None
            api_version = "v3"
        else:
            engine, ocr_engine = _init_engines_v2(lang, use_gpu, show_log)
            api_version = "v2"
        sys.stdout.write(json.dumps({"ok": True, "msg": "ready", "api": api_version}) + "\n")
        sys.stdout.flush()
    except Exception as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": f"init failed: {exc}"}) + "\n")
        sys.stdout.flush()
        sys.exit(1)

    # Main loop: read image, parse, return result
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "__EXIT__":
            break

        try:
            req = json.loads(line)
            img_b64 = req["image_b64"]
            img_bytes = base64.b64decode(img_b64)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")

            if _PADDLEOCR_MAJOR >= 3:
                result = _parse_image_v3(image, engine)
            else:
                result = _parse_image_v2(image, engine, ocr_engine)

            sys.stdout.write(json.dumps({"ok": True, "result": result}) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            tb = traceback.format_exc()
            sys.stdout.write(json.dumps({"ok": False, "error": str(exc), "traceback": tb}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
