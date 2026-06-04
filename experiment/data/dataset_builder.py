from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .masks import compute_anchor_mask, compute_content_mask


CATEGORY_MAP = {1: "Text", 2: "Title", 3: "List", 4: "Table", 5: "Figure"}

DOCLAYNET_CATEGORY_MAP = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
}


def resize_long_edge(image: np.ndarray, target: int = 1024) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = target / max(h, w)
    if abs(scale - 1.0) < 1e-6:
        return image, 1.0
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def compute_layout_entropy(annotations: list[dict]) -> float:
    if not annotations:
        return 0.0
    total_area = sum(max(1.0, ann.get("area", 1.0)) for ann in annotations)
    if total_area <= 0:
        return 0.0
    area_by_cat = defaultdict(float)
    for ann in annotations:
        area_by_cat[ann["category"]] += float(max(1.0, ann.get("area", 1.0)))
    probs = [v / total_area for v in area_by_cat.values()]
    return float(-sum(p * np.log2(p) for p in probs if p > 0))


def load_publaynet_annotations(val_json_path: Path) -> tuple[dict[str, list[dict]], dict[str, str]]:
    with open(val_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    ann_map = defaultdict(list)
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        x, y, bw, bh = ann["bbox"]
        item = {
            "bbox": [float(x), float(y), float(x + bw), float(y + bh)],
            "category": CATEGORY_MAP.get(int(ann["category_id"]), str(ann["category_id"])),
            "area": float(ann.get("area", bw * bh)),
        }
        ann_map[id_to_name[image_id]].append(item)

    return dict(ann_map), {str(k): v for k, v in id_to_name.items()}


def _rescale_annotations(annotations: list[dict], scale: float) -> list[dict]:
    out = []
    for ann in annotations:
        x1, y1, x2, y2 = ann["bbox"]
        out.append(
            {
                **ann,
                "bbox": [x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                "area": ann.get("area", (x2 - x1) * (y2 - y1)) * (scale * scale),
            }
        )
    return out


def _complexity_level(n_blocks: int, h_layout: float) -> str:
    if n_blocks <= 8 and h_layout < 1.5:
        return "simple"
    if n_blocks > 18 or h_layout >= 2.5:
        return "complex"
    return "medium"


def _compute_all_candidates(data_root: Path, delta: int = 5) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    """Compute complexity metadata for ALL images in the dataset.

    Returns (candidates_df, ann_map).
    candidates_df has columns: file_name, N_blocks, H_layout, D_gap, level.
    """
    val_json = data_root / "val.json"
    image_dir = data_root / "val"
    ann_map, _ = load_publaynet_annotations(val_json)

    candidates = []
    for file_name, anns in tqdm(ann_map.items(), desc="Compute complexity"):
        img_path = image_dir / file_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        n_blocks = len(anns)
        h_layout = compute_layout_entropy(anns)
        m_anchor = compute_anchor_mask(anns, (h, w), delta=delta)
        d_gap = float(m_anchor.sum() / (h * w))
        candidates.append(
            {
                "file_name": file_name,
                "N_blocks": n_blocks,
                "H_layout": h_layout,
                "D_gap": d_gap,
                "level": _complexity_level(n_blocks, h_layout),
            }
        )

    return pd.DataFrame(candidates), ann_map


def load_images_by_filenames(
    data_root: Path,
    filenames: list[str],
    ann_map: dict[str, list[dict]],
    delta: int = 5,
    long_edge: int = 1024,
) -> list[dict]:
    """Load and preprocess specific images by filename."""
    image_dir = data_root / "val"
    dataset = []
    for file_name in tqdm(filenames, desc="Load backfill images"):
        img_path = image_dir / file_name
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        anns = ann_map.get(file_name, [])
        image_resized, scale = resize_long_edge(image, target=long_edge)
        anns_rescaled = _rescale_annotations(anns, scale)
        h, w = image_resized.shape[:2]
        m_content = compute_content_mask(anns_rescaled, (h, w))
        m_anchor = compute_anchor_mask(anns_rescaled, (h, w), delta=delta)
        m_bg = (1 - np.clip(m_content + m_anchor, 0, 1)).astype(np.uint8)

        n_blocks = len(anns_rescaled)
        h_layout = compute_layout_entropy(anns_rescaled)
        d_gap = float(m_anchor.sum() / (h * w))

        dataset.append(
            {
                "image_id": file_name,
                "image": image_resized,
                "annotation": anns_rescaled,
                "M_content": m_content,
                "M_anchor": m_anchor,
                "M_bg": m_bg,
                "complexity": {
                    "N_blocks": n_blocks,
                    "H_layout": h_layout,
                    "D_gap": d_gap,
                    "level": _complexity_level(n_blocks, h_layout),
                },
                "H": h,
                "W": w,
            }
        )
    return dataset


def build_stratified_dataset(data_root: Path, n_total: int = 1000, delta: int = 5, seed: int = 42, long_edge: int = 1024) -> list[dict]:
    df, ann_map = _compute_all_candidates(data_root, delta=delta)
    image_dir = data_root / "val"
    rng = np.random.default_rng(seed)

    n_simple = int(round(n_total * 0.3))
    n_medium = int(round(n_total * 0.4))
    n_complex = n_total - n_simple - n_medium
    target = {"simple": n_simple, "medium": n_medium, "complex": n_complex}

    sampled_files = []
    for level, n in target.items():
        pool = df[df["level"] == level]["file_name"].tolist()
        if len(pool) <= n:
            sampled = pool
        else:
            idx = rng.choice(len(pool), size=n, replace=False)
            sampled = [pool[i] for i in idx]
        sampled_files.extend(sampled)

    dataset = load_images_by_filenames(data_root, sampled_files, ann_map, delta=delta, long_edge=long_edge)
    return dataset


def build_stratified_dataset_with_pool(
    data_root: Path, n_total: int = 1000, delta: int = 5, seed: int = 42, long_edge: int = 1024,
) -> tuple[list[dict], pd.DataFrame, dict[str, list[dict]]]:
    """Like build_stratified_dataset but also returns full candidate pool info.

    Returns (dataset, candidates_df, ann_map) so the caller can backfill
    after filtering.
    """
    df, ann_map = _compute_all_candidates(data_root, delta=delta)
    rng = np.random.default_rng(seed)

    n_simple = int(round(n_total * 0.3))
    n_medium = int(round(n_total * 0.4))
    n_complex = n_total - n_simple - n_medium
    target = {"simple": n_simple, "medium": n_medium, "complex": n_complex}

    sampled_files = []
    for level, n in target.items():
        pool = df[df["level"] == level]["file_name"].tolist()
        if len(pool) <= n:
            sampled = pool
        else:
            idx = rng.choice(len(pool), size=n, replace=False)
            sampled = [pool[i] for i in idx]
        sampled_files.extend(sampled)

    dataset = load_images_by_filenames(data_root, sampled_files, ann_map, delta=delta, long_edge=long_edge)
    return dataset, df, ann_map


def load_doclaynet_annotations(
    val_json_path: Path,
) -> tuple[dict[str, list[dict]], dict[str, str], dict[str, str]]:
    """Load DocLayNet COCO JSON.

    Returns (ann_map, id_to_name, id_to_doc_category).
    ann_map values have bbox in [x1,y1,x2,y2] format.
    """
    with open(val_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_doc_cat = {img["id"]: img.get("doc_category", "unknown") for img in coco["images"]}
    ann_map = defaultdict(list)
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        x, y, bw, bh = ann["bbox"]
        item = {
            "bbox": [float(x), float(y), float(x + bw), float(y + bh)],
            "category": DOCLAYNET_CATEGORY_MAP.get(int(ann["category_id"]), str(ann["category_id"])),
            "area": float(ann.get("area", bw * bh)),
        }
        ann_map[id_to_name[image_id]].append(item)

    return dict(ann_map), {str(k): v for k, v in id_to_name.items()}, {str(k): v for k, v in id_to_doc_cat.items()}


def _selected_dataset_error(selected_root: Path, source: str) -> str:
    return (
        f"Local selected dataset for {source!r} is missing or incomplete under "
        f"{selected_root / source}. This public repository does not redistribute "
        "PubLayNet/DocLayNet images or COCO annotation JSON files. Download the "
        "datasets from their official sources, confirm their licenses, and prepare:\n"
        f"  {selected_root / source / 'selected_600.json'}\n"
        f"  {selected_root / source / 'images'}"
    )


def _require_selected_dataset_files(selected_root: Path, source: str) -> tuple[Path, Path]:
    sel_json = selected_root / source / "selected_600.json"
    image_dir = selected_root / source / "images"
    if not sel_json.is_file() or not image_dir.is_dir():
        raise FileNotFoundError(_selected_dataset_error(selected_root, source))
    if not any(p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in image_dir.iterdir()):
        raise FileNotFoundError(_selected_dataset_error(selected_root, source))
    return sel_json, image_dir


def build_selected_dataset(
    selected_root: Path,
    source: str,
    delta: int = 5,
    long_edge: int = 1024,
) -> list[dict]:
    """Load a pre-sampled selected dataset from *data/selected/{source}/*.

    Reads ``selected_600.json`` (COCO format) and images from ``images/``.
    Returns the same list[dict] schema as ``build_stratified_dataset``.
    """
    sel_json, image_dir = _require_selected_dataset_files(selected_root, source)

    with open(sel_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build category map from JSON
    cat_map = {c["id"]: c["name"] for c in coco.get("categories", [])}
    if not cat_map:
        cat_map = CATEGORY_MAP if source == "publaynet" else DOCLAYNET_CATEGORY_MAP

    id_to_img = {img["id"]: img for img in coco["images"]}
    ann_map: dict[str, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        fname = id_to_img[ann["image_id"]]["file_name"]
        x, y, bw, bh = ann["bbox"]
        ann_map[fname].append({
            "bbox": [float(x), float(y), float(x + bw), float(y + bh)],
            "category": cat_map.get(int(ann["category_id"]), str(ann["category_id"])),
            "area": float(ann.get("area", bw * bh)),
        })

    dataset = []
    for img_record in tqdm(coco["images"], desc=f"Load selected/{source}"):
        fname = img_record["file_name"]
        img_path = image_dir / fname
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        anns = ann_map.get(fname, [])
        image_resized, scale = resize_long_edge(image, target=long_edge)
        anns_rescaled = _rescale_annotations(anns, scale)
        h, w = image_resized.shape[:2]
        m_content = compute_content_mask(anns_rescaled, (h, w))
        m_anchor = compute_anchor_mask(anns_rescaled, (h, w), delta=delta)
        m_bg = (1 - np.clip(m_content + m_anchor, 0, 1)).astype(np.uint8)

        n_blocks = len(anns_rescaled)
        h_layout = compute_layout_entropy(anns_rescaled)
        d_gap = float(m_anchor.sum() / (h * w))

        dataset.append({
            "image_id": fname,
            "image": image_resized,
            "annotation": anns_rescaled,
            "M_content": m_content,
            "M_anchor": m_anchor,
            "M_bg": m_bg,
            "complexity": {
                "N_blocks": n_blocks,
                "H_layout": h_layout,
                "D_gap": d_gap,
                "level": _complexity_level(n_blocks, h_layout),
            },
            "H": h,
            "W": w,
            "data_source": source,
            "doc_category": img_record.get("doc_category", ""),
        })

    if not dataset:
        raise FileNotFoundError(_selected_dataset_error(selected_root, source))

    return dataset


def build_combined_dataset(
    selected_root: Path,
    delta: int = 5,
    long_edge: int = 1024,
) -> list[dict]:
    """Load PubLayNet + DocLayNet from *data/selected/* and merge into one list."""
    ds_pub = build_selected_dataset(selected_root, "publaynet", delta=delta, long_edge=long_edge)
    ds_doc = build_selected_dataset(selected_root, "doclaynet", delta=delta, long_edge=long_edge)
    return ds_pub + ds_doc


def build_shared_dataset(
    shared_index_path: Path,
    data_root: Path,
    delta: int = 5,
    long_edge: int = 1024,
    selected_root: Path | None = None,
) -> list[dict]:
    """Load the exact images listed in a shared evaluation index.

    The shared index is generated by ``generate_shared_dataset.py`` and
    guarantees that every image is valid for all target parsers.  This
    bypasses stratified sampling, backfill, and trim entirely.

    Supports both single-source (PubLayNet) and dual-source (PubLayNet +
    DocLayNet) indices.  When ``data_mode`` in the index is ``"selected"``,
    loads via ``build_combined_dataset`` and filters to the listed images.
    """
    import logging

    with open(shared_index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    metadata = index.get("metadata", {})
    data_mode = metadata.get("data_mode", "legacy")
    # V2 JSON: detect selected mode from "selected_root" key even if data_mode absent
    if data_mode == "legacy" and "selected_root" in metadata:
        data_mode = "selected"
    target_ids = {entry["image_id"] for entry in index["images"]}

    if data_mode == "selected" and selected_root is not None:
        raw = metadata.get("selected_root", str(selected_root))
        sel_root = Path(raw.replace("\\", "/"))
        if not sel_root.exists():
            sel_root = selected_root
        all_items = build_combined_dataset(sel_root, delta=delta, long_edge=long_edge)
        dataset = [item for item in all_items if item["image_id"] in target_ids]
    else:
        filenames = [entry["image_id"] for entry in index["images"]]
        val_json = data_root / "val.json"
        ann_map, _ = load_publaynet_annotations(val_json)
        dataset = load_images_by_filenames(
            data_root, filenames, ann_map, delta=delta, long_edge=long_edge,
        )

    loaded_ids = {item["image_id"] for item in dataset}
    missing = [f for f in target_ids if f not in loaded_ids]
    if missing:
        logging.warning(
            "build_shared_dataset: %d/%d images not found on disk: %s",
            len(missing), len(target_ids), sorted(missing)[:5],
        )

    dataset.sort(key=lambda x: x["image_id"])
    return dataset
