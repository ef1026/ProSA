from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from experiment.engine import AttackEngine
from experiment_add.shared.data.load_manifest import load_manifest
from experiment_add.shared.perturb.perturb_metadata import make_metadata_record
from experiment_add.shared.perturb.select_probe_conditions import select_probe_conditions
from experiment_add.shared.utils.io import atomic_write_jsonl, ensure_dir, read_jsonl, safe_read_json
from experiment_add.shared.utils.path_manager import PathManager


CONDITIONS = ["area_matched_erasure", "structural_probe", "large_area_erasure"]
GENERATION_ORDER = ["structural_probe", "area_matched_erasure", "large_area_erasure"]


def _resolve(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _blocks_to_data(clean_record: dict[str, Any], image: np.ndarray) -> dict[str, Any]:
    h, w = image.shape[:2]
    anns = []
    mask = np.zeros((h, w), dtype=np.uint8)
    for block in clean_record.get("blocks", []):
        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = 1
        anns.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "category": str(block.get("layout_type", "Text")).title(), "area": float((x2 - x1) * (y2 - y1)), "block_id": block.get("block_id"), "layout_type": block.get("layout_type")})
    return {"annotation": anns, "M_content": mask, "M_anchor": mask, "H": h, "W": w}


def _target_block(clean_record: dict[str, Any], center: tuple[int, int] | None) -> dict[str, Any] | None:
    blocks = clean_record.get("blocks", [])
    if not blocks:
        return None
    if center is None:
        return blocks[0]
    cx, cy = center
    best = None
    best_dist = 1e18
    for block in blocks:
        b = block.get("bbox") or [0, 0, 0, 0]
        bx = (float(b[0]) + float(b[2])) / 2
        by = (float(b[1]) + float(b[3])) / 2
        d = (bx - cx) ** 2 + (by - cy) ** 2
        if d < best_dist:
            best_dist = d
            best = block
    return best


def _area_matched_plan(target_tor: float, h: int, w: int) -> list[dict[str, Any]]:
    area_ratio = max(0.0002, min(0.05, target_tor * h * w / (min(h, w) ** 2)))
    return [{"probe_type": "P4", "params": {"area_ratio": area_ratio, "beta": 1.0}, "target_strategy": "random", "target_location": None, "reason": "area_matched_erasure"}]


def _summarize(records: list[dict[str, Any]], root: Path) -> Path:
    summary_path = root / "experiment_add/outputs/shared/perturbed_pages/perturb_summary.csv"
    ensure_dir(summary_path.parent)
    by_cond: dict[str, list[dict[str, Any]]] = {c: [r for r in records if r["condition"] == c] for c in CONDITIONS}
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "image_count", "metadata_count", "mean_TOR", "median_TOR", "min_TOR", "max_TOR", "std_TOR"])
        writer.writeheader()
        for cond, items in by_cond.items():
            tors = [float(r["TOR"]) for r in items]
            if tors:
                mean = float(np.mean(tors)); median = float(np.median(tors)); std = float(np.std(tors)); mn = min(tors); mx = max(tors)
            else:
                mean = median = std = mn = mx = 0.0
            image_dir = root / "experiment_add/outputs/shared/perturbed_pages" / cond / "images"
            writer.writerow({"condition": cond, "image_count": len(list(image_dir.glob("*.png"))), "metadata_count": len(items), "mean_TOR": mean, "median_TOR": median, "min_TOR": mn, "max_TOR": mx, "std_TOR": std})
    return summary_path


def generate_perturbed_pages(config_path: str | Path, debug: bool = False) -> dict[str, Any]:
    pm = PathManager("experiment_add/configs/base.yaml", create_dirs=True)
    root = pm.project_root
    selection_path = root / "experiment_add/outputs/shared/perturbed_pages/probe_selection.json"
    selection = safe_read_json(selection_path)
    if not selection:
        selection = select_probe_conditions(config_path, debug=debug)
    manifest = load_manifest(pm.page_manifest_debug20 if debug else pm.page_manifest_500)
    clean = {r["page_id"]: r for r in read_jsonl(pm.clean_parse_merged_path("mineru"))}
    engine = AttackEngine()
    all_meta: list[dict[str, Any]] = []
    failed_path = pm.shared_log_root / "failed_pages.csv"
    failed_rows = []
    seed_base = 42
    for i, row in enumerate(manifest):
        page_id = row["page_id"]
        image_path = _resolve(row["image_path"], root)
        image = cv2.imread(str(image_path))
        clean_record = clean.get(page_id, {})
        if image is None or not clean_record:
            failed_rows.append({"page_id": page_id, "reason": "missing_image_or_clean_parse"})
            continue
        data = _blocks_to_data(clean_record, image)
        structural_actual_tor = float(selection["area_matched_erasure"].get("target_tor", 0.003))
        for condition in GENERATION_ORDER:
            out_dir = pm.perturbed_images_dir(condition)
            ensure_dir(out_dir)
            out_path = out_dir / f"{Path(page_id).stem}.png"
            rng = np.random.default_rng(seed_base + i)
            if condition == "structural_probe":
                plan = selection["structural_probe"]["plan"]
                probe_id = selection["structural_probe"]["probe_id"]; family = "structural"
            elif condition == "area_matched_erasure":
                plan = _area_matched_plan(structural_actual_tor, image.shape[0], image.shape[1])
                probe_id = "P4_area_matched"; family = "area_matched_erasure"
            else:
                plan = selection["large_area_erasure"]["plan"]
                probe_id = "A08"; family = "large_area_erasure"
            try:
                adv, mask, used_plan, elog = engine.execute(image, data, policy=None, rng=rng, override_plan=plan)
                cv2.imwrite(str(out_path), adv)
                center = tuple(elog[0].get("center", (0, 0))) if elog else None
                target = _target_block(clean_record, center)
                meta = make_metadata_record(page_id, condition, probe_id, family, row["image_path"], str(out_path.relative_to(root)), image.shape[1], image.shape[0], mask, target, json.dumps(elog, ensure_ascii=False), seed_base + i)
                if condition == "structural_probe":
                    structural_actual_tor = float(meta["TOR"])
                all_meta.append(meta)
            except Exception as exc:
                failed_rows.append({"page_id": page_id, "condition": condition, "reason": str(exc)})
    for condition in CONDITIONS:
        path = pm.perturb_metadata_path(condition)
        atomic_write_jsonl(path, [r for r in all_meta if r["condition"] == condition])
    merged_path = root / "experiment_add/outputs/shared/perturbed_pages/merged_perturb_metadata.jsonl"
    atomic_write_jsonl(merged_path, all_meta)
    summary_path = _summarize(all_meta, root)
    if failed_rows:
        with failed_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in failed_rows for k in r}))
            writer.writeheader(); writer.writerows(failed_rows)
    return {"metadata_count": len(all_meta), "failed_count": len(failed_rows), "summary_path": str(summary_path), "merged_metadata_path": str(merged_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment_add/configs/perturb.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    result = generate_perturbed_pages(args.config, debug=args.debug)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
