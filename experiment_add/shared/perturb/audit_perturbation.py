from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from PIL import Image

from experiment_add.shared.data.load_manifest import load_manifest
from experiment_add.shared.utils.io import ensure_dir, read_yaml, write_text
from experiment_add.shared.utils.path_manager import PathManager


CONDITIONS = ["area_matched_erasure", "structural_probe", "large_area_erasure"]
REQUIRED_META_FIELDS = [
    "page_id",
    "condition",
    "probe_id",
    "probe_family",
    "image_path",
    "perturbed_image_path",
    "page_width",
    "page_height",
    "support_bbox",
    "support_mask_area",
    "TOR",
    "target_block_id",
    "target_layout_type",
    "placement_info",
    "random_seed",
]


def _infer_root(config_path: Path) -> Path:
    p = config_path.resolve()
    if p.parent.name == "configs" and p.parent.parent.name == "experiment_add":
        return p.parent.parent.parent
    return Path.cwd()


def _resolve_path(root: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def _metadata_paths(root: Path, condition: str) -> list[Path]:
    base = root / "experiment_add/outputs/shared/perturbed_pages" / condition
    return [base / "perturb_metadata.jsonl", base / "metadata.jsonl"]


def _read_jsonl_existing(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errs: list[str] = []
    if not path.exists():
        return rows, errs
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                errs.append(f"{path.name}:{i}: {e}")
    return rows, errs


def _valid_bbox_shape(bbox: Any) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


def _bbox_out_of_range(bbox: list[Any], w: int, h: int) -> bool:
    tol_x = max(10.0, w * 0.02)
    tol_y = max(10.0, h * 0.02)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return x1 < -tol_x or y1 < -tol_y or x2 > w + tol_x or y2 > h + tol_y


def _tor_mask_mismatch(tor: float, mask_area: int, w: int, h: int) -> bool:
    if w <= 0 or h <= 0:
        return True
    expected = mask_area / (w * h)
    diff = abs(float(tor) - expected)
    if diff <= 1e-5:
        return False
    scale = max(abs(float(tor)), expected, 1e-12)
    return (diff / scale) > 0.01


def _perturbed_image_path(images_dir: Path, page_id: str) -> Path:
    return images_dir / f"{Path(page_id).stem}.png"


def audit(debug: bool, root: Path, pm: PathManager) -> dict[str, Any]:
    manifest_rows = load_manifest(pm.page_manifest_debug20 if debug else pm.page_manifest_500)
    expected_ids = [r["page_id"] for r in manifest_rows]
    expected_set = set(expected_ids)
    n_expect = len(expected_ids)
    dup_manifest = sum(1 for _pid, c in Counter(expected_ids).items() if c > 1)
    manifest_paths_by_id = {r["page_id"]: str(r["image_path"]) for r in manifest_rows}
    stem_set = {Path(pid).stem for pid in expected_ids}

    per_condition_images: dict[str, dict[str, Path]] = {}
    per_condition_meta: dict[str, list[dict[str, Any]]] = {}
    meta_parse_errors: list[str] = []

    merged_primary = root / "experiment_add/outputs/shared/perturbed_pages/merged_perturb_metadata.jsonl"
    merged_rows, merged_errs = _read_jsonl_existing(merged_primary)
    meta_parse_errors.extend(merged_errs)
    if not merged_primary.exists():
        meta_parse_errors.append(f"merged metadata file missing: {merged_primary}")
    elif len(merged_rows) != n_expect * 3:
        meta_parse_errors.append(
            f"merged metadata record count {len(merged_rows)} != expected {n_expect * 3}"
        )

    agg: dict[str, int] = defaultdict(int)
    tors_by_cond: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    null_target_reports: dict[str, int] = defaultdict(int)

    for condition in CONDITIONS:
        img_dir = pm.perturbed_images_dir(condition)
        stem_to_path: dict[str, Path] = {}
        if img_dir.exists():
            for p in sorted(img_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    stem_to_path[p.stem] = p
                    if p.stem not in stem_set:
                        agg["extra_image_files_count"] += 1

        mapped: dict[str, Path] = {}
        for pid in expected_ids:
            cand = _perturbed_image_path(img_dir, pid)
            if cand.exists():
                mapped[pid] = cand
            else:
                alt = stem_to_path.get(Path(pid).stem)
                if alt is not None:
                    mapped[pid] = alt
        per_condition_images[condition] = mapped

        meta_loaded: list[dict[str, Any]] = []
        tried: list[str] = []
        for mp in _metadata_paths(root, condition):
            tried.append(f"{mp.name}[{'exists' if mp.exists() else 'missing'}]")
            rows, errs = _read_jsonl_existing(mp)
            meta_parse_errors.extend(errs)
            if rows:
                meta_loaded = rows
                break
        if not meta_loaded:
            meta_parse_errors.append(
                f"no usable metadata rows for {condition}; tried: {', '.join(tried)}"
            )
        per_condition_meta[condition] = meta_loaded

    def scan_metadata(condition: str, rows: list[dict[str, Any]]) -> None:
        for rec in rows:
            if rec.get("condition") != condition:
                agg["wrong_condition_in_row"] += 1

            for field in REQUIRED_META_FIELDS:
                if field not in rec:
                    agg["missing_required_field"] += 1

            if rec.get("target_block_id") is None:
                null_target_reports["target_block_id"] += 1
            if rec.get("target_layout_type") is None:
                null_target_reports["target_layout_type"] += 1

            w, h = rec.get("page_width"), rec.get("page_height")
            try:
                iw, ih = int(w), int(h)
            except (TypeError, ValueError):
                iw = ih = -1

            clean_path = _resolve_path(root, str(rec.get("image_path", "")))
            pert_path = _resolve_path(root, str(rec.get("perturbed_image_path", "")))

            if not pert_path.exists():
                agg["perturbed_path_missing_in_metadata"] += 1
            if not clean_path.exists():
                agg["clean_path_missing"] += 1

            bbox = rec.get("support_bbox")
            if bbox is None or not _valid_bbox_shape(bbox):
                agg["invalid_support_bbox_count"] += 1
            elif iw > 0 and ih > 0 and _bbox_out_of_range(bbox, iw, ih):
                agg["support_bbox_out_of_range_count"] += 1

            try:
                sma = int(rec["support_mask_area"])
            except (KeyError, TypeError, ValueError):
                sma = -1
            if sma <= 0:
                agg["invalid_support_mask_area_count"] += 1

            tor_v: float | None = None
            try:
                tor_v = float(rec["TOR"])
            except (KeyError, TypeError, ValueError):
                agg["invalid_TOR_count"] += 1
            else:
                if not (tor_v > 0 and tor_v <= 1):
                    agg["invalid_TOR_count"] += 1
                else:
                    tors_by_cond[condition].append(tor_v)
                    if iw > 0 and ih > 0 and sma >= 0:
                        if _tor_mask_mismatch(tor_v, sma, iw, ih):
                            agg["TOR_mask_area_mismatch_count"] += 1

            if clean_path.exists() and iw > 0 and ih > 0:
                try:
                    with Image.open(clean_path) as cim:
                        cw, ch = cim.size
                    if cw != iw or ch != ih:
                        agg["metadata_clean_dimension_mismatch_count"] += 1
                except OSError:
                    agg["clean_image_unreadable_count"] += 1

    for condition in CONDITIONS:
        scan_metadata(condition, per_condition_meta[condition])

    merged_pair_counts = Counter((str(r.get("page_id")), str(r.get("condition"))) for r in merged_rows)
    duplicate_page_condition_count = sum(c - 1 for c in merged_pair_counts.values() if c > 1)
    merged_by_page = Counter(str(r.get("page_id")) for r in merged_rows)
    merged_by_cond = Counter(str(r.get("condition")) for r in merged_rows)
    bad_page_coverage = sum(1 for pid in expected_set if merged_by_page[pid] != 3)
    bad_cond_coverage = sum(1 for c in CONDITIONS if merged_by_cond[c] != n_expect)

    for condition in CONDITIONS:
        mapped = per_condition_images[condition]
        for pid in expected_ids:
            clean_p = _resolve_path(root, manifest_paths_by_id[pid])
            if pid not in mapped:
                agg["missing_image_count"] += 1
                continue
            pth = mapped[pid]
            try:
                if pth.stat().st_size == 0:
                    agg["invalid_image_count"] += 1
                    continue
                with Image.open(pth) as im:
                    im.verify()
                with Image.open(pth) as pim:
                    pw, ph = pim.size
                if clean_p.exists():
                    with Image.open(clean_p) as cim:
                        if cim.size != (pw, ph):
                            agg["image_size_mismatch_count"] += 1
            except OSError:
                agg["invalid_image_count"] += 1

    image_counts = {c: len(per_condition_images[c]) for c in CONDITIONS}
    meta_counts = {c: len(per_condition_meta[c]) for c in CONDITIONS}

    def stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        return {
            "mean": float(statistics.mean(vals)),
            "median": float(statistics.median(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        }

    st_area = stats(tors_by_cond["area_matched_erasure"])
    st_struct = stats(tors_by_cond["structural_probe"])
    st_large = stats(tors_by_cond["large_area_erasure"])

    mean_s = st_struct["mean"]
    rel_gap = 0.0 if mean_s <= 0 else abs(st_area["mean"] - mean_s) / mean_s * 100.0

    large_mean = st_large["mean"]
    large_ok = 0.15 <= large_mean <= 0.25

    meta_ids_by_cond = {c: {str(r.get("page_id")) for r in per_condition_meta[c]} for c in CONDITIONS}
    meta_counts_per_id = {c: Counter(str(r.get("page_id")) for r in per_condition_meta[c]) for c in CONDITIONS}

    merged_duplicate_pair_keys = sum(1 for c in merged_pair_counts.values() if c > 1)

    checks = [
        image_counts["area_matched_erasure"] == n_expect,
        image_counts["structural_probe"] == n_expect,
        image_counts["large_area_erasure"] == n_expect,
        meta_counts["area_matched_erasure"] == n_expect,
        meta_counts["structural_probe"] == n_expect,
        meta_counts["large_area_erasure"] == n_expect,
        len(merged_rows) == n_expect * 3,
        duplicate_page_condition_count == 0,
        bad_page_coverage == 0,
        bad_cond_coverage == 0,
        dup_manifest == 0,
        agg["invalid_support_bbox_count"] == 0,
        agg["support_bbox_out_of_range_count"] == 0,
        agg["invalid_support_mask_area_count"] == 0,
        agg["invalid_TOR_count"] == 0,
        agg["TOR_mask_area_mismatch_count"] == 0,
        agg["perturbed_path_missing_in_metadata"] == 0,
        rel_gap <= 20.0 + 1e-6,
        large_ok,
        agg["invalid_image_count"] == 0,
        agg["image_size_mismatch_count"] == 0,
        agg["missing_image_count"] == 0,
        agg["wrong_condition_in_row"] == 0,
        agg["missing_required_field"] == 0,
        agg["metadata_clean_dimension_mismatch_count"] == 0,
        agg["extra_image_files_count"] == 0,
        not meta_parse_errors,
        agg["clean_path_missing"] == 0,
        all(len(tors_by_cond[c]) == n_expect for c in CONDITIONS),
        all(expected_set <= meta_ids_by_cond[c] for c in CONDITIONS),
        all(all(meta_counts_per_id[c][pid] == 1 for pid in expected_set) for c in CONDITIONS if meta_counts_per_id[c]),
    ]

    ready = all(checks)

    return {
        "n_expect": n_expect,
        "merged_path": str(merged_primary),
        "merged_metadata_count": len(merged_rows),
        "image_counts": image_counts,
        "metadata_counts": meta_counts,
        "st_area": st_area,
        "st_struct": st_struct,
        "st_large": st_large,
        "area_structural_TOR_relative_gap": rel_gap,
        "large_area_TOR_in_expected_range": "YES" if large_ok else "NO",
        "agg": dict(agg),
        "duplicate_page_condition_count": duplicate_page_condition_count,
        "merged_duplicate_pair_keys": merged_duplicate_pair_keys,
        "merged_bad_page_coverage_count": bad_page_coverage,
        "merged_bad_condition_coverage_count": bad_cond_coverage,
        "null_target_reports": dict(null_target_reports),
        "meta_parse_errors": meta_parse_errors,
        "ready_for_perturbed_parsing": "YES" if ready else "NO",
        "ready_prompt7": "YES" if ready else "NO",
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    ic = result["image_counts"]
    mc = result["metadata_counts"]
    sa, ss, sl = result["st_area"], result["st_struct"], result["st_large"]
    ag = result["agg"]

    lines = [
        "# Perturbation debug20 audit",
        "",
        "## Summary metrics",
        "",
        f"- `image_count_area_matched`: {ic['area_matched_erasure']}",
        f"- `image_count_structural`: {ic['structural_probe']}",
        f"- `image_count_large_area`: {ic['large_area_erasure']}",
        "",
        f"- `metadata_count_area_matched`: {mc['area_matched_erasure']}",
        f"- `metadata_count_structural`: {mc['structural_probe']}",
        f"- `metadata_count_large_area`: {mc['large_area_erasure']}",
        "",
        f"- `merged_metadata_count`: {result['merged_metadata_count']}",
        "",
        f"- `mean_TOR_area_matched`: {sa['mean']:.8f}",
        f"- `mean_TOR_structural`: {ss['mean']:.8f}",
        f"- `mean_TOR_large_area`: {sl['mean']:.8f}",
        "",
        f"- `median_TOR_area_matched`: {sa['median']:.8f}",
        f"- `median_TOR_structural`: {ss['median']:.8f}",
        f"- `median_TOR_large_area`: {sl['median']:.8f}",
        "",
        f"- `min_TOR_area_matched`: {sa['min']:.8f}",
        f"- `min_TOR_structural`: {ss['min']:.8f}",
        f"- `min_TOR_large_area`: {sl['min']:.8f}",
        "",
        f"- `max_TOR_area_matched`: {sa['max']:.8f}",
        f"- `max_TOR_structural`: {ss['max']:.8f}",
        f"- `max_TOR_large_area`: {sl['max']:.8f}",
        "",
        f"- `std_TOR_area_matched`: {sa['std']:.8f}",
        f"- `std_TOR_structural`: {ss['std']:.8f}",
        f"- `std_TOR_large_area`: {sl['std']:.8f}",
        "",
        f"- `area_structural_TOR_relative_gap` (%): {result['area_structural_TOR_relative_gap']:.4f}",
        f"- `large_area_TOR_in_expected_range`: {result['large_area_TOR_in_expected_range']}",
        "",
        "## Integrity counts",
        "",
        f"- `invalid_support_bbox_count`: {ag.get('invalid_support_bbox_count', 0)}",
        f"- `support_bbox_out_of_range_count`: {ag.get('support_bbox_out_of_range_count', 0)}",
        f"- `invalid_support_mask_area_count`: {ag.get('invalid_support_mask_area_count', 0)}",
        f"- `invalid_TOR_count`: {ag.get('invalid_TOR_count', 0)}",
        f"- `TOR_mask_area_mismatch_count`: {ag.get('TOR_mask_area_mismatch_count', 0)}",
        "",
        f"- `missing_image_count`: {ag.get('missing_image_count', 0)}",
        f"- `invalid_image_count`: {ag.get('invalid_image_count', 0)}",
        f"- `image_size_mismatch_count`: {ag.get('image_size_mismatch_count', 0)}",
        "",
        f"- `duplicate_page_condition_count`: {result['duplicate_page_condition_count']}",
        "",
        f"- `clean_path_missing`: {ag.get('clean_path_missing', 0)}",
        "",
        "## Null optional fields (allowed but documented)",
        "",
        json.dumps(result["null_target_reports"], indent=2),
        "",
        "## JSONL parse errors",
        "",
        "none" if not result["meta_parse_errors"] else "\n".join(result["meta_parse_errors"][:50]),
        "",
        "## Metadata files",
        "",
        "Per-condition JSONL is accepted as `perturb_metadata.jsonl` (written by `generate_perturbed_pages.py`) or legacy `metadata.jsonl`.",
        "",
        "## Merged coverage (from merged file)",
        "",
        f"- pages with != 3 rows: {result['merged_bad_page_coverage_count']}",
        f"- conditions with != {result['n_expect']} rows: {result['merged_bad_condition_coverage_count']}",
        f"- duplicate (page_id, condition) extra rows: {result['duplicate_page_condition_count']}",
        "",
        "## Merged file",
        "",
        f"- path: `{result['merged_path']}`",
        "",
        f"- `ready_for_perturbed_parsing`: {result['ready_for_perturbed_parsing']}",
        "",
        "---",
        "",
        f"**Ready for Prompt 7:** {result['ready_prompt7']}",
        "",
    ]
    write_text(path, "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit debug20 perturbed pages and metadata.")
    parser.add_argument("--config", default="experiment_add/configs/perturb.yaml")
    parser.add_argument("--debug", action="store_true", help="Use page_manifest_debug20 and expect full coverage.")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    root = _infer_root(cfg_path)
    read_yaml(cfg_path)

    base_cfg = root / "experiment_add/configs/base.yaml"
    pm = PathManager(base_cfg, create_dirs=False)

    result = audit(debug=args.debug, root=root, pm=pm)
    out = root / "experiment_add/outputs/shared/perturbed_pages/perturbation_debug20_audit_report.md"
    ensure_dir(out.parent)
    _write_report(out, result)

    print(json.dumps({**result, "report_path": str(out)}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ready_prompt7"] == "YES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
