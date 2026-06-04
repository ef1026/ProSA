from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from PIL import Image

from experiment_add.shared.data.sampling import random_sample, stratified_sample
from experiment_add.shared.utils.io import ensure_dir, read_yaml, write_text
from experiment_add.shared.utils.path_manager import PathManager


MANIFEST_FIELDS = [
    "page_id",
    "dataset",
    "split",
    "complexity",
    "image_path",
    "width",
    "height",
    "n_orig_spans",
    "source_doc_id",
    "page_index",
]

TARGET_DISTRIBUTION = {"simple": 0.145, "medium": 0.464, "complex": 0.391}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _relative_or_absolute(path: Path, root: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _dataset_label(value: str | None) -> str:
    """Normalize dataset labels for the manifest."""
    if value == "publaynet":
        return "PubLayNet"
    if value == "doclaynet":
        return "DocLayNet"
    return "unknown"


def _infer_doc_and_page(image_id: str) -> tuple[str, str]:
    """Infer source document id and page index from common PubLayNet filenames."""
    stem = Path(image_id).stem
    if "_" not in stem:
        return "unknown", "unknown"
    doc_id, maybe_page = stem.rsplit("_", 1)
    if maybe_page.isdigit():
        return doc_id, str(int(maybe_page))
    return "unknown", "unknown"


def _read_selected_image_metadata(selected_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Read lightweight image metadata from selected_600.json files."""
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for source in ("publaynet", "doclaynet"):
        json_path = selected_root / source / "selected_600.json"
        if not json_path.exists():
            continue
        with json_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)
        for image in coco.get("images", []):
            file_name = image.get("file_name")
            if file_name:
                index[(source, file_name)] = image
    return index


def _record_from_shared_entry(
    entry: dict[str, Any],
    selected_root: Path,
    selected_meta: dict[tuple[str, str], dict[str, Any]],
    project_root: Path,
) -> dict[str, Any]:
    image_id = str(entry.get("image_id", ""))
    source = str(entry.get("data_source", "unknown")).lower()
    image_path = selected_root / source / "images" / image_id
    meta = selected_meta.get((source, image_id), {})
    source_doc_id, page_index = _infer_doc_and_page(image_id)
    if source_doc_id == "unknown":
        source_doc_id = str(meta.get("source_doc_id") or meta.get("doc_id") or "unknown")
    if page_index == "unknown":
        page_index = str(meta.get("page_index", "unknown"))

    return {
        "page_id": image_id,
        "dataset": _dataset_label(source),
        "split": "validation",
        "complexity": str(entry.get("level") or entry.get("complexity") or "unknown"),
        "image_path": _relative_or_absolute(image_path, project_root),
        "width": "",
        "height": "",
        "n_orig_spans": entry.get("n_elements_mineru", entry.get("n_orig_spans", "")),
        "source_doc_id": source_doc_id or "unknown",
        "page_index": page_index or "unknown",
    }


def _load_from_shared_eval_set(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load candidates from config/shared_eval_set.json if present."""
    shared_path = project_root / "config" / "shared_eval_set.json"
    if not shared_path.exists():
        return [], {"used": False, "path": str(shared_path), "reason": "missing"}

    with shared_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    metadata = index.get("metadata", {})
    selected_root = project_root / str(metadata.get("selected_root", "data/selected"))
    selected_meta = _read_selected_image_metadata(selected_root)
    candidates = [
        _record_from_shared_entry(entry, selected_root, selected_meta, project_root)
        for entry in index.get("images", [])
    ]
    return candidates, {"used": True, "path": str(shared_path), "metadata": metadata}


def _load_from_phase_csv(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fallback: derive unique candidates from original phase CSV files."""
    output_dirs = [project_root / "experiment" / "output", project_root / "output_v3", project_root / "output_previous3"]
    csv_paths: list[Path] = []
    for output_dir in output_dirs:
        if output_dir.exists():
            csv_paths.extend(sorted(output_dir.glob("phase*.csv")))

    by_page: dict[str, dict[str, Any]] = {}
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                image_id = row.get("image_id", "")
                if not image_id or image_id in by_page:
                    continue
                source = row.get("data_source", "unknown")
                image_path = project_root / "data" / "selected" / source / "images" / image_id
                source_doc_id, page_index = _infer_doc_and_page(image_id)
                by_page[image_id] = {
                    "page_id": image_id,
                    "dataset": _dataset_label(source),
                    "split": "validation",
                    "complexity": row.get("complexity_level", "unknown") or "unknown",
                    "image_path": _relative_or_absolute(image_path, project_root),
                    "width": "",
                    "height": "",
                    "n_orig_spans": row.get("n_orig_spans", ""),
                    "source_doc_id": source_doc_id,
                    "page_index": page_index,
                }
    return list(by_page.values()), {"used": bool(by_page), "csv_count": len(csv_paths)}


def _load_from_clean_page_dir(project_root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fallback: scan data.clean_page_dir for image files."""
    clean_page_dir = config.get("data", {}).get("clean_page_dir")
    if not clean_page_dir:
        return [], {"used": False, "reason": "data.clean_page_dir not configured"}
    root = Path(clean_page_dir)
    if not root.is_absolute():
        root = project_root / root
    if not root.exists():
        return [], {"used": False, "path": str(root), "reason": "missing"}

    candidates = []
    for image_path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        source_doc_id, page_index = _infer_doc_and_page(image_path.name)
        candidates.append({
            "page_id": image_path.name,
            "dataset": "unknown",
            "split": "unknown",
            "complexity": "unknown",
            "image_path": _relative_or_absolute(image_path, project_root),
            "width": "",
            "height": "",
            "n_orig_spans": "",
            "source_doc_id": source_doc_id,
            "page_index": page_index,
        })
    return candidates, {"used": bool(candidates), "path": str(root)}


def _resolve_image_path(image_path: str, project_root: Path) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else project_root / path


def _validate_candidates(candidates: list[dict[str, Any]], project_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Check images, fill dimensions, and deduplicate by page_id and image_path."""
    filtered = Counter()
    valid: list[dict[str, Any]] = []
    seen_page_ids: set[str] = set()
    seen_image_paths: set[str] = set()

    for record in candidates:
        page_id = str(record.get("page_id", "")).strip()
        image_path_text = str(record.get("image_path", "")).strip()
        if not page_id:
            filtered["missing_page_id"] += 1
            continue
        if page_id in seen_page_ids:
            filtered["duplicate_page_id"] += 1
            continue
        image_path = _resolve_image_path(image_path_text, project_root)
        image_key = str(image_path.resolve()) if image_path.exists() else str(image_path)
        if image_key in seen_image_paths:
            filtered["duplicate_image_path"] += 1
            continue
        if not image_path.exists():
            filtered["missing_image_path"] += 1
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            filtered["image_read_failed"] += 1
            continue
        if width <= 0 or height <= 0:
            filtered["invalid_dimensions"] += 1
            continue

        clean = {field: record.get(field, "unknown") for field in MANIFEST_FIELDS}
        clean["width"] = int(width)
        clean["height"] = int(height)
        for field in ("dataset", "split", "complexity", "source_doc_id", "page_index"):
            if clean.get(field) in (None, ""):
                clean[field] = "unknown"
        if clean.get("n_orig_spans") is None:
            clean["n_orig_spans"] = ""
        valid.append(clean)
        seen_page_ids.add(page_id)
        seen_image_paths.add(image_key)

    return valid, dict(filtered)


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in MANIFEST_FIELDS})


def _distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record.get(key, "unknown") or "unknown") for record in records))


def _missing_field_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {}
    for field in MANIFEST_FIELDS:
        counts[field] = sum(1 for record in records if record.get(field) in (None, "", "unknown"))
    return counts


def _write_report(
    path: Path,
    source_info: dict[str, Any],
    candidate_count: int,
    filtered: dict[str, int],
    valid_count: int,
    full_records: list[dict[str, Any]],
    debug_records: list[dict[str, Any]],
    output_paths: dict[str, Path],
    used_clean_page_fallback: bool,
    target_n: int,
) -> None:
    total_filtered = sum(filtered.values())
    missing_counts = _missing_field_counts(full_records)
    lines = [
        "# Selected Pages Report",
        "",
        "## Data Source",
        "",
        f"- Primary source used: `{source_info.get('source_name', 'unknown')}`",
        f"- Source path: `{source_info.get('path', 'unknown')}`",
        f"- Fallback scan clean_page_dir used: `{used_clean_page_fallback}`",
        "",
        "## Candidate And Filter Counts",
        "",
        f"- Candidate pages before image validation: `{candidate_count}`",
        f"- Valid pages after image validation/deduplication: `{valid_count}`",
        f"- Filtered pages total: `{total_filtered}`",
        f"- Filter reasons: `{filtered}`",
        "",
        "## Manifest Counts",
        "",
        f"- Full manifest target: `{target_n}`",
        f"- Full manifest actual: `{len(full_records)}`",
        f"- Debug manifest actual: `{len(debug_records)}`",
        f"- Debug is subset of full: `{set(r['page_id'] for r in debug_records).issubset({r['page_id'] for r in full_records})}`",
        f"- Reached {target_n}-page target: `{len(full_records) >= target_n}`",
        "",
        "## Dataset Distribution",
        "",
        f"- Full: `{_distribution(full_records, 'dataset')}`",
        f"- Debug: `{_distribution(debug_records, 'dataset')}`",
        "",
        "## Complexity Distribution",
        "",
        f"- Full: `{_distribution(full_records, 'complexity')}`",
        f"- Debug: `{_distribution(debug_records, 'complexity')}`",
        "",
        "## Missing Field Counts In Full Manifest",
        "",
        f"`{missing_counts}`",
        "",
        "## Image Path Check",
        "",
        "- All manifest rows have existing image paths readable by PIL.",
        "- Width and height are greater than 0 for all manifest rows.",
        "",
        "## Output Files",
        "",
        f"- Full manifest: `{output_paths['full']}`",
        f"- Debug manifest: `{output_paths['debug']}`",
        f"- Report: `{path}`",
        "",
        "## Notes",
        "",
        "- Missing `source_doc_id` or `page_index` values are set to `unknown` when not inferable from filename/metadata.",
        "- `n_orig_spans` uses `n_elements_mineru` from `config/shared_eval_set.json` when available.",
        "- No images were copied or moved.",
        "- No parser, DeepSeek, perturbation, QA answering, or evaluation code was called.",
    ]
    write_text(path, "\n".join(lines) + "\n")


def build_manifests(config_path: str | Path, debug: bool = False) -> dict[str, Any]:
    """Build full-size and debug20 page manifests without calling parsers."""
    pm = PathManager(config_path, create_dirs=True)
    config = read_yaml(config_path)
    project_root = pm.project_root
    seed = int(config.get("random", {}).get("seed", 42))

    candidates, source_info = _load_from_shared_eval_set(project_root)
    source_name = "config/shared_eval_set.json"
    used_clean_page_fallback = False
    if not candidates:
        candidates, source_info = _load_from_phase_csv(project_root)
        source_name = "phase_csv"
    if not candidates:
        candidates, source_info = _load_from_clean_page_dir(project_root, config)
        source_name = "clean_page_dir"
        used_clean_page_fallback = bool(candidates)
    source_info["source_name"] = source_name

    valid_candidates, filtered = _validate_candidates(candidates, project_root)
    target_n = 1000
    debug_n = 20
    full_records = stratified_sample(valid_candidates, target_n, "complexity", TARGET_DISTRIBUTION, seed)
    debug_records = stratified_sample(full_records, debug_n, "complexity", TARGET_DISTRIBUTION, seed + 1)
    if len(debug_records) > len(full_records):
        debug_records = random_sample(full_records, min(debug_n, len(full_records)), seed + 1)

    report_path = project_root / "experiment_add" / "data" / "selected_pages_report.md"
    _write_manifest(pm.page_manifest_500, full_records)
    _write_manifest(pm.page_manifest_debug20, debug_records)
    _write_report(
        report_path,
        source_info,
        candidate_count=len(candidates),
        filtered=filtered,
        valid_count=len(valid_candidates),
        full_records=full_records,
        debug_records=debug_records,
        output_paths={"full": pm.page_manifest_500, "debug": pm.page_manifest_debug20},
        used_clean_page_fallback=used_clean_page_fallback,
        target_n=target_n,
    )
    return {
        "source": source_name,
        "candidate_count": len(candidates),
        "valid_count": len(valid_candidates),
        "filtered": filtered,
        "full_count": len(full_records),
        "debug_count": len(debug_records),
        "debug_subset": set(r["page_id"] for r in debug_records).issubset({r["page_id"] for r in full_records}),
        "full_path": str(pm.page_manifest_500),
        "debug_path": str(pm.page_manifest_debug20),
        "report_path": str(report_path),
        "debug_mode": debug,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build experiment_add page manifests.")
    parser.add_argument("--config", default="experiment_add/configs/base.yaml")
    parser.add_argument("--debug", action="store_true", help="Run manifest build in debug mode; outputs keep final schema.")
    args = parser.parse_args()
    summary = build_manifests(args.config, debug=args.debug)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
