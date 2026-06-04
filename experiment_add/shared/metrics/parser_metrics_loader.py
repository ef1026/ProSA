"""Load parser-level metrics from the original AdvDoc experiment outputs.

This module is read-only with respect to the source files; it never mutates
``experiment/output/*.csv``. It is used by
:mod:`experiment_add.shared.metrics.build_shared_parser_metrics` to assemble a
unified ``merged_parser_metrics.csv`` for downstream correlation analysis.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


CANDIDATE_INPUT_DIRS: tuple[str, ...] = (
    "experiment/output",
    "output_v3",
    "output_previous3",
)

UNIFIED_FIELDS: tuple[str, ...] = (
    "page_id",
    "pipeline",
    "condition",
    "config_id",
    "probe_type",
    "target_strategy",
    "TOR",
    "CER_matched_mean",
    "delta_CER",
    "B_SLR",
    "SLR_miss",
    "SLR_topo",
    "B_SLR_per_TOR",
    "source_file",
)

CONDITION_FROM_CONFIG_ID: dict[str, str] = {
    "A03": "structural_probe",
    "A08": "large_area_erasure",
}

# Tolerant header aliasing in case future CSVs use different casing or dashes.
# The first entry is the canonical name; subsequent entries are aliases.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "page_id": ("page_id", "image_id", "imageId", "image-id"),
    "pipeline": ("pipeline", "Pipeline"),
    "condition": ("condition", "Condition"),
    "config_id": ("config_id", "configId", "config-id"),
    "probe_type": ("probe_type", "probeType", "probe-type"),
    "target_strategy": ("target_strategy", "targetStrategy", "target-strategy"),
    "TOR": ("TOR", "tor"),
    "CER_matched_mean": ("CER_matched_mean", "cer_matched_mean", "CER-matched-mean"),
    "delta_CER": ("delta_CER", "deltaCER", "delta-CER", "Δ_CER"),
    "B_SLR": ("B_SLR", "B-SLR", "b_slr", "BSLR", "B_slr"),
    "SLR_miss": ("SLR_miss", "slr_miss", "SLR-miss"),
    "SLR_topo": ("SLR_topo", "slr_topo", "SLR-topo"),
}

# Filename keywords that must trigger discovery of a parser-metrics CSV.
_FILENAME_KEYWORDS: tuple[str, ...] = (
    "phase",
    "anchor",
    "policy",
    "ppstructure",
    "global",
    "parser_metric",
)


def _is_jupyter_artifact(path: Path) -> bool:
    """Return True for paths inside ``.ipynb_checkpoints`` directories."""
    return any(part == ".ipynb_checkpoints" for part in path.parts)


def _looks_like_parser_metrics(path: Path) -> bool:
    """Return True if a CSV filename matches any expected parser-metrics keyword."""
    name = path.name.lower()
    return any(kw in name for kw in _FILENAME_KEYWORDS)


def _detect_pipeline(path: Path) -> str:
    """Detect parser pipeline from filename.

    Filenames containing ``ppstructure`` map to PPStructure; otherwise the
    file is attributed to MinerU per the experiment_add contract.
    """
    return "ppstructure" if "ppstructure" in path.name.lower() else "mineru"


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map a raw CSV row's columns onto canonical names defined in ``_FIELD_ALIASES``.

    Unknown columns are dropped; known columns are remapped to their canonical
    name. Empty strings are preserved as-is so the caller can decide how to
    represent missing/NaN values in the output.
    """
    canonical: dict[str, Any] = {}
    for canon, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            if alias in row:
                canonical[canon] = row[alias]
                break
    return canonical


def discover_metric_csvs(project_root: str | Path) -> list[Path]:
    """Return parser-metrics CSV paths under any of the candidate input dirs.

    Skips Jupyter checkpoint copies so the same logical file is never loaded
    twice. The returned list is deterministic (sorted by path).
    """
    root = Path(project_root)
    found: set[Path] = set()
    for directory in CANDIDATE_INPUT_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*.csv"):
            if _is_jupyter_artifact(path):
                continue
            if not _looks_like_parser_metrics(path):
                continue
            found.add(path.resolve())
    return sorted(found)


def _read_csv(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (rows, fieldnames) for a CSV. Missing files produce empty results."""
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def load_parser_metrics(project_root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and unify parser metrics from candidate input directories.

    Returns ``(unified_rows, source_summaries)``:

    * ``unified_rows`` is a list of dicts whose keys are a subset of
      :data:`UNIFIED_FIELDS`. The caller is expected to compute
      ``B_SLR_per_TOR`` and finalise NaN handling.
    * ``source_summaries`` is a list of per-source-file dicts containing
      ``source_file``, ``pipeline``, ``rows``, and ``original_fields`` so the
      builder can write a faithful report without re-reading the files.
    """
    root = Path(project_root).resolve()
    csvs = discover_metric_csvs(root)
    unified: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for path in csvs:
        rows, fieldnames = _read_csv(path)
        pipeline = _detect_pipeline(path)
        rel_source = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        for row in rows:
            normalized = _normalize_row(row)
            normalized["pipeline"] = pipeline
            normalized.setdefault("condition", _condition_from_config(normalized.get("config_id")))
            normalized["source_file"] = rel_source
            unified.append(normalized)
        summaries.append(
            {
                "source_file": rel_source,
                "pipeline": pipeline,
                "rows": len(rows),
                "original_fields": fieldnames,
            }
        )
    return unified, summaries


def _condition_from_config(config_id: Any) -> str:
    """Map a ``config_id`` to its perturbation condition (or ``""`` if unknown).

    Per the Exp1 contract:

    * ``A03`` → ``structural_probe``
    * ``A08`` → ``large_area_erasure``
    * other config_ids have no defined parser-side condition mapping yet.
    """
    if not config_id:
        return ""
    return CONDITION_FROM_CONFIG_ID.get(str(config_id).strip().upper(), "")


def unified_field_coverage(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Return per-canonical-field non-empty value counts across ``rows``.

    Used by the builder to populate ``fields_available`` /
    ``fields_missing`` in the unification report without inferring schema
    from a single source CSV.
    """
    counts = {field: 0 for field in UNIFIED_FIELDS}
    for row in rows:
        for field in UNIFIED_FIELDS:
            value = row.get(field, "")
            if value not in ("", None):
                counts[field] += 1
    return counts
