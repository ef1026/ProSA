"""Build the unified shared parser-metrics CSV for downstream Exp1/Exp2 use.

Reads source CSVs from the original AdvDoc experiment outputs (read-only) via
:mod:`experiment_add.shared.metrics.parser_metrics_loader`, normalizes field
names, derives ``B_SLR_per_TOR``, and writes:

* ``experiment_add/outputs/shared/parser_metrics/merged_parser_metrics.csv``
* ``experiment_add/outputs/shared/parser_metrics/parser_metrics_unification_report.md``

This script does not run any parser, does not call any LLM, does not modify
the original experiment outputs, and does not run Full500 evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiment_add.shared.metrics.parser_metrics_loader import (
    UNIFIED_FIELDS,
    load_parser_metrics,
    unified_field_coverage,
)
from experiment_add.shared.utils.io import ensure_dir, write_text
from experiment_add.shared.utils.path_manager import PathManager


_NUMERIC_FIELDS: tuple[str, ...] = (
    "TOR",
    "CER_matched_mean",
    "delta_CER",
    "B_SLR",
    "SLR_miss",
    "SLR_topo",
)


def _to_float(value: Any) -> float | None:
    """Parse a CSV cell into a float; return None for empties / unparsable."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _format_float(value: float | None) -> str:
    """Render a float (or NaN-equivalent) as a CSV cell.

    Empty cell represents NaN. This matches pandas' default ``read_csv``
    convention so downstream correlation analysis treats the cell as missing.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return f"{value:.1f}"
        return f"{value:.10g}"
    return str(value)


def _compute_b_slr_per_tor(b_slr: Any, tor: Any) -> float | None:
    """Return ``B_SLR / TOR`` as a float, or None when TOR is 0/missing."""
    b = _to_float(b_slr)
    t = _to_float(tor)
    if b is None or t is None:
        return None
    if t == 0.0:
        return None
    return b / t


def build_shared_parser_metrics(config_path: str | Path) -> dict[str, Any]:
    """Assemble the unified parser-metrics CSV and the unification report."""
    pm = PathManager(config_path, create_dirs=True)
    project_root = pm.project_root

    unified_rows, source_summaries = load_parser_metrics(project_root)

    output_rows: list[dict[str, str]] = []
    for raw in unified_rows:
        out: dict[str, str] = {field: "" for field in UNIFIED_FIELDS}
        for field in UNIFIED_FIELDS:
            if field in raw and raw[field] not in (None, ""):
                if field in _NUMERIC_FIELDS:
                    out[field] = _format_float(_to_float(raw[field]))
                else:
                    out[field] = str(raw[field]).strip()
        out["B_SLR_per_TOR"] = _format_float(
            _compute_b_slr_per_tor(raw.get("B_SLR"), raw.get("TOR"))
        )
        if not out["pipeline"] and "pipeline" in raw:
            out["pipeline"] = str(raw["pipeline"]).strip()
        if not out["condition"]:
            cond = raw.get("condition")
            out["condition"] = "" if cond is None else str(cond).strip()
        output_rows.append(out)

    out_dir = pm.parser_metrics_dir
    ensure_dir(out_dir)
    merged_path = out_dir / "merged_parser_metrics.csv"
    report_path = out_dir / "parser_metrics_unification_report.md"

    with merged_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(UNIFIED_FIELDS))
        writer.writeheader()
        writer.writerows(output_rows)

    coverage_counts = unified_field_coverage(output_rows)
    pipelines_detected = sorted({row["pipeline"] for row in output_rows if row["pipeline"]})
    config_id_counts = Counter(row["config_id"] for row in output_rows if row["config_id"])
    rows_for_a03 = config_id_counts.get("A03", 0)
    rows_for_a08 = config_id_counts.get("A08", 0)
    fields_available = sorted(f for f, n in coverage_counts.items() if n > 0)
    fields_missing = sorted(f for f, n in coverage_counts.items() if n == 0)

    has_b_slr = coverage_counts.get("B_SLR", 0) > 0
    has_slr_topo = coverage_counts.get("SLR_topo", 0) > 0
    has_cer = coverage_counts.get("CER_matched_mean", 0) > 0
    if has_b_slr and has_slr_topo and has_cer and pipelines_detected:
        readiness = "YES"
    elif has_b_slr and (has_slr_topo or has_cer) and pipelines_detected:
        readiness = "PARTIAL"
    else:
        readiness = "NO"

    source_files_found = [s["source_file"] for s in source_summaries]
    source_files_loaded = [s["source_file"] for s in source_summaries if s["rows"] > 0]
    conditions_mapped = sorted({row["condition"] for row in output_rows if row["condition"]})

    lines: list[str] = [
        "# Parser Metrics Unification Report",
        "",
        f"- output_csv: `{merged_path.relative_to(project_root)}`",
        f"- total_rows: `{len(output_rows)}`",
        f"- pipelines_detected: `{pipelines_detected}`",
        f"- conditions_mapped: `{conditions_mapped}`",
        f"- rows_for_A03: `{rows_for_a03}`",
        f"- rows_for_A08: `{rows_for_a08}`",
        f"- whether_B_SLR_available: `{'YES' if has_b_slr else 'NO'}`",
        f"- whether_SLR_topo_available: `{'YES' if has_slr_topo else 'NO'}`",
        f"- whether_CER_available: `{'YES' if has_cer else 'NO'}`",
        f"- ready_for_full_correlation: `{readiness}`",
        "",
        "## Source files",
        "",
        f"- source_files_found: `{len(source_files_found)}`",
        f"- source_files_loaded: `{len(source_files_loaded)}`",
        "",
        "| source_file | pipeline | rows |",
        "| --- | --- | --- |",
    ]
    for s in source_summaries:
        lines.append(f"| `{s['source_file']}` | `{s['pipeline']}` | `{s['rows']}` |")

    lines.extend([
        "",
        "## Field coverage in merged_parser_metrics.csv",
        "",
        f"- fields_available: `{fields_available}`",
        f"- fields_missing: `{fields_missing}`",
        "",
        "| field | non_empty_rows |",
        "| --- | --- |",
    ])
    for field in UNIFIED_FIELDS:
        lines.append(f"| `{field}` | `{coverage_counts.get(field, 0)}` |")

    lines.extend([
        "",
        "## Per-config_id row counts",
        "",
        "| config_id | rows |",
        "| --- | --- |",
    ])
    for cfg, n in config_id_counts.most_common():
        lines.append(f"| `{cfg}` | `{n}` |")

    lines.extend([
        "",
        "## Notes",
        "",
        "- Source CSVs in `experiment/output/` are treated as read-only and are not modified by this builder.",
        "- `condition` is derived from `config_id` (`A03` -> `structural_probe`, `A08` -> `large_area_erasure`); other config_ids leave `condition` empty.",
        "- `area_matched_erasure` has no source rows in the original parser metrics; downstream consumers should join on `page_id` + `config_id` and tolerate missing rows for that condition.",
        "- `B_SLR_per_TOR` is `B_SLR / TOR`; rows with `TOR == 0` or missing `TOR`/`B_SLR` get an empty cell (NaN).",
        "- Empty cells in the output CSV are NaN by convention (compatible with pandas `read_csv` default).",
        "- This stage did not run Full500, did not call DeepSeek, did not run any parser, and did not modify any original experiment output.",
    ])

    write_text(report_path, "\n".join(lines) + "\n")

    return {
        "merged_parser_metrics_csv": str(merged_path),
        "parser_metrics_unification_report": str(report_path),
        "total_rows": len(output_rows),
        "pipelines_detected": pipelines_detected,
        "conditions_mapped": conditions_mapped,
        "rows_for_A03": rows_for_a03,
        "rows_for_A08": rows_for_a08,
        "fields_available": fields_available,
        "fields_missing": fields_missing,
        "whether_B_SLR_available": has_b_slr,
        "whether_SLR_topo_available": has_slr_topo,
        "whether_CER_available": has_cer,
        "source_files_found": source_files_found,
        "source_files_loaded": source_files_loaded,
        "ready_for_full_correlation": readiness,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build shared parser metrics CSV and report.")
    parser.add_argument("--config", default="experiment_add/configs/base.yaml")
    args = parser.parse_args()
    result = build_shared_parser_metrics(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ready_for_full_correlation"] in {"YES", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
