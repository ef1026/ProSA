from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiment_add.shared.utils.io import atomic_write_json, ensure_dir, read_yaml, write_text


STRUCTURAL_IDS = {"A01", "A02", "A03", "A04", "A09", "A10", "A11", "A12", "A13", "A14", "A17", "A18", "A20", "A22"}


def _project_root(config_path: Path) -> Path:
    return config_path.resolve().parents[2] if config_path.name == "perturb.yaml" else Path.cwd()


def _read_phase_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for directory in [root / "experiment/output", root / "output_v3", root / "output_previous3"]:
        if not directory.exists():
            continue
        for path in directory.glob("phase*.csv"):
            parser = "ppstructure" if "ppstructure" in path.name else "mineru"
            with path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    row["_parser"] = parser
                    rows.append(row)
    return rows


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except Exception:
        return 0.0


def _select_structural(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any], bool, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cid = row.get("config_id", "")
        if cid in STRUCTURAL_IDS:
            grouped[cid].append(row)
    if not grouped:
        return "A10", {"old_TOR": None, "old_B_SLR": None, "old_SLR_topo": None, "old_SLR_miss": None}, True, "No usable phase CSV structural rows found."
    best_id = "A10"
    best_score = -1.0
    best_metrics: dict[str, Any] = {}
    for cid, items in grouped.items():
        parsers = {r.get("_parser") for r in items}
        tor = sum(_float(r, "TOR") for r in items) / max(1, len(items))
        bslr = sum(_float(r, "B_SLR") for r in items) / max(1, len(items))
        topo = sum(_float(r, "SLR_topo") for r in items) / max(1, len(items))
        miss = sum(_float(r, "SLR_miss") for r in items) / max(1, len(items))
        if tor <= 0 or tor > 0.03 or len(parsers) < 2:
            continue
        score = (bslr / tor) + topo * 10 - miss * 2
        if score > best_score:
            best_score = score
            best_id = cid
            best_metrics = {"old_TOR": tor, "old_B_SLR": bslr, "old_SLR_topo": topo, "old_SLR_miss": miss, "B_SLR_per_TOR": bslr / tor}
    if not best_metrics:
        return "A10", {"old_TOR": None, "old_B_SLR": None, "old_SLR_topo": None, "old_SLR_miss": None}, True, "CSV rows found, but no stable low-TOR cross-parser structural candidate met criteria."
    return best_id, best_metrics, False, "Selected from phase CSV by low TOR and high B_SLR/TOR."


def _plan_from_anchor_config(config_id: str) -> list[dict[str, Any]]:
    """Return original ANCHOR_CONFIGS plan if available."""
    try:
        from experiment.config import ANCHOR_CONFIGS

        plan = ANCHOR_CONFIGS.get(config_id)
        if plan:
            return [{**step, "target_location": None, "reason": f"structural_probe:{config_id}"} for step in plan]
    except Exception:
        pass
    return [{"probe_type": "P5", "params": {"w": 3, "l_ratio": 0.5}, "target_strategy": "bridge", "target_location": None, "reason": f"structural_probe:{config_id}"}]


def select_probe_conditions(config_path: str | Path, debug: bool = False) -> dict[str, Any]:
    """Select concrete probe condition configs and write notes."""
    config_path = Path(config_path)
    root = _project_root(config_path)
    rows = _read_phase_rows(root)
    structural_id, metrics, fallback, note = _select_structural(rows)
    # Keep debug perturbation stable and interpretable.
    structural_plan = _plan_from_anchor_config(structural_id)
    target_tor = float(metrics.get("old_TOR") or 0.003)
    selection = {
        "structural_probe": {"probe_id": structural_id, "probe_family": "structural", "plan": structural_plan, **metrics},
        "area_matched_erasure": {"probe_id": "P4_area_matched", "probe_family": "area_matched_erasure", "target_tor": target_tor, "target_strategy": "random"},
        "large_area_erasure": {"probe_id": "A08", "probe_family": "large_area_erasure", "plan": [{"probe_type": "P4", "params": {"area_ratio": 0.20, "beta": 1.0}, "target_strategy": "content", "target_location": None, "reason": "large_area_erasure:A08"}]},
        "fallback_used": fallback,
        "selection_note": note,
    }
    out_dir = root / "experiment_add/outputs/shared/perturbed_pages"
    ensure_dir(out_dir)
    atomic_write_json(out_dir / "probe_selection.json", selection)
    lines = [
        "# Probe Selection Note",
        "",
        f"- selected structural probe id: `{structural_id}`",
        "- probe family: `structural`",
        f"- probe type: `{structural_plan[0].get('probe_type')}`",
        f"- target strategy: `{structural_plan[0].get('target_strategy')}`",
        f"- old TOR: `{metrics.get('old_TOR')}`",
        f"- old B_SLR: `{metrics.get('old_B_SLR')}`",
        f"- old SLR_topo: `{metrics.get('old_SLR_topo')}`",
        f"- old SLR_miss: `{metrics.get('old_SLR_miss')}`",
        f"- B_SLR per TOR: `{metrics.get('B_SLR_per_TOR')}`",
        f"- area-matched erasure TOR matching strategy: `P4 random placement with per-page area_ratio derived from structural target TOR={target_tor}`",
        "- large-area erasure selected: `A08`",
        "- expected structural_probe effect: low-footprint structure-sensitive disruption.",
        "- expected area_matched_erasure effect: footprint-matched non-structural erasure baseline.",
        "- expected large_area_erasure effect: high-footprint erasure baseline with target TOR around 15%-25%.",
        f"- fallback used: `{fallback}`",
        f"- note: {note}",
    ]
    if fallback:
        lines.append("- 部分 old metrics 缺失，因此使用 config-level fallback 选择。")
    write_text(out_dir / "probe_selection_note.md", "\n".join(lines) + "\n")
    return selection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment_add/configs/perturb.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    result = select_probe_conditions(args.config, debug=args.debug)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
