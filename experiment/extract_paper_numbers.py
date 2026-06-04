"""Extract all paper numbers required by acl_latex2.tex.

Emits both a human-readable stdout report and a machine-readable
``experiment/output/paper_numbers.json`` side-car containing the full
per-pipeline aggregates. The JSON is the canonical source for filling
the ``[TBD_P1]`` placeholders in Sections 5.2--5.5, the abstract, the
conclusion, and the appendix tables of acl_latex2.tex.

Usage::

    python experiment/extract_paper_numbers.py
    python experiment/extract_paper_numbers.py --output_dir experiment/output
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIVE_LEVEL = ["TOR", "ACR", "BPO", "BOC", "EIR"]
PATHWAYS = ["SLR_miss", "SLR_topo"]
CHANNEL_COLS = ["B_SLR", "B_SLR_iou_only", "B_SLR_text_only"]

# Matched-TOR band anchors referenced in Section 5.5.
MATCHED_ANCHORS = {
    "A05": "P3_a_alpha03",
    "A06": "P3_a_alpha1",
    "A09": "P5_b_w1",
    "A10": "P5_b_w3",
    "A08": "P4_c_20pct",
    "A19": "P4_b_20pct",
}

NT_ENDPOINTS = ("NT01", "NT07")


def _r2(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    if len(xv) < 3:
        return float("nan")
    ss_res = np.sum((yv - np.polyval(np.polyfit(xv, yv, 1), xv)) ** 2)
    ss_tot = np.sum((yv - yv.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _spearman(x, y) -> float:
    xs = pd.Series(np.asarray(x, dtype=float))
    ys = pd.Series(np.asarray(y, dtype=float))
    mask = xs.notna() & ys.notna()
    if mask.sum() < 3:
        return float("nan")
    rx = xs[mask].rank()
    ry = ys[mask].rank()
    val = rx.corr(ry)
    return float(val) if pd.notna(val) else float("nan")


def _config_means(df: pd.DataFrame) -> pd.DataFrame:
    cols = FIVE_LEVEL + PATHWAYS + CHANNEL_COLS + [
        "CER_matched_mean",
        "delta_CER",
        "delta_mAP",
    ]
    present = [c for c in cols if c in df.columns]
    return df.groupby("config_id")[present].mean()


def _efficiency_per_image(df: pd.DataFrame, cid: str, numerator: str) -> float:
    """Mean per-image ratio ``numerator / TOR`` for a given config."""
    sub = df[df.config_id == cid]
    if sub.empty:
        return float("nan")
    num = sub[numerator].to_numpy(dtype=float)
    den = sub["TOR"].to_numpy(dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den > 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(num[mask] / den[mask]))


def analyze_pipeline(name: str, p1a_path: Path) -> dict | None:
    if not p1a_path.exists():
        print(f"  [SKIP] {p1a_path} not found")
        return None

    df = pd.read_csv(p1a_path)
    n_rows = len(df)
    n_configs = df["config_id"].nunique()
    print(f"  Loaded {n_rows} rows / {n_configs} configs from {p1a_path.name}")

    cfg = _config_means(df)

    # ---- Channel partition and coupling (Section 5.2, Table tab:fail_channel). ----
    channel_means = {
        "B_SLR": float(cfg["B_SLR"].mean()),
        "B_SLR_IoU": float(cfg["B_SLR_iou_only"].mean()),
        "B_SLR_text": float(cfg["B_SLR_text_only"].mean()),
    }
    channel_means["text_share"] = (
        channel_means["B_SLR_text"] / channel_means["B_SLR"]
        if channel_means["B_SLR"] > 0
        else float("nan")
    )
    rho_channels = _spearman(cfg["B_SLR_iou_only"], cfg["B_SLR_text_only"])
    r2_iou_composite = _r2(cfg["B_SLR_iou_only"], cfg["B_SLR"])
    r2_text_composite = _r2(cfg["B_SLR_text_only"], cfg["B_SLR"])

    # ---- Faithfulness (Section 5.2 closing). ----
    r2_bslr_cer = _r2(cfg["B_SLR"], cfg["CER_matched_mean"])
    rho_bslr_cer = _spearman(cfg["B_SLR"], cfg["CER_matched_mean"])
    r2_bslr_dmap = _r2(cfg["B_SLR"], cfg["delta_mAP"])
    rho_bslr_dmap = _spearman(cfg["B_SLR"], cfg["delta_mAP"])

    # ---- Gradient R^2 for Table tab:grad_pathway (Section 5.3). ----
    grad_pathway: dict[str, dict[str, float]] = {}
    for pathway in PATHWAYS:
        grad_pathway[pathway] = {g: _r2(cfg[g], cfg[pathway]) for g in FIVE_LEVEL}

    # ---- Gradient R^2 against every B-SLR channel (Appendix tab:fail_channel_gradient). ----
    channel_grad: dict[str, dict[str, float]] = {}
    for ch_key, ch_col in [
        ("composite", "B_SLR"),
        ("iou_only", "B_SLR_iou_only"),
        ("text_only", "B_SLR_text_only"),
    ]:
        channel_grad[ch_key] = {g: _r2(cfg[g], cfg[ch_col]) for g in FIVE_LEVEL}

    # ---- Footprint-Bias predictors (Section 5.5). ----
    r2_tor_cer = _r2(cfg["TOR"], cfg["CER_matched_mean"])
    r2_bslr_cer_main = r2_bslr_cer  # same quantity, reported in Section 5.5 right panel

    # ---- Matched-TOR anchor CER (Section 5.5 left panel). ----
    anchor_cer = {}
    anchor_tor = {}
    for cid, label in MATCHED_ANCHORS.items():
        sub = df[df.config_id == cid]
        if not sub.empty:
            anchor_cer[label] = float(sub["CER_matched_mean"].mean())
            anchor_tor[label] = float(sub["TOR"].mean())
        else:
            anchor_cer[label] = float("nan")
            anchor_tor[label] = float("nan")

    # ---- NT endpoints (Section 5.4). ----
    nt = {}
    for cid in NT_ENDPOINTS:
        sub = df[df.config_id == cid]
        if not sub.empty:
            nt[cid] = {
                "CER": float(sub["CER_matched_mean"].mean()),
                "B_SLR": float(sub["B_SLR"].mean()),
                "TOR": float(sub["TOR"].mean()),
            }
        else:
            nt[cid] = {"CER": float("nan"), "B_SLR": float("nan"), "TOR": float("nan")}

    # ---- Per-pixel CER efficiency comparison for Abstract / Conclusion. ----
    eff_cer_a10 = _efficiency_per_image(df, "A10", "CER_matched_mean")
    eff_cer_a19 = _efficiency_per_image(df, "A19", "CER_matched_mean")
    eff_cer_a08 = _efficiency_per_image(df, "A08", "CER_matched_mean")
    eff_cer_nt07 = _efficiency_per_image(df, "NT07", "CER_matched_mean")
    eff_bslr_a10 = _efficiency_per_image(df, "A10", "B_SLR")
    eff_bslr_a19 = _efficiency_per_image(df, "A19", "B_SLR")

    # Ratio vs area-matched P4 erasures. Abstract language: "per-pixel CER increase
    # ... fold larger than area-matched erasures". We pick the P4 erasure pair
    # (A08 content / A19 bridge) and use the bridge variant as the direct
    # comparator for A10 (P5 bridge).
    fold_vs_a19 = eff_cer_a10 / eff_cer_a19 if eff_cer_a19 > 0 else float("nan")
    fold_vs_a08 = eff_cer_a10 / eff_cer_a08 if eff_cer_a08 > 0 else float("nan")
    fold_nt07_vs_a19 = eff_cer_nt07 / eff_cer_a19 if eff_cer_a19 > 0 else float("nan")

    # ---- delta_mAP at the two P4 erasures (Appendix mAP paragraph). ----
    dmap_p4 = {}
    for cid, label in [("A08", "P4_c_20pct"), ("A19", "P4_b_20pct")]:
        sub = df[df.config_id == cid]
        dmap_p4[label] = float(sub["delta_mAP"].mean()) if not sub.empty else float("nan")

    # ---- Per-config means table (kept in JSON for reproducibility). ----
    cfg_table = cfg.reset_index().to_dict(orient="records")

    result = {
        "pipeline": name,
        "n_rows": n_rows,
        "n_configs": int(n_configs),
        "channel_means": channel_means,
        "channel_coupling": {
            "rho_spearman_iou_text": rho_channels,
            "R2_iou_vs_composite": r2_iou_composite,
            "R2_text_vs_composite": r2_text_composite,
        },
        "faithfulness": {
            "R2_BSLR_to_CER": r2_bslr_cer,
            "rho_BSLR_to_CER": rho_bslr_cer,
            "R2_BSLR_to_deltamAP": r2_bslr_dmap,
            "rho_BSLR_to_deltamAP": rho_bslr_dmap,
        },
        "gradient_pathway_R2": grad_pathway,
        "channel_gradient_R2": channel_grad,
        "footprint_bias": {
            "R2_TOR_to_CER": r2_tor_cer,
            "R2_BSLR_to_CER": r2_bslr_cer_main,
        },
        "matched_tor_anchors": {
            "CER": anchor_cer,
            "TOR": anchor_tor,
        },
        "nt_endpoints": nt,
        "efficiency_per_image": {
            "Eff_CER_A10": eff_cer_a10,
            "Eff_CER_A19": eff_cer_a19,
            "Eff_CER_A08": eff_cer_a08,
            "Eff_CER_NT07": eff_cer_nt07,
            "Eff_BSLR_A10": eff_bslr_a10,
            "Eff_BSLR_A19": eff_bslr_a19,
            "fold_A10_over_A19": fold_vs_a19,
            "fold_A10_over_A08": fold_vs_a08,
            "fold_NT07_over_A19": fold_nt07_vs_a19,
        },
        "deltamAP_P4_erasures": dmap_p4,
        "config_level_means": cfg_table,
    }

    # ---- Stdout echo for quick human inspection. ----
    print()
    print(f"  ── Channel partition ({name}, n={n_configs}) ──")
    print(
        f"    B-SLR={channel_means['B_SLR']:.3f} | "
        f"IoU-only={channel_means['B_SLR_IoU']:.3f} | "
        f"text-only={channel_means['B_SLR_text']:.3f} | "
        f"text/composite={channel_means['text_share']:.3f}"
    )
    print(
        f"    rho_s(IoU,text)={rho_channels:.3f} | "
        f"R2(IoU->composite)={r2_iou_composite:.3f} | "
        f"R2(text->composite)={r2_text_composite:.3f}"
    )
    print(f"  ── Faithfulness ──")
    print(
        f"    R2(B-SLR->CER)={r2_bslr_cer:.3f} | rho_s={rho_bslr_cer:.3f} | "
        f"R2(B-SLR->dmAP)={r2_bslr_dmap:.3f}"
    )
    print(f"  ── Gradient x pathway R^2 ──")
    for pw, row in grad_pathway.items():
        cells = " & ".join(f"{row[g]:.3f}" for g in FIVE_LEVEL)
        print(f"    {pw}: {cells}")
    print(f"  ── Footprint Bias (Sec 5.5) ──")
    print(f"    R2(TOR->CER)={r2_tor_cer:.3f}  R2(B-SLR->CER)={r2_bslr_cer_main:.3f}")
    print(f"  ── Matched-TOR CER anchors ──")
    for label, val in anchor_cer.items():
        print(f"    {label}: CER={val:.4f} (TOR={anchor_tor[label]:.5f})")
    print(f"  ── NT endpoints ──")
    for cid, v in nt.items():
        print(f"    {cid}: CER={v['CER']:.4f}  B-SLR={v['B_SLR']:.4f}  TOR={v['TOR']:.5f}")
    print(f"  ── Per-pixel efficiency ──")
    print(
        f"    Eff_CER(A10)={eff_cer_a10:.2f}  Eff_CER(A19)={eff_cer_a19:.2f}  "
        f"Eff_CER(A08)={eff_cer_a08:.2f}  Eff_CER(NT07)={eff_cer_nt07:.2f}"
    )
    print(
        f"    fold(A10/A19)={fold_vs_a19:.1f}x  fold(A10/A08)={fold_vs_a08:.1f}x  "
        f"fold(NT07/A19)={fold_nt07_vs_a19:.1f}x"
    )
    print(f"  ── Delta mAP at P4 erasures ──")
    for k, v in dmap_p4.items():
        print(f"    {k}: dmAP={v:.3f}")

    return result


def _format_latex_tables(results: dict[str, dict]) -> str:
    """Emit LaTeX-ready rows for quick insertion into the .tex source."""
    lines: list[str] = []
    lines.append("")
    lines.append("% ── Table tab:fail_channel (a) channel partition ──")
    for name in ("MinerU", "PPV3"):
        if name not in results:
            continue
        r = results[name]["channel_means"]
        lines.append(
            f"% {name}: B-SLR={r['B_SLR']:.3f} IoU={r['B_SLR_IoU']:.3f} "
            f"text={r['B_SLR_text']:.3f}"
        )
    lines.append("% ── Table tab:fail_channel (b) cross-channel coupling ──")
    for name in ("MinerU", "PPV3"):
        if name not in results:
            continue
        c = results[name]["channel_coupling"]
        lines.append(
            f"% {name}: rho_s={c['rho_spearman_iou_text']:.3f} "
            f"R2_IoU={c['R2_iou_vs_composite']:.3f} "
            f"R2_text={c['R2_text_vs_composite']:.3f}"
        )
    lines.append("% ── Table tab:grad_pathway ──")
    for name in ("MinerU", "PPV3"):
        if name not in results:
            continue
        gp = results[name]["gradient_pathway_R2"]
        for pw in PATHWAYS:
            cells = " & ".join(f".{gp[pw][g]:.3f}"[1:] for g in FIVE_LEVEL)
            lines.append(f"% {name} {pw}: {cells}")
    lines.append("% ── Table tab:fail_channel_gradient ──")
    for name in ("MinerU", "PPV3"):
        if name not in results:
            continue
        cg = results[name]["channel_gradient_R2"]
        for ch in ("composite", "iou_only", "text_only"):
            cells = " & ".join(f".{cg[ch][g]:.3f}"[1:] for g in FIVE_LEVEL)
            lines.append(f"% {name} {ch}: {cells}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="experiment/output")
    args = parser.parse_args()

    out = Path(args.output_dir)

    print("=" * 72)
    print("PAPER NUMBER EXTRACTION (post-rerun)")
    print("=" * 72)

    results: dict[str, dict] = {}
    for pname, suffix in [("MinerU", ""), ("PPV3", "_ppstructure")]:
        print(f"\n{'=' * 36}")
        print(f"  Pipeline: {pname}")
        print(f"{'=' * 36}")
        p1a = out / f"phase1a_anchors{suffix}.csv"
        r = analyze_pipeline(pname, p1a)
        if r is not None:
            results[pname] = r

    if len(results) == 2:
        print("\n" + _format_latex_tables(results))

    # Write JSON side-car.
    json_path = out / "paper_numbers.json"
    json_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote: {json_path}")

    print(f"\n{'=' * 72}")
    print("DONE — paper_numbers.json is the canonical source for the .tex fill-in.")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
