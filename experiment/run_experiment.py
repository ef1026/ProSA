from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiment.config import (
    ANCHOR_CONFIGS,
    EIR_TARGETED_CONFIGS,
    SENSITIVITY_CONFIGS,
    SWEEP_CONFIGS,
    ExperimentConfig,
    ensure_output_dirs,
    setup_advdoc_paths,
    setup_cuda_dll_paths,
)
from experiment.data.dataset_builder import build_stratified_dataset, build_stratified_dataset_with_pool, load_images_by_filenames, build_combined_dataset, build_shared_dataset
from experiment.engine import AttackEngine
from experiment.gpu_pool import GPUParserPool
from experiment.metrics import (
    calculate_b_slr,
    calculate_decomposed_b_slr,
    calculate_ssim,
    compute_boc,
    compute_character_slr,
    compute_eir,
    compute_acr,
    compute_eir_from_detection,
    compute_ncsic_mask,  # used as BPO base
    calculate_ocr_ablation_metrics,
    compute_splash_ratio,
    compute_tor,
    extract_elements_from_result,
    extract_full_text_from_result,
    extract_span_elements_from_result,
)
from experiment.metrics_map import compute_delta_map, compute_delta_cer_from_elements
from experiment.eir_targeting import find_positions_for_eir
from experiment.policies.llm_policy_biased import LLMPolicyBiased
from experiment.policies.llm_policy_neutral import LLMPolicyNeutral
from experiment.policies.random_policy import RandomPolicy
from experiment.policies.rule_policy import RuleBasedPolicy


# ── Incremental CSV helpers ─────────────────────────────────────────────
def _load_completed_keys(csv_path: Path, key_columns: list[str]) -> set[tuple]:
    """Load already-completed composite keys from a partial CSV for resume."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=key_columns)
        if df.empty:
            return set()
        return set(df[key_columns].itertuples(index=False, name=None))
    except Exception:
        return set()


def _append_row(csv_path: Path, row: dict | None) -> None:
    """Append a single row to CSV. Write header only if the file is new/empty.

    ``row=None`` is a no-op: callers (notably ``_record_common``) skip writing
    when the clean-image baseline is empty, since every parser-dependent
    metric would otherwise collapse to zero and pollute downstream stats.
    """
    if row is None:
        return
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=write_header, index=False)


# Track image_ids we've already warned about to avoid tqdm spam: phase 1/2
# calls _record_common many times per image, but one warning per image is
# enough to diagnose a bad baseline.
_EMPTY_BASELINE_WARNED: set[str] = set()


# ─────────────────────────────────────────────────────────────────────────

def _sample_sweep_step(config: dict, rng: np.random.Generator) -> dict:
    params = {}
    for k, bounds in config.get("params", {}).items():
        lo, hi = bounds
        if isinstance(lo, int) and isinstance(hi, int):
            params[k] = int(rng.integers(lo, hi + 1))
        else:
            params[k] = float(rng.uniform(float(lo), float(hi)))
    params.update(config.get("fixed", {}))
    return {
        "probe_type": config["probe"],
        "params": params,
        "target_strategy": config["target"],
        "target_location": None,
        "reason": "phase1b_sweep",
    }


def _apply_global_corruption(image: np.ndarray, corruption_type: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    severity = int(np.clip(severity, 1, 3))
    img = image.astype(np.uint8)

    if corruption_type == "gaussian_noise":
        sigma_map = {1: 8.0, 2: 16.0, 3: 24.0}
        sigma = sigma_map[severity]
        noise = rng.normal(loc=0.0, scale=sigma, size=img.shape)
        out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out

    if corruption_type == "gaussian_blur":
        k_map = {1: 3, 2: 5, 3: 9}
        k = k_map[severity]
        return cv2.GaussianBlur(img, (k, k), 0)

    if corruption_type == "jpeg_compression":
        q_map = {1: 70, 2: 45, 3: 25}
        quality = q_map[severity]
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return img
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return dec if dec is not None else img

    if corruption_type == "brightness_shift":
        shift_map = {1: 22, 2: 45, 3: 70}
        shift = shift_map[severity]
        # apply both brighten and darken so baseline isn't one-sided
        sign = -1 if rng.random() < 0.5 else 1
        out = np.clip(img.astype(np.int16) + sign * shift, 0, 255).astype(np.uint8)
        return out

    return img


def _phase0plus_global(dataset: list[dict], baseline: dict, pool: GPUParserPool, rng: np.random.Generator, csv_path: Path) -> None:
    key_cols = ["image_id", "config_id"]
    done = _load_completed_keys(csv_path, key_cols)
    if done:
        print(f"[resume] Phase0plus: {len(done)} rows already done, resuming …")

    global_defs = {
        "G01": {"type": "gaussian_noise", "severity": 1},
        "G02": {"type": "gaussian_noise", "severity": 2},
        "G03": {"type": "gaussian_noise", "severity": 3},
        "G04": {"type": "gaussian_blur", "severity": 1},
        "G05": {"type": "gaussian_blur", "severity": 2},
        "G06": {"type": "gaussian_blur", "severity": 3},
        "G07": {"type": "jpeg_compression", "severity": 1},
        "G08": {"type": "jpeg_compression", "severity": 2},
        "G09": {"type": "jpeg_compression", "severity": 3},
        "G10": {"type": "brightness_shift", "severity": 1},
        "G11": {"type": "brightness_shift", "severity": 2},
        "G12": {"type": "brightness_shift", "severity": 3},
    }

    for config_id, conf in tqdm(global_defs.items(), desc="Phase0plus global"):
        for item in tqdm(dataset, desc=f"{config_id}", leave=False):
            if (item["image_id"], config_id) in done:
                continue
            adv_image = _apply_global_corruption(item["image"], conf["type"], conf["severity"], rng)

            diff = np.any(adv_image != item["image"], axis=2).astype(np.uint8)
            if int(diff.sum()) == 0:
                diff = np.ones(item["image"].shape[:2], dtype=np.uint8)

            adv_result = pool.parse_one(adv_image)
            pseudo_plan = [
                {
                    "probe_type": f"GLOBAL_{conf['type']}",
                    "params": {"severity": conf["severity"]},
                    "target_strategy": "global",
                    "target_location": None,
                    "reason": "rodla_style_baseline",
                }
            ]
            row = _record_common(
                item,
                "global-corruption",
                config_id,
                pseudo_plan,
                diff,
                adv_image,
                baseline[item["image_id"]],
                adv_result,
                exec_log=None,
            )
            _append_row(csv_path, row)

    print(f"[Phase0plus] Done. CSV → {csv_path}")


def _phase0_baseline(dataset: list[dict], pool: GPUParserPool) -> dict:
    baseline = {}
    for item in tqdm(dataset, desc="Phase0 baseline parse"):
        result = pool.parse_one(item["image"])
        baseline[item["image_id"]] = {
            "result": result,
            "elements": extract_elements_from_result(result),
            "spans": extract_span_elements_from_result(result),
            "text": extract_full_text_from_result(result),
        }
    return baseline


def _baseline_entry_is_usable(entry) -> bool:
    """An entry is usable iff it has non-empty ``elements`` AND ``spans``.

    A cached entry with an empty ``elements``/``spans`` list is almost always
    the fingerprint of a silently-failed parse in a previous run (see
    ``GPUParserPool.parse_one``'s blanket ``except`` returning ``None`` →
    ``extract_elements_from_result(None) == []``).  Treating such entries as
    "complete" leads to ``n_orig_spans = 0`` and all downstream B-SLR / EIR /
    SLR_{miss,topo} values collapsing to zero even though the attack is fine.
    """
    if not isinstance(entry, dict):
        return False
    if "spans" not in entry or "elements" not in entry:
        return False
    # Both zero = prior silent-failure fingerprint. Reparse.
    if len(entry.get("elements", [])) == 0 and len(entry.get("spans", [])) == 0:
        return False
    return True


def _load_or_build_baseline(
    dataset: list[dict],
    pool: GPUParserPool,
    baseline_cache: Path,
    *,
    force_rebuild: bool = False,
) -> dict:
    baseline: dict = {}
    if force_rebuild and baseline_cache.exists():
        try:
            baseline_cache.unlink()
            logging.info("--rebuild_baseline: deleted existing cache %s", baseline_cache)
        except OSError as exc:
            logging.warning("Could not delete baseline cache %s: %s", baseline_cache, exc)

    if baseline_cache.exists() and not force_rebuild:
        try:
            with open(baseline_cache, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict):
                baseline = cached
            else:
                logging.warning("Baseline cache is not a dict, rebuilding: %s", baseline_cache)
        except Exception as exc:
            logging.warning("Failed to load baseline cache, rebuilding: %s (%s)", baseline_cache, exc)

    dataset_ids = {item["image_id"] for item in dataset}
    cached_ids = set(baseline.keys())

    # Entries are "incomplete" if any required key is missing (legacy V2
    # caches) OR if both elements+spans are empty (prior silent parse failure).
    # The old heuristic only checked for the presence of the "spans" key,
    # which let empty-but-present entries persist indefinitely.
    incomplete_ids = {
        img_id for img_id in cached_ids & dataset_ids
        if not _baseline_entry_is_usable(baseline.get(img_id))
    }
    missing_ids = (dataset_ids - cached_ids) | incomplete_ids

    if missing_ids:
        n_empty_reparse = len(incomplete_ids)
        if n_empty_reparse > 0:
            logging.warning(
                "Baseline cache has %d entries with empty elements/spans "
                "(silent-parse-failure fingerprint); reparsing them.",
                n_empty_reparse,
            )
        logging.info(
            "Baseline cache missing %d/%d image_ids, parsing incrementally.",
            len(missing_ids),
            len(dataset_ids),
        )
        missing_items = [item for item in dataset if item["image_id"] in missing_ids]
        for item in tqdm(missing_items, desc="Phase0 baseline backfill"):
            # strict=True: baseline is the "ground truth" for every downstream
            # metric, so a silent None here would poison the whole pipeline.
            # Fail loud instead.
            result = pool.parse_one(item["image"], strict=True)
            baseline[item["image_id"]] = {
                "result": result,
                "elements": extract_elements_from_result(result),
                "spans": extract_span_elements_from_result(result),
                "text": extract_full_text_from_result(result),
            }

        baseline_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_cache, "wb") as f:
            pickle.dump(baseline, f)

    # ── Post-build sanity check (Edit B) ────────────────────────────────
    # Even with strict=True, some parsers may return a non-None ParseResult
    # whose content_list is empty (e.g. MinerU on a page whose layout model
    # genuinely finds nothing, or a pure-image page). That's legitimate for a
    # handful of pages, but NOT for the majority. If >2% of baseline entries
    # in the current dataset are empty, abort with an actionable error
    # message rather than silently producing a CSV full of zeros.
    scoped = {img_id: baseline[img_id] for img_id in dataset_ids if img_id in baseline}
    if scoped:
        empty_ids = [
            img_id for img_id, entry in scoped.items()
            if len(entry.get("elements", [])) == 0
        ]
        empty_frac = len(empty_ids) / max(len(scoped), 1)
        if empty_frac > 0.02:
            sample = sorted(empty_ids)[:10]
            raise RuntimeError(
                f"Baseline build produced empty elements for "
                f"{len(empty_ids)}/{len(scoped)} images "
                f"({empty_frac:.1%} > 2% threshold). This almost always means "
                f"the parser is silently failing on clean images. "
                f"Delete the cache and rerun after verifying the parser: "
                f"`rm {baseline_cache}` ; sample image_ids: {sample}"
            )
        if empty_ids:
            logging.warning(
                "Baseline: %d/%d images have empty elements; they will be "
                "excluded from metric rows downstream.",
                len(empty_ids), len(scoped),
            )

    return baseline


def _record_common(item: dict, policy_name: str, config_id: str, step_plan: list, s_mask: np.ndarray, adv_image: np.ndarray, baseline_item: dict, adv_result, exec_log: list | None = None):
    # ── Guard against an empty clean-image baseline (Edit C) ───────────────
    # If det_orig is empty, every parser-dependent metric (B_SLR, EIR,
    # EIR_det, SLR_miss, SLR_topo, splash_ratio, TextSim, ΔCER, ΔmAP…)
    # collapses to zero by construction, which looks indistinguishable from
    # "the attack did nothing" in the CSV. That's misleading at best and
    # silently poisons aggregated stats at worst. Skip the row and log once.
    orig_elements_preview = baseline_item.get("elements", []) if isinstance(baseline_item, dict) else []
    if not orig_elements_preview:
        img_id = item.get("image_id", "?")
        if img_id not in _EMPTY_BASELINE_WARNED:
            _EMPTY_BASELINE_WARNED.add(img_id)
            logging.warning(
                "Skipping row for image_id=%s (policy=%s, config=%s): "
                "clean baseline has 0 elements; cannot compute B-SLR/EIR/…",
                img_id, policy_name, config_id,
            )
        return None

    adv_elements = extract_elements_from_result(adv_result)
    adv_spans = extract_span_elements_from_result(adv_result)
    adv_text = extract_full_text_from_result(adv_result)

    orig_elements = baseline_item["elements"]
    orig_spans = baseline_item["spans"]
    orig_text = baseline_item["text"]

    # Use content_list (para_block level) for B-SLR, EIR, and splash.
    # MinerU v2.7 produces word-level spans whose non-deterministic
    # re-segmentation inflates B_SLR baseline noise; para_blocks are stable
    # across original/adversarial pairs.  PP-Structure content_list is
    # likewise at the layout-block level and equally stable.
    det_orig, det_adv = orig_elements, adv_elements

    row = {
        "image_id": item["image_id"],
        "data_source": item.get("data_source", "publaynet"),
        "complexity_level": item["complexity"]["level"],
        "N_blocks": item["complexity"]["N_blocks"],
        "H_layout": item["complexity"]["H_layout"],
        "policy": policy_name,
        "config_id": config_id,
        # ── Five-level perturbation metrics (L1–L5) ──
        "TOR": compute_tor(s_mask, item["image"].shape),              # L1: global pixel area
        "ACR": compute_acr(s_mask, item["annotation"]),               # L2: annotation coverage ratio
        "BPO": compute_ncsic_mask(s_mask, item["M_anchor"]),          # L3: boundary pixel overlap
        "BOC": compute_boc(s_mask, item["annotation"]),               # L4: block overlap count
        "EIR": compute_eir(det_orig, s_mask),                         # L5: element interference ratio
        # ── Legacy / auxiliary metrics ──
        "EIR_mask": compute_ncsic_mask(s_mask, item["M_anchor"]),     # same as BPO, kept for compat
        "EIR_det": compute_eir_from_detection(det_orig, det_adv),
        "B_SLR": calculate_b_slr(det_orig, det_adv),
        "B_SLR_iou_only": calculate_b_slr(det_orig, det_adv, iou_only=True),
        "SLR": compute_character_slr(orig_text, adv_text),
        "SSIM": calculate_ssim(item["image"], adv_image),
        "splash_ratio": compute_splash_ratio(det_orig, det_adv, s_mask),
        "n_orig_spans": len(det_orig),
        "n_adv_spans": len(det_adv),
        "target_strategy": ";".join(str(s.get("target_strategy")) for s in step_plan),
        "probe_type": ";".join(str(s.get("probe_type")) for s in step_plan),
        "target_fallback": any(s.get("target_fallback", False) for s in (exec_log or []) if isinstance(s, dict)),
        "plan_json": json.dumps(step_plan, ensure_ascii=False),
    }
    # Pathway decomposition of B-SLR (paper Eqs. 12–20): U is partitioned by
    # omega(x) into U_miss / U_topo, and U_topo is further partitioned into
    # merge / misclass / degraded. By construction the decomposed B_SLR equals
    # the standalone calculate_b_slr(det_orig, det_adv) used for row["B_SLR"],
    # so B-SLR = SLR_miss + SLR_topo = (n_miss+n_merge+n_misclass+n_degraded)/|E|.
    decomp = calculate_decomposed_b_slr(det_orig, det_adv, probe_mask=s_mask)
    assert abs(decomp["B_SLR"] - row["B_SLR"]) < 1e-9, (
        row["image_id"], row["config_id"], decomp["B_SLR"], row["B_SLR"],
    )
    row["SLR_miss"] = decomp["SLR_miss"]
    row["SLR_topo"] = decomp["SLR_topo"]
    row["n_merge"] = decomp["n_merge"]
    row["n_misclass"] = decomp["n_misclass"]
    row["n_degraded"] = decomp["n_degraded"]
    row["n_miss"] = decomp["n_miss"]
    # ── OCR-only ablation: text degradation for geometrically-matched blocks ──
    ocr_abl = calculate_ocr_ablation_metrics(det_orig, det_adv)
    row["TextSim_matched_mean"] = ocr_abl["TextSim_matched_mean"]
    row["TextSim_matched_min"] = ocr_abl["TextSim_matched_min"]
    row["CER_matched_mean"] = ocr_abl["CER_matched_mean"]
    row["n_matched"] = ocr_abl["n_matched"]
    row["n_text_degraded"] = ocr_abl["n_text_degraded"]
    row["B_SLR_text_only"] = ocr_abl["B_SLR_text_only"]
    # ── Terminal Judges: ΔmAP and ΔCER (community-standard metrics) ──
    # ΔCER: standard Levenshtein CER over block-aligned matched elements
    try:
        row["delta_CER"] = compute_delta_cer_from_elements(det_orig, det_adv, iou_threshold=0.1)
    except Exception:
        row["delta_CER"] = None
    # ΔmAP: standard per-class VOC mAP@0.5 against GT annotations
    gt_annotations = item.get("annotation", [])
    if gt_annotations:
        def _elems_to_det_boxes(elems):
            boxes = []
            for i, e in enumerate(elems):
                bbox = e.get("bbox")
                if bbox and len(bbox) >= 4:
                    b = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                else:
                    hw, hh = e.get("w", 0) / 2, e.get("h", 0) / 2
                    b = [e.get("x", 0) - hw, e.get("y", 0) - hh, e.get("x", 0) + hw, e.get("y", 0) + hh]
                score = e.get("confidence", e.get("score", 1.0 - i * 1e-6))
                boxes.append({"bbox": b, "type": e.get("type", "text"), "score": score})
            return boxes
        gt_boxes = [{"bbox": a["bbox"], "type": a.get("category", "text")} for a in gt_annotations]
        clean_boxes = _elems_to_det_boxes(det_orig)
        adv_boxes = _elems_to_det_boxes(det_adv)
        try:
            row["delta_mAP"] = compute_delta_map(gt_boxes, clean_boxes, adv_boxes, iou_threshold=0.5)
        except Exception:
            row["delta_mAP"] = None
    else:
        row["delta_mAP"] = None
    return row


def _phase1a(dataset: list[dict], baseline: dict, engine: AttackEngine, pool: GPUParserPool, rng: np.random.Generator, csv_path: Path, *, static_plan: dict | None = None) -> None:
    key_cols = ["image_id", "config_id"]
    done = _load_completed_keys(csv_path, key_cols)
    if done:
        print(f"[resume] Phase1A: {len(done)} rows already done, resuming …")

    sp_anchor = static_plan.get("phase1a", {}) if static_plan else {}
    sp_eir = static_plan.get("phase1a_eir", {}) if static_plan else {}
    if sp_anchor:
        logging.info("[Phase1A] Using static plan for %d anchor configs", len(sp_anchor))
    if sp_eir:
        logging.info("[Phase1A] Using static plan for %d EIR-targeted configs", len(sp_eir))

    # ── Standard anchor configs (A01–A22) ──
    for config_id, plan_template in tqdm(ANCHOR_CONFIGS.items(), desc="Phase1A configs"):
        for item in tqdm(dataset, desc=f"{config_id}", leave=False):
            if (item["image_id"], config_id) in done:
                continue
            # Use static plan if available; otherwise fall back to template
            plan = sp_anchor.get(config_id, {}).get(item["image_id"]) or plan_template
            adv_image, s_mask, used_plan, elog = engine.execute(item["image"], item, policy=RandomPolicy(), rng=rng, override_plan=plan)
            adv_result = pool.parse_one(adv_image)
            row = _record_common(item, "anchor_override", config_id, used_plan, s_mask, adv_image, baseline[item["image_id"]], adv_result, elog)
            _append_row(csv_path, row)

    # ── EIR-targeted configs (NT01–NT07) ──
    for config_id, nt_conf in tqdm(EIR_TARGETED_CONFIGS.items(), desc="Phase1A EIR-targeted"):
        for item in tqdm(dataset, desc=f"{config_id} eir={nt_conf['target_eir']:.0%}", leave=False):
            if (item["image_id"], config_id) in done:
                continue

            # Static plan: pre-computed positions
            static_steps = sp_eir.get(config_id, {}).get(item["image_id"])
            if static_steps:
                plan = static_steps
            else:
                bl = baseline[item["image_id"]]
                spans = bl["elements"]
                positions = find_positions_for_eir(
                    item["image"].shape,
                    spans,
                    nt_conf["target_eir"],
                    influence_radius=int(nt_conf["params"].get("r", 70)),
                )
                if positions:
                    plan = [
                        {
                            "probe_type": nt_conf["probe_type"],
                            "params": nt_conf["params"],
                            "target_strategy": "eir_targeted",
                            "target_location": pos,
                            "reason": f"eir_target={nt_conf['target_eir']:.2f}",
                        }
                        for pos in positions
                    ]
                else:
                    h, w = item["image"].shape[:2]
                    plan = [
                        {
                            "probe_type": nt_conf["probe_type"],
                            "params": nt_conf["params"],
                            "target_strategy": "eir_targeted",
                            "target_location": (w // 2, h // 2),
                            "reason": f"eir_target={nt_conf['target_eir']:.2f}_fallback",
                        }
                    ]
            bl = baseline[item["image_id"]]
            adv_image, s_mask, used_plan, elog = engine.execute(
                item["image"], item, policy=RandomPolicy(), rng=rng, override_plan=plan
            )
            adv_result = pool.parse_one(adv_image)
            row = _record_common(item, "eir_targeted", config_id, used_plan, s_mask, adv_image, bl, adv_result, elog)
            _append_row(csv_path, row)

    print(f"[Phase1A] Done. CSV → {csv_path}")


def _phase1b(dataset: list[dict], baseline: dict, engine: AttackEngine, pool: GPUParserPool, rng: np.random.Generator, csv_path: Path, *, static_plan: dict | None = None) -> None:
    key_cols = ["image_id", "config_id"]
    done = _load_completed_keys(csv_path, key_cols)
    if done:
        print(f"[resume] Phase1B: {len(done)} rows already done, resuming …")

    sp_sweep = static_plan.get("phase1b", {}) if static_plan else {}
    if sp_sweep:
        logging.info("[Phase1B] Using static plan for %d sweep configs", len(sp_sweep))

    pair_seed_map = {
        "S01": "pair_w_line", "S10": "pair_w_line", "S11": "pair_w_line",
        "S03": "pair_stamp", "S12": "pair_stamp", "S13": "pair_stamp",
    }

    for config_id, conf in tqdm(SWEEP_CONFIGS.items(), desc="Phase1B sweeps"):
        for item_idx, item in enumerate(tqdm(dataset, desc=f"{config_id}", leave=False)):
            if (item["image_id"], config_id) in done:
                continue

            # Static plan: pre-computed parameters + placement
            static_steps = sp_sweep.get(config_id, {}).get(item["image_id"])
            if static_steps:
                step = static_steps[0]
                local_rng = rng
            else:
                if config_id in pair_seed_map:
                    local_rng = np.random.default_rng((item_idx + 1) * 100_000 + (1 if pair_seed_map[config_id] == "pair_w_line" else 2))
                else:
                    local_rng = rng
                step = _sample_sweep_step(conf, local_rng)

            adv_image, s_mask, used_plan, elog = engine.execute(item["image"], item, policy=RandomPolicy(), rng=local_rng, override_plan=[step])
            adv_result = pool.parse_one(adv_image)
            row = _record_common(item, "sweep_override", config_id, used_plan, s_mask, adv_image, baseline[item["image_id"]], adv_result, elog)
            _append_row(csv_path, row)
    print(f"[Phase1B] Done. CSV → {csv_path}")


def _phase2(dataset: list[dict], baseline: dict, engine: AttackEngine, pool: GPUParserPool, rng: np.random.Generator, csv_path: Path, api_key: str | None = None, replay_log: dict | None = None, export_log_path: Path | None = None) -> None:
    key_cols = ["image_id", "config_id"]
    done = _load_completed_keys(csv_path, key_cols)
    if done:
        print(f"[resume] Phase2: {len(done)} rows already done, resuming …")

    # Accumulate attack decisions for export
    attack_log: dict[str, dict[str, dict]] = {}

    if replay_log:
        # ── Replay mode: use pre-recorded attack plans ──
        logging.info("[Phase2] Replay mode: using pre-recorded attack log")
        policy_names = [k for k in replay_log if k != "metadata"]
        for policy_name in tqdm(policy_names, desc="Phase2 replay"):
            for item in tqdm(dataset, desc=policy_name, leave=False):
                if (item["image_id"], policy_name) in done:
                    continue
                entry = replay_log[policy_name].get(item["image_id"], {})
                plan = entry.get("plan")
                if not plan:
                    continue
                adv_image, s_mask, used_plan, elog = engine.execute(
                    item["image"], item, policy=RandomPolicy(), rng=rng, override_plan=plan
                )
                adv_result = pool.parse_one(adv_image)
                row = _record_common(item, policy_name, policy_name, used_plan, s_mask, adv_image, baseline[item["image_id"]], adv_result, elog)
                _append_row(csv_path, row)
    else:
        # ── Live mode: run policies dynamically ──
        policies = [RandomPolicy(), RuleBasedPolicy()]

        resolved_api_key = (api_key or os.environ.get("DEEPSEEK_API_KEY") or "").strip().strip("\"'")
        if resolved_api_key:
            policies.append(LLMPolicyBiased(api_key=resolved_api_key))
            policies.append(LLMPolicyNeutral(api_key=resolved_api_key))
        else:
            print("[phase2] DeepSeek key not found; skipping LLM policies")

        for policy in tqdm(policies, desc="Phase2 policies"):
            attack_log[policy.name] = {}
            for item in tqdm(dataset, desc=policy.name, leave=False):
                if (item["image_id"], policy.name) in done:
                    continue
                adv_image, s_mask, used_plan, elog = engine.execute(item["image"], item, policy=policy, rng=rng)
                adv_result = pool.parse_one(adv_image)
                row = _record_common(item, policy.name, policy.name, used_plan, s_mask, adv_image, baseline[item["image_id"]], adv_result, elog)
                _append_row(csv_path, row)

                # Record for export
                serializable_plan = []
                for step in (used_plan or []):
                    sp = {
                        "probe_type": step.get("probe_type"),
                        "params": step.get("params"),
                        "target_strategy": step.get("target_strategy"),
                    }
                    if step.get("target_location") is not None:
                        sp["target_location"] = list(step["target_location"])
                    serializable_plan.append(sp)
                log_entry = {"plan": serializable_plan}
                if elog:
                    log_entry["exec_log"] = [
                        {k: v for k, v in e.items() if k != "reason"}
                        for e in elog
                    ]
                    if elog[0].get("reason"):
                        log_entry["chain_of_thought"] = elog[0]["reason"]
                attack_log[policy.name][item["image_id"]] = log_entry

    # ── Export attack log for reproducibility ──
    # Never export when replaying: attack_log is empty, and writing to the same
    # path as the replay source would truncate it. Guarded here as defense in
    # depth in addition to the main() arg validation.
    if replay_log:
        if export_log_path:
            logging.info("[Phase2] Replay mode: skipping export of attack log to %s", export_log_path)
    elif export_log_path and attack_log:
        export_log_path = Path(export_log_path)
        # Refuse to shrink an existing log (e.g. live run with fewer policies
        # than what the file already contains). Protects committed replay logs.
        if export_log_path.exists():
            try:
                with open(export_log_path, "r", encoding="utf-8") as _f:
                    _existing = json.load(_f)
                _existing_policies = {k for k in _existing if k != "metadata"}
                _new_policies = set(attack_log.keys())
                _shrinking = _existing_policies - _new_policies
                if _shrinking:
                    logging.error(
                        "[Phase2] Refusing to overwrite %s: existing file has policies "
                        "%s that would be lost (new run only has %s). "
                        "Pass --export_phase2_log to a different path to export anyway.",
                        export_log_path, sorted(_shrinking), sorted(_new_policies),
                    )
                    print(f"[Phase2] Done. CSV → {csv_path}")
                    return
            except Exception as _exc:
                logging.warning("[Phase2] Could not inspect existing export %s: %s", export_log_path, _exc)
        export_log_path.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        export_data = {"metadata": {"created": datetime.now().isoformat()}}
        export_data.update(attack_log)
        with open(export_log_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        logging.info("[Phase2] Attack log exported to %s", export_log_path)

    print(f"[Phase2] Done. CSV → {csv_path}")


def _phase_sensitivity(
    dataset: list[dict], baseline: dict, engine: AttackEngine,
    pool: GPUParserPool, cfg: ExperimentConfig, csv_path: Path,
) -> None:
    """Sensitivity analysis: delta for anchor mask dilation, IoU threshold for B-SLR."""
    from experiment.data.masks import compute_anchor_mask

    key_cols = ["analysis", "param_value", "image_id"]
    done = _load_completed_keys(csv_path, key_cols)
    if done:
        print(f"[resume] Sensitivity: {len(done)} rows already done, resuming …")

    sens = SENSITIVITY_CONFIGS
    delta_conf = sens["delta_sensitivity"]
    iou_conf = sens["iou_threshold_sensitivity"]
    sample_size = min(delta_conf.get("sample_size", 200), len(dataset))
    sample = dataset[:sample_size]

    # ── Part 1: delta sensitivity ──
    for delta_val in tqdm(delta_conf["delta_values"], desc="Sensitivity: delta"):
        seed_rng = np.random.default_rng(cfg.seed)
        for item in tqdm(sample, desc=f"delta={delta_val}", leave=False):
            if ("delta_sensitivity", delta_val, item["image_id"]) in done:
                continue
            h, w = item["image"].shape[:2]
            m_anchor_d = compute_anchor_mask(item["annotation"], (h, w), delta=delta_val)
            item_d = {**item, "M_anchor": m_anchor_d}
            plan = [{**delta_conf["probe_config"], "target_location": None,
                     "reason": f"delta_sens_{delta_val}"}]
            adv_img, s_mask, _, _ = engine.execute(
                item["image"], item_d, policy=RandomPolicy(),
                rng=seed_rng, override_plan=plan,
            )
            adv_result = pool.parse_one(adv_img)
            bl = baseline[item["image_id"]]
            det_orig = bl["elements"]  # para_block level
            adv_elems = extract_elements_from_result(adv_result)
            det_adv = adv_elems
            row = {
                "analysis": "delta_sensitivity",
                "param_name": "delta",
                "param_value": delta_val,
                "image_id": item["image_id"],
                "TOR": compute_tor(s_mask, item["image"].shape),
                "EIR": compute_eir(det_orig, s_mask),
                "B_SLR": calculate_b_slr(det_orig, det_adv),
                "B_SLR_iou_only": calculate_b_slr(det_orig, det_adv, iou_only=True),
            }
            _append_row(csv_path, row)

    # ── Part 2: IoU threshold sensitivity ──
    seed_rng = np.random.default_rng(cfg.seed)
    for item in tqdm(sample, desc="Sensitivity: IoU", leave=False):
        all_iou_done = all(
            ("iou_sensitivity", iou_val, item["image_id"]) in done
            for iou_val in iou_conf["iou_values"]
        )
        if all_iou_done:
            continue
        plan = [{**delta_conf["probe_config"], "target_location": None,
                 "reason": "iou_sens"}]
        adv_img, s_mask, _, _ = engine.execute(
            item["image"], item, policy=RandomPolicy(),
            rng=seed_rng, override_plan=plan,
        )
        adv_result = pool.parse_one(adv_img)
        bl = baseline[item["image_id"]]
        det_orig = bl["elements"]  # para_block level
        adv_elems = extract_elements_from_result(adv_result)
        det_adv = adv_elems
        tor = compute_tor(s_mask, item["image"].shape)
        eir = compute_eir(det_orig, s_mask)
        for iou_val in iou_conf["iou_values"]:
            if ("iou_sensitivity", iou_val, item["image_id"]) in done:
                continue
            row = {
                "analysis": "iou_sensitivity",
                "param_name": "iou_threshold",
                "param_value": iou_val,
                "image_id": item["image_id"],
                "TOR": tor,
                "EIR": eir,
                "B_SLR": calculate_b_slr(det_orig, det_adv, iou_threshold=iou_val),
                "B_SLR_iou_only": calculate_b_slr(det_orig, det_adv, iou_threshold=iou_val, iou_only=True),
            }
            _append_row(csv_path, row)

    print(f"[Sensitivity] Done. CSV → {csv_path}")


def _should_run(phases: set[str], key: str) -> bool:
    return key in phases or "all" in phases


def main():
    parser = argparse.ArgumentParser(description="Probe Space × Policy experiment runner")
    parser.add_argument("--run_mode", type=str, default="formal", choices=["pilot", "formal"])
    parser.add_argument("--n_images", type=int, default=None,
                        help="Override image count. Default: pilot=100, formal=1000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=int, default=5)
    parser.add_argument("--phases", type=str, default="0,0plus,1a,1b,2")
    parser.add_argument("--data_mode", type=str, default="selected", choices=["legacy", "selected"],
                        help="'legacy'=PubLayNet only (old path), 'selected'=PubLayNet+DocLayNet from data/selected/")
    parser.add_argument("--output_dir", type=str, default="experiment/output")
    parser.add_argument("--deepseek_api_key", type=str, default=None,
                        help="DeepSeek API key for LLM policy in phase 2")
    parser.add_argument("--parser", type=str, default="mineru",
                        choices=["mineru", "ppstructure"],
                        help="Target DLA parser: 'mineru' (default) or 'ppstructure' (PP-StructureV3).")
    parser.add_argument("--data_source", type=str, default=None,
                        choices=["publaynet", "doclaynet"],
                        help="Filter selected dataset by source (default: use both)")
    parser.add_argument("--doc_categories", type=str, default=None,
                        help="Comma-separated DocLayNet doc categories to keep "
                             "(e.g. 'financial_reports,manuals'). Only effective with --data_mode selected.")
    parser.add_argument("--shared_dataset", type=str, default=None,
                        help="Path to shared_eval_set.json. Bypasses sampling/backfill/trim; "
                             "loads exactly the listed images for cross-pipeline reproducibility.")
    parser.add_argument("--static_plan", type=str, default=None,
                        help="Path to static_attack_plan.json (Phase 1 only). "
                             "Provides pre-computed probe placement for pixel-identical attacks "
                             "across pipelines. Generated by generate_static_attack_plan.py.")
    parser.add_argument("--export_phase2_log", type=str, default=None,
                        help="Export Phase 2 attack decisions to JSON for reproducibility. "
                             "Default: config/phase2_attack_log_{parser}.json when Phase 2 runs.")
    parser.add_argument("--replay_phase2_log", type=str, default=None,
                        help="Replay Phase 2 from a previously exported attack log JSON "
                             "instead of running live policies.")
    parser.add_argument("--rebuild_baseline", action="store_true",
                        help="Ignore and overwrite any existing baseline cache "
                             "(output_dir/cache/baseline*.pkl). Use after a parser "
                             "fix when the cached baseline may contain silent "
                             "parse-failure fingerprints (empty elements/spans).")
    args = parser.parse_args()

    # Safety: never let --export_phase2_log clobber --replay_phase2_log.
    # An earlier bug where these paths silently defaulted to the same file
    # caused a live 2-policy run to truncate the committed 5-policy replay log.
    if args.export_phase2_log and args.replay_phase2_log:
        _exp_resolved = Path(args.export_phase2_log).resolve()
        _rep_resolved = Path(args.replay_phase2_log).resolve()
        if _exp_resolved == _rep_resolved:
            parser.error(
                f"--export_phase2_log and --replay_phase2_log resolve to the same file "
                f"({_exp_resolved}); refusing to overwrite the replay source. "
                f"Point --export_phase2_log at a different path (e.g. under --output_dir)."
            )

    setup_advdoc_paths()
    setup_cuda_dll_paths()

    n_images = args.n_images if args.n_images is not None else (100 if args.run_mode == "pilot" else 1000)

    cfg = ExperimentConfig(
        run_mode=args.run_mode,
        n_images=n_images,
        seed=args.seed,
        delta=args.delta,
        output_dir=Path(args.output_dir),
        data_mode=args.data_mode,
    )
    ensure_output_dirs(cfg)

    phases = {p.strip().lower() for p in args.phases.split(",")}

    # ── Shared dataset mode (cross-pipeline reproducibility) ──
    if args.shared_dataset:
        _shared_path = Path(args.shared_dataset)
        dataset = build_shared_dataset(
            shared_index_path=_shared_path,
            data_root=cfg.data_root,
            delta=cfg.delta,
            long_edge=cfg.long_edge,
            selected_root=cfg.selected_root,
        )
        print(f"[data] Loaded {len(dataset)} images from shared index: {_shared_path}")
        _candidates_df = None
        _ann_map = None
    elif cfg.data_mode == "selected":
        dataset = build_combined_dataset(
            selected_root=cfg.selected_root,
            delta=cfg.delta,
            long_edge=cfg.long_edge,
        )
        print(f"[data] Loaded {len(dataset)} images from selected/ (dual-source)")

        # ── Filter by data_source (publaynet / doclaynet) ──
        if args.data_source:
            pre = len(dataset)
            dataset = [d for d in dataset if d.get("data_source") == args.data_source]
            print(f"[data] Filtered by data_source={args.data_source}: {pre} → {len(dataset)}")

        # ── Filter by doc_categories (DocLayNet only) ──
        if args.doc_categories:
            cats = {c.strip() for c in args.doc_categories.split(",") if c.strip()}
            pre = len(dataset)
            dataset = [d for d in dataset
                       if d.get("data_source") != "doclaynet" or d.get("doc_category") in cats]
            print(f"[data] Filtered by doc_categories={cats}: {pre} → {len(dataset)}")

        _candidates_df = None
        _ann_map = None
    else:
        # Oversample ~20% to have room after cleaning, but keep the full
        # candidate pool so we can backfill from the same complexity level
        # if too many images are filtered out.
        _oversample_n = int(cfg.n_images * 1.2)
        dataset, _candidates_df, _ann_map = build_stratified_dataset_with_pool(
            data_root=cfg.data_root,
            n_total=_oversample_n,
            delta=cfg.delta,
            seed=cfg.seed,
            long_edge=cfg.long_edge,
        )

    rng = np.random.default_rng(cfg.seed)
    engine = AttackEngine()
    pool = GPUParserPool(num_workers=cfg.parser_workers, parser_name=args.parser)
    pool.warmup()

    # When using alternative parser, suffix CSV filenames to avoid overwriting MinerU results
    _parser_suffix = f"_{args.parser.replace('-', '_')}" if args.parser != "mineru" else ""

    baseline_cache = cfg.output_dir / "cache" / f"baseline{_parser_suffix}.pkl"

    need_baseline = any(_should_run(phases, key) for key in ["0", "0plus", "1a", "1b", "2", "sensitivity"])
    baseline = (
        _load_or_build_baseline(
            dataset, pool, baseline_cache,
            force_rebuild=args.rebuild_baseline,
        )
        if need_baseline else {}
    )

    # ── Filter out images with no text spans / no elements ─────────────
    # Historical behaviour skipped this filter entirely in --shared_dataset
    # mode, relying on generate_shared_dataset.py's ``n_elements >= 5``
    # guarantee. But that guarantee only reflects the parser state at
    # index-build time; if the *current* baseline parse silently failed for
    # some images, the shared index will not protect us. So: always run the
    # safety filter, and in shared_dataset mode log it loudly (so any
    # unexpected drop is visible) while NOT doing dataset backfill/trim
    # (that's still the shared index's job).
    _bad_ids: set[str] = set()
    if baseline and args.parser in ("mineru", "ppstructure"):
        pre_filter = len(dataset)
        no_span_ids = {
            img_id for img_id, bl in baseline.items()
            if len(bl.get("spans", [])) == 0 or len(bl.get("elements", [])) == 0
        }
        _bad_ids.update(no_span_ids)
        if no_span_ids:
            dataset = [item for item in dataset if item["image_id"] not in no_span_ids]
            log_fn = logging.warning if args.shared_dataset else logging.info
            log_fn(
                "Data cleaning: removed %d/%d images with 0 text spans or 0 "
                "elements (pure image/table pages OR silent parse failure)%s: %s",
                pre_filter - len(dataset), pre_filter,
                " [shared_dataset]" if args.shared_dataset else "",
                sorted(no_span_ids)[:20] + (["…"] if len(no_span_ids) > 20 else []),
            )
    if args.shared_dataset:
        logging.info(
            "Shared dataset mode: skipping runtime backfill/trim "
            "(already joint-cleaned by generate_shared_dataset.py)."
        )

    # ── Backfill from unused candidates to reach exact target ──────────────
    # Skipped in shared dataset mode (already at exact count).
    _MAX_BACKFILL_ROUNDS = 5
    _backfill_rng = np.random.default_rng(cfg.seed + 2)
    if not args.shared_dataset and _candidates_df is not None and _ann_map is not None and baseline:
        for _round in range(_MAX_BACKFILL_ROUNDS):
            if len(dataset) >= cfg.n_images:
                break
            deficit = cfg.n_images - len(dataset)
            used_ids = {item["image_id"] for item in dataset} | _bad_ids
            # Count deficit per complexity level
            by_level_count: dict[str, int] = {}
            for item in dataset:
                lvl = item["complexity"]["level"]
                by_level_count[lvl] = by_level_count.get(lvl, 0) + 1
            n_simple_t = int(round(cfg.n_images * 0.3))
            n_medium_t = int(round(cfg.n_images * 0.4))
            n_complex_t = cfg.n_images - n_simple_t - n_medium_t
            level_targets = {"simple": n_simple_t, "medium": n_medium_t, "complex": n_complex_t}
            level_deficit = {lvl: max(0, level_targets[lvl] - by_level_count.get(lvl, 0))
                            for lvl in level_targets}
            # If all level deficits are 0, fill remaining from any level proportionally
            if sum(level_deficit.values()) == 0:
                for lvl in level_targets:
                    level_deficit[lvl] = deficit  # allow any level to fill

            backfill_files: list[str] = []
            for lvl, need in level_deficit.items():
                if need <= 0:
                    continue
                avail = _candidates_df[
                    (_candidates_df["level"] == lvl) &
                    (~_candidates_df["file_name"].isin(used_ids))
                ]["file_name"].tolist()
                if not avail:
                    continue
                # Sample up to 2x the deficit to account for potential further filtering
                n_sample = min(len(avail), need * 2)
                idx = _backfill_rng.choice(len(avail), size=n_sample, replace=False)
                backfill_files.extend(avail[i] for i in idx)

            if not backfill_files:
                logging.warning(
                    "Backfill round %d: no more candidates available in the pool, "
                    "stopping at %d images.", _round + 1, len(dataset))
                break

            logging.info("Backfill round %d: loading %d candidate images to fill deficit of %d",
                         _round + 1, len(backfill_files), deficit)
            new_items = load_images_by_filenames(
                cfg.data_root, backfill_files, _ann_map,
                delta=cfg.delta, long_edge=cfg.long_edge,
            )
            # Build baselines for new items
            new_baseline = _load_or_build_baseline(new_items, pool, baseline_cache)
            baseline.update(new_baseline)

            # Filter new items the same way
            new_bad = {img_id for img_id, bl in new_baseline.items()
                       if img_id in {it["image_id"] for it in new_items} and len(bl["spans"]) == 0}
            _bad_ids.update(new_bad)
            good_items = [item for item in new_items if item["image_id"] not in new_bad]
            if new_bad:
                logging.info("Backfill round %d: filtered %d/%d new images (0 spans)",
                             _round + 1, len(new_bad), len(new_items))
            dataset.extend(good_items)
            logging.info("Backfill round %d: dataset now has %d images", _round + 1, len(dataset))

    # ── Trim back to exact target size maintaining stratification ──────────
    if not args.shared_dataset and len(dataset) > cfg.n_images:
        by_level: dict[str, list[dict]] = {}
        for item in dataset:
            by_level.setdefault(item["complexity"]["level"], []).append(item)
        n_simple = int(round(cfg.n_images * 0.3))
        n_medium = int(round(cfg.n_images * 0.4))
        n_complex = cfg.n_images - n_simple - n_medium
        _level_targets = {"simple": n_simple, "medium": n_medium, "complex": n_complex}
        _trim_rng = np.random.default_rng(cfg.seed + 1)   # separate stream
        trimmed: list[dict] = []
        for _lvl, _n_t in _level_targets.items():
            _pool = by_level.get(_lvl, [])
            if len(_pool) <= _n_t:
                trimmed.extend(_pool)
            else:
                _idx = _trim_rng.choice(len(_pool), size=_n_t, replace=False)
                trimmed.extend(_pool[i] for i in sorted(_idx))
        if len(trimmed) < cfg.n_images:
            _used = {item["image_id"] for item in trimmed}
            _extra = [item for item in dataset if item["image_id"] not in _used]
            trimmed.extend(_extra[: cfg.n_images - len(trimmed)])
        dataset = trimmed[: cfg.n_images]
        logging.info("Trimmed to exact target: %d images (stratified)", len(dataset))
    elif len(dataset) < cfg.n_images:
        logging.warning(
            "After cleaning + backfill, only %d/%d images available "
            "(candidate pool exhausted).",
            len(dataset), cfg.n_images,
        )
    logging.info("Final dataset: %d images", len(dataset))

    # ── Load static attack plan for Phase 1 (cross-pipeline reproducibility) ──
    _static_plan: dict | None = None
    if args.static_plan:
        _sp = Path(args.static_plan)
        if _sp.exists():
            with open(_sp, "r", encoding="utf-8") as f:
                _static_plan = json.load(f)
            logging.info("Loaded static attack plan from %s (%d Phase 1A + %d EIR + %d Phase 1B configs)",
                         _sp,
                         len(_static_plan.get("phase1a", {})),
                         len(_static_plan.get("phase1a_eir", {})),
                         len(_static_plan.get("phase1b", {})))
        else:
            logging.warning("Static plan not found: %s — falling back to dynamic mode", _sp)

    # ── Load Phase 2 replay log if provided ──
    _phase2_replay: dict | None = None
    if args.replay_phase2_log:
        _rp = Path(args.replay_phase2_log)
        if _rp.exists():
            with open(_rp, "r", encoding="utf-8") as f:
                _phase2_replay = json.load(f)
            logging.info("Loaded Phase 2 replay log from %s", _rp)
        else:
            logging.warning("Phase 2 replay log not found: %s — running live policies", _rp)

    phase0plus_csv = cfg.output_dir / f"phase0plus_global{_parser_suffix}.csv"
    phase1a_csv = cfg.output_dir / f"phase1a_anchors{_parser_suffix}.csv"
    phase1b_csv = cfg.output_dir / f"phase1b_sweeps{_parser_suffix}.csv"
    phase2_csv = cfg.output_dir / f"phase2_policies{_parser_suffix}.csv"
    sens_csv = cfg.output_dir / f"sensitivity{_parser_suffix}.csv"

    if _should_run(phases, "0plus"):
        _phase0plus_global(dataset, baseline, pool, rng, csv_path=phase0plus_csv)

    if _should_run(phases, "1a"):
        _phase1a(dataset, baseline, engine, pool, rng, csv_path=phase1a_csv,
                 static_plan=_static_plan)

    if _should_run(phases, "1b"):
        _phase1b(dataset, baseline, engine, pool, rng, csv_path=phase1b_csv,
                 static_plan=_static_plan)

    if _should_run(phases, "2"):
        _phase2_log_path = None
        if args.export_phase2_log:
            _phase2_log_path = Path(args.export_phase2_log)
        elif args.shared_dataset and not args.replay_phase2_log:
            # Only auto-derive an export path when NOT replaying, otherwise
            # a live run with fewer policies would truncate the replay log.
            _phase2_log_path = Path(f"config/phase2_attack_log{_parser_suffix}.json")
        _phase2(dataset, baseline, engine, pool, rng, csv_path=phase2_csv,
                api_key=args.deepseek_api_key, replay_log=_phase2_replay,
                export_log_path=_phase2_log_path)

    if _should_run(phases, "sensitivity"):
        _phase_sensitivity(dataset, baseline, engine, pool, cfg, csv_path=sens_csv)

    del dataset, baseline, engine
    import gc; gc.collect()


if __name__ == "__main__":
    main()
