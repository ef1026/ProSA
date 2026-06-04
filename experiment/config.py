from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class ExperimentConfig:
    data_root: Path = None  # type: ignore[assignment]
    selected_root: Path = None  # type: ignore[assignment]
    output_dir: Path = None  # type: ignore[assignment]
    data_mode: str = "legacy"  # "legacy" (PubLayNet only) or "selected" (PubLayNet+DocLayNet)
    run_mode: str = "formal"  # "pilot" or "formal"
    n_images: int = 1000
    seed: int = 42
    delta: int = 5
    long_edge: int = 1024
    parser_workers: int = 1
    parser_lang: str = "en"
    # Valid parsers: "mineru", "ppstructure"
    parser_name: str = "mineru"

    def __post_init__(self) -> None:
        root = _project_root()
        if self.data_root is None:
            self.data_root = root / "data" / "publaynet"
        if self.selected_root is None:
            self.selected_root = root / "data" / "selected"
        if self.output_dir is None:
            self.output_dir = root / "experiment" / "output"


ANCHOR_CONFIGS = {
    "A01": [{"probe_type": "P1", "params": {"w": 1, "l_ratio": 1.0}, "target_strategy": "anchor"}],
    "A02": [{"probe_type": "P1", "params": {"w": 8, "l_ratio": 1.0}, "target_strategy": "anchor"}],
    "A03": [{"probe_type": "P2", "params": {"w": 1, "l_ratio": 1.0}, "target_strategy": "anchor"}],
    "A04": [{"probe_type": "P2", "params": {"w": 8, "l_ratio": 1.0}, "target_strategy": "anchor"}],
    "A05": [{"probe_type": "P3", "params": {"r": 60, "alpha": 0.3}, "target_strategy": "anchor"}],
    "A06": [{"probe_type": "P3", "params": {"r": 60, "alpha": 1.0}, "target_strategy": "anchor"}],
    "A07": [{"probe_type": "P4", "params": {"area_ratio": 0.05, "beta": 0.3}, "target_strategy": "content"}],
    "A08": [{"probe_type": "P4", "params": {"area_ratio": 0.20, "beta": 1.0}, "target_strategy": "content"}],
    "A09": [{"probe_type": "P5", "params": {"w": 1, "l_ratio": 0.5}, "target_strategy": "bridge"}],
    "A10": [{"probe_type": "P5", "params": {"w": 3, "l_ratio": 0.5}, "target_strategy": "bridge"}],
    "A11": [{"probe_type": "P6", "params": {"alpha": 0.1, "w": 5}, "target_strategy": "anchor"}],
    "A12": [{"probe_type": "P6", "params": {"alpha": 0.3, "w": 5}, "target_strategy": "anchor"}],
    "A13": [{"probe_type": "P1", "params": {"w": 3, "l_ratio": 1.0}, "target_strategy": "content"}],
    "A14": [{"probe_type": "P1", "params": {"w": 3, "l_ratio": 1.0}, "target_strategy": "random"}],
    "A15": [{"probe_type": "P3", "params": {"r": 60, "alpha": 0.5}, "target_strategy": "content"}],
    "A16": [{"probe_type": "P3", "params": {"r": 60, "alpha": 0.5}, "target_strategy": "random"}],
    "A17": [{"probe_type": "P5", "params": {"w": 2, "l_ratio": 0.5}, "target_strategy": "content"}],
    "A18": [{"probe_type": "P5", "params": {"w": 2, "l_ratio": 0.5}, "target_strategy": "random"}],
    # ── Control Group B strict pairs (same target, different probe) ──
    # A19: P4 on bridge (pairs with A10-P5 on bridge for granularity comparison)
    "A19": [{"probe_type": "P4", "params": {"area_ratio": 0.20, "beta": 1.0}, "target_strategy": "bridge"}],
    # A20: P5 on content (pairs with A08-P4 on content for granularity comparison)
    "A20": [{"probe_type": "P5", "params": {"w": 3, "l_ratio": 0.5}, "target_strategy": "content"}],
    # ── Control Group A completion: P3 with identical params on anchor ──
    # A21: pairs with A15(content)/A16(random) for structure mediation triplet
    "A21": [{"probe_type": "P3", "params": {"r": 60, "alpha": 0.5}, "target_strategy": "anchor"}],
    # ── Control Group C completion: P1 w=3 on anchor ──
    # A22: pairs with A13(content)/A14(random) for splash effect triplet
    "A22": [{"probe_type": "P1", "params": {"w": 3, "l_ratio": 1.0}, "target_strategy": "anchor"}],
}


# ── EIR-targeted configs ───────────────────────────────────────────
# Stamp probes (P3) deliberately placed to achieve specific EIR targets.
# find_positions_for_eir() will greedily place multiple stamps per image.
EIR_TARGETED_CONFIGS = {
    "NT01": {"probe_type": "P3", "params": {"r": 70, "alpha": 0.55}, "target_eir": 0.05},
    "NT02": {"probe_type": "P3", "params": {"r": 70, "alpha": 0.55}, "target_eir": 0.10},
    "NT03": {"probe_type": "P3", "params": {"r": 70, "alpha": 0.55}, "target_eir": 0.20},
    "NT04": {"probe_type": "P3", "params": {"r": 70, "alpha": 0.55}, "target_eir": 0.40},
    "NT05": {"probe_type": "P3", "params": {"r": 70, "alpha": 0.55}, "target_eir": 0.60},
    "NT06": {"probe_type": "P3", "params": {"r": 70, "alpha": 0.55}, "target_eir": 0.80},
    "NT07": {"probe_type": "P3", "params": {"r": 70, "alpha": 0.55}, "target_eir": 1.00},
}
NCSIC_TARGETED_CONFIGS = EIR_TARGETED_CONFIGS  # backward compat alias


SWEEP_CONFIGS = {
    "S01": {"probe": "P1", "params": {"w": (1, 10)}, "fixed": {"l_ratio": 1.0}, "target": "anchor"},
    "S02": {"probe": "P2", "params": {"w": (1, 10)}, "fixed": {"l_ratio": 1.0}, "target": "anchor"},
    "S03": {"probe": "P3", "params": {"r": (30, 90), "alpha": (0.2, 1.0)}, "fixed": {}, "target": "anchor"},
    "S04": {"probe": "P4", "params": {"area_ratio": (0.03, 0.25), "beta": (0.2, 1.0)}, "fixed": {}, "target": "content"},
    "S05": {"probe": "P5", "params": {"w": (1, 5), "l_ratio": (0.2, 0.8)}, "fixed": {}, "target": "bridge"},
    "S06": {"probe": "P6", "params": {"alpha": (0.05, 0.4), "w": (2, 10)}, "fixed": {}, "target": "anchor"},
    "S07": {"probe": "P7", "params": {"n_points": (10, 100), "r": (1, 4), "sigma": (10, 50)}, "fixed": {}, "target": "random"},
    "S08": {"probe": "P8", "params": {"r_base": (30, 80), "epsilon": (0.1, 0.5), "alpha": (0.3, 0.7)}, "fixed": {}, "target": "anchor"},
    "S09": {"probe": "P9", "params": {"theta": (20, 70), "w": (1, 6)}, "fixed": {}, "target": "anchor"},
    "S10": {"probe": "P1", "params": {"w": (1, 10)}, "fixed": {"l_ratio": 1.0}, "target": "content"},
    "S11": {"probe": "P1", "params": {"w": (1, 10)}, "fixed": {"l_ratio": 1.0}, "target": "random"},
    "S12": {"probe": "P3", "params": {"r": (30, 90), "alpha": (0.2, 1.0)}, "fixed": {}, "target": "content"},
    "S13": {"probe": "P3", "params": {"r": (30, 90), "alpha": (0.2, 1.0)}, "fixed": {}, "target": "random"},
}


# ── Sensitivity analysis configs (storyline §5.9) ──────────────────
SENSITIVITY_CONFIGS = {
    "delta_sensitivity": {
        "description": "Vary delta for anchor mask construction",
        "delta_values": [3, 5, 7, 10],
        "probe_config": {
            "probe_type": "P5",
            "params": {"w": 3, "l_ratio": 0.5},
            "target_strategy": "anchor",
        },
        "sample_size": 200,
    },
    "iou_threshold_sensitivity": {
        "description": "Vary IoU threshold for B-SLR matching",
        "iou_values": [0.05, 0.1, 0.2, 0.3],
    },
}


def setup_advdoc_paths(project_root: Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def setup_cuda_dll_paths() -> None:
    if sys.platform != "win32":
        return
    candidates = [
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia",
    ]
    subdirs = ["cublas", "cuda_runtime", "cudnn", "cusparse", "cusolver", "cufft", "curand"]
    for base in candidates:
        if not base.exists():
            continue
        for sub in subdirs:
            dll_dir = base / sub / "bin"
            if dll_dir.exists():
                try:
                    os.add_dll_directory(str(dll_dir))
                except Exception:
                    pass


def get_phase1a_configs() -> dict[str, list[dict[str, Any]]]:
    return ANCHOR_CONFIGS


def get_eir_targeted_configs() -> dict[str, dict[str, Any]]:
    return EIR_TARGETED_CONFIGS


get_ncsic_targeted_configs = get_eir_targeted_configs  # backward compat alias


def get_phase1b_configs() -> dict[str, dict[str, Any]]:
    return SWEEP_CONFIGS


def ensure_output_dirs(cfg: ExperimentConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "probe_samples").mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "cache").mkdir(parents=True, exist_ok=True)
