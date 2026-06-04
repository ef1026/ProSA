from __future__ import annotations

import numpy as np

from .base import PolicyBase


class RandomPolicy(PolicyBase):
    PARAM_RANGES = {
        "P1": {"w": (1, 10), "l_ratio": (0.5, 1.0)},
        "P2": {"w": (1, 10), "l_ratio": (0.5, 1.0)},
        "P3": {"r": (30, 90), "alpha": (0.2, 1.0)},
        "P4": {"area_ratio": (0.03, 0.25), "beta": (0.2, 1.0)},
        "P5": {"w": (1, 5), "l_ratio": (0.2, 0.8)},
        "P6": {"alpha": (0.05, 0.4), "w": (2, 10)},
        "P8": {"n_points": (10, 100), "r": (1, 4), "sigma": (10, 50)},
        "P9": {"r_base": (30, 80), "epsilon": (0.1, 0.5), "alpha": (0.3, 0.7)},
        "P10": {"theta": (20, 70), "w": (1, 6)},
    }
    PROBE_TYPES = list(PARAM_RANGES.keys())
    TARGET_STRATEGIES = ["anchor", "content", "random", "bridge"]

    @property
    def name(self):
        return "random"

    def select(self, context, rng: np.random.Generator) -> list:
        n_probes = int(rng.integers(1, 4))
        plan = []
        for _ in range(n_probes):
            ptype = str(rng.choice(self.PROBE_TYPES))
            params = {}
            for k, (lo, hi) in self.PARAM_RANGES[ptype].items():
                if isinstance(lo, int) and isinstance(hi, int):
                    params[k] = int(rng.integers(lo, hi + 1))
                else:
                    params[k] = float(rng.uniform(lo, hi))
            plan.append(
                {
                    "probe_type": ptype,
                    "params": params,
                    "target_strategy": str(rng.choice(self.TARGET_STRATEGIES)),
                    "target_location": None,
                    "reason": "random",
                }
            )
        return plan
