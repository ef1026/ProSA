from __future__ import annotations

import numpy as np

from .context import ContextEncoder
from .probes.action import BlendAction, EraseAction, InjectAction
from .probes.geometry import BlobGeometry, DiskGeometry, LineGeometry, PointGeometry, RectGeometry
from .probes.target import AnchorTarget, BridgeTarget, ContentTarget, RandomTarget
from .probes.visual import GradientVisual, NoiseVisual, RingVisual, SolidVisual


class AttackEngine:
    def __init__(self):
        self.context_encoder = ContextEncoder()

    def execute(self, image, data, policy, rng, override_plan=None):
        context = self.context_encoder.encode(image, data["annotation"], data["M_anchor"], data["M_content"])
        attack_plan = override_plan if override_plan is not None else policy.select(context, rng)

        i_perturbed = image.copy()
        s_union = np.zeros(image.shape[:2], dtype=np.uint8)
        exec_log = []

        for step_idx, step in enumerate(attack_plan):
            probe = self._build_probe(step, data, rng)
            if step.get("target_location") is not None:
                center = (int(step["target_location"][0]), int(step["target_location"][1]))
            else:
                target = self._get_target_strategy(step.get("target_strategy", "anchor"))
                layout_info = {
                    "M_anchor": data["M_anchor"],
                    "M_content": data["M_content"],
                    "H": data["H"],
                    "W": data["W"],
                    "annotations": data["annotation"],
                }
                center = target.place(probe["geometry"], layout_info, rng)

            h, w = image.shape[:2]
            s = probe["geometry"].generate((h, w), center)

            if isinstance(probe["visual"], RingVisual):
                v, ring_mask = probe["visual"].render(s, center, int(step["params"].get("r", 60)))
                s = ring_mask
            else:
                v = probe["visual"].render(s)

            i_perturbed = probe["action"].composite(i_perturbed, s, v)
            s_union = np.maximum(s_union, s)

            # Detect bridge fallback — `target` only exists when target_location is None
            _fallback = False
            if step.get("target_location") is None:
                _fallback = getattr(target, "last_fallback", False)

            exec_log.append(
                {
                    "step": step_idx,
                    "probe_type": step["probe_type"],
                    "params": step["params"],
                    "center": center,
                    "target_strategy": step.get("target_strategy"),
                    "pixels_affected": int(s.sum()),
                    "reason": step.get("reason", ""),
                    "target_fallback": _fallback,
                }
            )

        return i_perturbed, s_union, attack_plan, exec_log

    def _build_probe(self, step, data, rng):
        p = step["params"]
        ptype = step["probe_type"]
        h, w = data["H"], data["W"]

        builders = {
            "P1": lambda: {"geometry": LineGeometry(theta=0, length_ratio=p.get("l_ratio", 1.0), width=p.get("w", 3)), "visual": SolidVisual((0, 0, 0)), "action": InjectAction()},
            "P2": lambda: {"geometry": LineGeometry(theta=90, length_ratio=p.get("l_ratio", 1.0), width=p.get("w", 3)), "visual": SolidVisual((0, 0, 0)), "action": InjectAction()},
            "P3": lambda: {"geometry": DiskGeometry(rx=int(p.get("r", 60)), ry=int(p.get("r", 60))), "visual": RingVisual((220, 50, 50)), "action": BlendAction(alpha=p.get("alpha", 0.5))},
            "P4": lambda: {
                "geometry": RectGeometry(w_rect=int(np.sqrt(p.get("area_ratio", 0.1)) * min(h, w)), h_rect=int(np.sqrt(p.get("area_ratio", 0.1)) * min(h, w))),
                "visual": SolidVisual((255, 255, 255)),
                "action": EraseAction(beta=p.get("beta", 0.5)),
            },
            "P5": lambda: {"geometry": LineGeometry(theta=0, length_ratio=p.get("l_ratio", 0.5), width=p.get("w", 2)), "visual": SolidVisual((0, 0, 0)), "action": InjectAction()},
            "P6": lambda: {"geometry": LineGeometry(theta=0, length_ratio=1.0, width=p.get("w", 5)), "visual": GradientVisual((0, 0, 0), (200, 200, 200)), "action": BlendAction(alpha=p.get("alpha", 0.15))},
            "P8": lambda: {"geometry": PointGeometry(radius=int(p.get("r", 2)), n_points=int(p.get("n_points", 50)), spread_sigma=p.get("sigma", 30), rng=rng), "visual": SolidVisual((0, 0, 0)), "action": InjectAction()},
            "P9": lambda: {"geometry": BlobGeometry(r_base=int(p.get("r_base", 50)), epsilon=p.get("epsilon", 0.3), rng=rng), "visual": NoiseVisual((139, 90, 43), noise_scale=0.02, color_variation=0.3, seed=int(rng.integers(0, 2**31))), "action": BlendAction(alpha=p.get("alpha", 0.5))},
            "P10": lambda: {"geometry": LineGeometry(theta=p.get("theta", 45), length_ratio=1.0, width=p.get("w", 3)), "visual": SolidVisual((80, 80, 80)), "action": InjectAction()},
            # Legacy aliases for backward compatibility with older outputs/configs
            "P7": lambda: {"geometry": PointGeometry(radius=int(p.get("r", 2)), n_points=int(p.get("n_points", 50)), spread_sigma=p.get("sigma", 30), rng=rng), "visual": SolidVisual((0, 0, 0)), "action": InjectAction()},
        }
        return builders.get(ptype, builders["P1"])()

    def _get_target_strategy(self, name):
        return {
            "anchor": AnchorTarget(),
            "content": ContentTarget(),
            "random": RandomTarget(),
            "bridge": BridgeTarget(),
        }.get(name, AnchorTarget())
