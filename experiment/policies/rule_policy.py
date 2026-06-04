from __future__ import annotations

from .base import PolicyBase


class RuleBasedPolicy(PolicyBase):
    @property
    def name(self):
        return "rule-based"

    def select(self, context, rng) -> list:
        plan = []
        vulnerable = context.spatial["vulnerable_points"]
        if len(vulnerable) > 0:
            weakest = vulnerable[0]
            if weakest["width"] < 15:
                plan.append(
                    {
                        "probe_type": "P5",
                        "params": {"w": max(1.0, weakest["width"] * 0.4), "l_ratio": float(rng.uniform(0.3, 0.6))},
                        "target_strategy": "bridge",
                        "target_location": weakest["center"],
                        "reason": "Exploit narrow structural gap",
                    }
                )
            elif weakest["width"] < 40:
                if weakest["direction"] == "vertical_gap":
                    plan.append(
                        {
                            "probe_type": "P1",
                            "params": {"w": float(rng.uniform(2, 5)), "l_ratio": 1.0},
                            "target_strategy": "anchor",
                            "target_location": weakest["center"],
                            "reason": "Crease through vertical gap",
                        }
                    )
                else:
                    plan.append(
                        {
                            "probe_type": "P2",
                            "params": {"w": float(rng.uniform(2, 5)), "l_ratio": 1.0},
                            "target_strategy": "anchor",
                            "target_location": weakest["center"],
                            "reason": "Crease through horizontal gap",
                        }
                    )

        if context.spatial["is_multi_column"] and len(plan) < 2:
            plan.append(
                {
                    "probe_type": "P2",
                    "params": {"w": float(rng.uniform(2, 6)), "l_ratio": 1.0},
                    "target_strategy": "anchor",
                    "target_location": None,
                    "reason": "Multi-column bridge attack",
                }
            )

        if context.layout["has_table"] and len(plan) < 2:
            table_anns = [a for a in context.annotations if a["category"] == "Table"]
            if table_anns:
                tb = table_anns[0]["bbox"]
                edge_x = int(tb[2])
                edge_y = int((tb[1] + tb[3]) // 2)
                plan.append(
                    {
                        "probe_type": "P3",
                        "params": {"r": float(rng.uniform(40, 70)), "alpha": float(rng.uniform(0.4, 0.8))},
                        "target_strategy": "anchor",
                        "target_location": (edge_x, edge_y),
                        "reason": "Stamp on table boundary",
                    }
                )

        if len(plan) == 0:
            plan.append(
                {
                    "probe_type": str(rng.choice(["P1", "P5", "P6"])),
                    "params": {"w": float(rng.uniform(2, 5)), "l_ratio": float(rng.uniform(0.5, 1.0))},
                    "target_strategy": "anchor",
                    "target_location": None,
                    "reason": "Fallback",
                }
            )
        return plan
