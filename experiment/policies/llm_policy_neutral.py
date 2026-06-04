"""LLM policy with neutral (unbiased) framing — storyline §3.4 Control arm."""
from __future__ import annotations

import logging

from .llm_policy import LLMPolicy

LOGGER = logging.getLogger(__name__)


class LLMPolicyNeutral(LLMPolicy):
    """Neutral prompt: no adversarial framing + to_neutral_description().

    The system prompt describes the LLM as a "quality assurance analyst"
    testing document robustness — no mention of "attack", "adversarial",
    "failure", or "disruption".  The user prompt provides only block-type
    + bbox coordinates via ``context.to_neutral_description()``.

    Strategy names are neutralised (presented → internal):
      between → bridge, edge → anchor, inside → content, anywhere → random.
    Probe names are unified neutral across both prompts to isolate
    framing and context as the only controlled variables.
    """

    @property
    def name(self):
        return f"LLM-neutral-{self.model_name}"

    def _build_prompt(self, context):
        system = (
            "You are a quality assurance analyst evaluating the robustness of "
            "Document Layout Analysis (DLA) systems. A DLA model takes a document "
            "image as input and outputs detected layout blocks with bounding boxes. "
            "Your task: select ONE visual perturbation probe and a placement location "
            "to test how the model responds to realistic document degradation.\n\n"
            "You must reason step-by-step about the spatial arrangement of the blocks "
            "before deciding.\n\n"
            "Available perturbation types (choose exactly ONE):\n"
            "  P1  – Horizontal line artifact  (params: w in [1,10], l_ratio in [0.5,1.0])\n"
            "  P2  – Vertical line artifact     (params: w in [1,10], l_ratio in [0.5,1.0])\n"
            "  P3  – Circular overlay           (params: r in [30,90], alpha in [0.2,1.0])\n"
            "  P4  – Rectangular region modification (params: area_ratio in [0.03,0.25], beta in [0.2,1.0])\n"
            "  P5  – Thin horizontal line       (params: w in [1,5], l_ratio in [0.2,0.8])\n"
            "  P6  – Gradient band              (params: alpha in [0.05,0.4], w in [2,10])\n"
            "  P8  – Dot cluster                (params: n_points in [10,100], r in [1,4], sigma in [10,50])\n"
            "  P9  – Irregular patch             (params: r_base in [30,80], epsilon in [0.1,0.5], alpha in [0.3,0.7])\n"
            "  P10 – Diagonal line artifact      (params: theta in [20,70], w in [1,6])\n\n"
            "Placement options (choose exactly ONE):\n"
            "  between  – Between two regions\n"
            "  edge     – Near a region edge\n"
            "  inside   – Inside a region\n"
            "  anywhere – Anywhere on the page\n\n"
            "Output strictly valid JSON. Do not output anything outside the JSON object."
        )
        user = (
            f"Here is the document layout:\n{context.to_neutral_description()}\n\n"
            "Examine the blocks and their coordinates. "
            "Select a probe and placement that would represent a realistic "
            "document degradation scenario.\n\n"
            "You MUST respond with exactly this JSON schema:\n"
            "{\n"
            '  "chain_of_thought": "Your step-by-step reasoning about the block layout '
            'and why this probe and placement represents a good test case.",\n'
            '  "attack_plan": {\n'
            '    "probe_type": "<Pn>",\n'
            '    "params": {"<key>": "<value>", ...},\n'
            '    "target_strategy": "<strategy>",\n'
            '    "target_location": [x, y]\n'
            "  }\n"
            "}\n\n"
            "Notes:\n"
            "- target_location is an [x, y] pixel coordinate on the image, or null to let the system choose.\n"
            "- params must respect the allowed ranges listed above."
        )
        return system, user
