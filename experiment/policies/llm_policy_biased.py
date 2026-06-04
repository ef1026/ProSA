"""LLM policy with adversarial (biased) framing — storyline §3.4 Treatment arm."""
from __future__ import annotations

import logging

from .llm_policy import LLMPolicy

LOGGER = logging.getLogger(__name__)


class LLMPolicyBiased(LLMPolicy):
    """Biased prompt: adversarial system prompt + to_text_description().

    The system prompt explicitly tells the LLM it is an "adversarial tester"
    and asks it to cause "maximum parsing failure".  The user prompt reveals
    structural vulnerability information (gap density, vulnerable points,
    content area ratio, etc.) via ``context.to_text_description()``.
    """

    @property
    def name(self):
        return f"LLM-biased-{self.model_name}"

    def _build_prompt(self, context):
        system = (
            "You are an expert adversarial tester for Document Layout Analysis (DLA) systems. "
            "A DLA model takes a document image as input and outputs detected layout blocks "
            "with bounding boxes. Your goal: choose ONE adversarial visual probe and place it "
            "on the document image to cause the maximum parsing failure — forcing the model to "
            "merge two separate blocks, split a single block, or miss a block entirely.\n\n"
            "You must reason step-by-step about the spatial arrangement of the blocks before deciding.\n\n"
            "Available probe types (choose exactly ONE):\n"
            "  P1  – Horizontal line artifact  (params: w in [1,10], l_ratio in [0.5,1.0])\n"
            "  P2  – Vertical line artifact     (params: w in [1,10], l_ratio in [0.5,1.0])\n"
            "  P3  – Circular overlay           (params: r in [30,90], alpha in [0.2,1.0])\n"
            "  P4  – Rectangular region modification (params: area_ratio in [0.03,0.25], beta in [0.2,1.0])\n"
            "  P5  – Thin horizontal line       (params: w in [1,5], l_ratio in [0.2,0.8])\n"
            "  P6  – Gradient band              (params: alpha in [0.05,0.4], w in [2,10])\n"
            "  P8  – Dot cluster                (params: n_points in [10,100], r in [1,4], sigma in [10,50])\n"
            "  P9  – Irregular patch             (params: r_base in [30,80], epsilon in [0.1,0.5], alpha in [0.3,0.7])\n"
            "  P10 – Diagonal line artifact      (params: theta in [20,70], w in [1,6])\n\n"
            "Placement strategies (choose exactly ONE):\n"
            "  bridge  – Place between two blocks to force merging\n"
            "  anchor  – Place on the edge of a block to blur boundaries\n"
            "  content – Place inside a block to fragment it\n"
            "  random  – Place anywhere on the page\n\n"
            "Output strictly valid JSON. Do not output anything outside the JSON object."
        )
        user = (
            f"Here is the document layout with structural analysis:\n"
            f"{context.to_text_description()}\n\n"
            "Examine the blocks, their coordinates, and the identified structural gaps carefully. "
            "Think about how the DLA model might interpret spatial relationships between blocks. "
            "Target the most vulnerable structural weakness you can identify.\n\n"
            "Choose the single most effective probe and placement to cause maximum parsing disruption.\n\n"
            "You MUST respond with exactly this JSON schema:\n"
            "{\n"
            '  "chain_of_thought": "Your step-by-step reasoning about block positions, '
            'spatial relationships, vulnerable gaps, and why this probe and placement '
            'will maximally disrupt parsing.",\n'
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
