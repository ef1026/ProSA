from __future__ import annotations

import json
import logging
import os
import time

from .base import PolicyBase
from .random_policy import RandomPolicy


LOGGER = logging.getLogger(__name__)


class LLMPolicy(PolicyBase):
    """LLM-based attack policy using DeepSeek API (OpenAI-compatible)."""

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        temperature: float = 0.7,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 3,
        request_timeout: float = 30.0,
        disable_after_consecutive_connection_failures: int = 2,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.request_timeout = float(request_timeout)
        self.disable_after_consecutive_connection_failures = int(
            max(1, disable_after_consecutive_connection_failures)
        )
        self._consecutive_connection_failures = 0
        self._connection_fallback_enabled = False
        self._fallback = RandomPolicy()
        self._init_client(api_key, base_url)

    def _sanitize_api_key(self, key: str | None) -> str | None:
        if key is None:
            return None
        clean = str(key).strip()
        # Common Windows shell mistake: key copied with quotes.
        if (clean.startswith('"') and clean.endswith('"')) or (
            clean.startswith("'") and clean.endswith("'")
        ):
            clean = clean[1:-1].strip()
        return clean or None

    def _normalize_base_url(self, base_url: str) -> str:
        clean = (base_url or "").strip().rstrip("/")
        if not clean:
            clean = "https://api.deepseek.com"
        # DeepSeek OpenAI-compatible API is on /v1.
        if clean == "https://api.deepseek.com":
            return "https://api.deepseek.com/v1"
        return clean

    def _is_connection_error(self, exc: Exception) -> bool:
        name = exc.__class__.__name__.lower()
        msg = str(exc).lower()
        if "connection" in name or "timeout" in name:
            return True
        markers = (
            "connection error",
            "timed out",
            "temporary failure",
            "name or service not known",
            "network is unreachable",
            "max retries exceeded",
            "ssl",
        )
        return any(m in msg for m in markers)

    def _init_client(self, api_key: str | None, base_url: str) -> None:
        key = self._sanitize_api_key(api_key) or self._sanitize_api_key(os.environ.get("DEEPSEEK_API_KEY"))
        norm_base_url = self._normalize_base_url(base_url)
        if not key:
            LOGGER.warning("No API key provided; LLMPolicy will fall back to RandomPolicy on every call")
            self.client = None
            return
        try:
            from openai import OpenAI
            # Disable SDK-level retries here to avoid multiplying retry loops.
            self.client = OpenAI(api_key=key, base_url=norm_base_url, timeout=self.request_timeout, max_retries=0)
            LOGGER.info("LLMPolicy client initialised: model=%s, base_url=%s, timeout=%.1fs", self.model_name, norm_base_url, self.request_timeout)
        except Exception as exc:
            LOGGER.warning("Failed to initialise OpenAI client: %s", exc)
            self.client = None

    @property
    def name(self):
        return f"LLM-{self.model_name}"

    def _build_prompt(self, context):
        system = (
            "You are an expert adversarial tester for Document Layout Analysis (DLA) systems. "
            "A DLA model takes a document image as input and outputs detected layout blocks "
            "with bounding boxes. Your goal: choose ONE adversarial visual probe and place it "
            "on the document image to cause the maximum parsing failure — forcing the model to "
            "merge two separate blocks, split a single block, or miss a block entirely.\n\n"
            "You must reason step-by-step about the spatial arrangement of the blocks before deciding.\n\n"
            "Available probe types (choose exactly ONE):\n"
            "  P1  – Horizontal crease    (params: w in [1,10], l_ratio in [0.5,1.0])\n"
            "  P2  – Vertical crease       (params: w in [1,10], l_ratio in [0.5,1.0])\n"
            "  P3  – Stamp/seal overlay    (params: r in [30,90], alpha in [0.2,1.0])\n"
            "  P4  – Rectangular whitening  (params: area_ratio in [0.03,0.25], beta in [0.2,1.0])\n"
            "  P5  – Fake separator line    (params: w in [1,5], l_ratio in [0.2,0.8])\n"
            "  P6  – Gradient crease        (params: alpha in [0.05,0.4], w in [2,10])\n"
            "  P8  – Noise dot cluster      (params: n_points in [10,100], r in [1,4], sigma in [10,50])\n"
            "  P9  – Irregular stain blob   (params: r_base in [30,80], epsilon in [0.1,0.5], alpha in [0.3,0.7])\n"
            "  P10 – Diagonal crease        (params: theta in [20,70], w in [1,6])\n\n"
            "Placement strategies (choose exactly ONE):\n"
            "  bridge  – Place between two blocks\n"
            "  anchor  – Place on the edge of a block\n"
            "  content – Place inside a block\n"
            "  random  – Place anywhere on the page\n\n"
            "Output strictly valid JSON. Do not output anything outside the JSON object."
        )
        user = (
            f"Here is the document layout:\n{context.to_neutral_description()}\n\n"
            "Examine the blocks and their coordinates carefully. "
            "Think about how the model might interpret spatial relationships between blocks. "
            "Choose the single most effective probe and placement to cause maximum parsing disruption.\n\n"
            "You MUST respond with exactly this JSON schema:\n"
            "{\n"
            '  "chain_of_thought": "Your step-by-step reasoning about block positions, '
            'spatial relationships, and why this probe and placement will maximally disrupt parsing.",\n'
            '  "attack_plan": {\n'
            '    "probe_type": "P5",\n'
            '    "params": {"w": 2, "l_ratio": 0.5},\n'
            '    "target_strategy": "bridge",\n'
            '    "target_location": [x, y]\n'
            "  }\n"
            "}\n\n"
            "Notes:\n"
            "- target_location is an [x, y] pixel coordinate on the image, or null to let the system choose.\n"
            "- params must respect the allowed ranges listed above."
        )
        return system, user

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        if self.client is None:
            raise RuntimeError("No API client configured (missing API key)")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _parse_llm_plan(self, llm_output: str) -> list:
        """Parse the new single-step schema with chain_of_thought."""
        obj = json.loads(llm_output)
        if not isinstance(obj, dict):
            return []

        # ── New schema: {chain_of_thought, attack_plan} ──
        if "attack_plan" in obj:
            plan_obj = obj["attack_plan"]
            if isinstance(plan_obj, dict) and "probe_type" in plan_obj:
                cot = obj.get("chain_of_thought", "")
                plan_obj["chain_of_thought"] = cot
                return [plan_obj]

        # ── Legacy fallback: {plan: [...]} ──
        legacy = obj.get("plan", [])
        if isinstance(legacy, list) and legacy:
            return legacy

        return []

    def select(self, context, rng):
        if self._connection_fallback_enabled:
            plan = self._fallback.select(context, rng)
            for step in plan:
                step["reason"] = "LLM_connection_disabled_fallback"
                step["chain_of_thought"] = ""
            return plan

        system_prompt, user_prompt = self._build_prompt(context)
        connection_failures_this_call = 0
        for attempt in range(self.max_retries):
            try:
                llm_output = self._call_llm(system_prompt, user_prompt)
                LOGGER.debug("LLM raw output: %s", llm_output[:500])
                parsed = self._parse_llm_plan(llm_output)
                if parsed:
                    self._consecutive_connection_failures = 0
                    for step in parsed:
                        step.setdefault("target_location", None)
                        step.setdefault("reason", "LLM")
                        step.setdefault("chain_of_thought", "")
                    return parsed
            except Exception as exc:
                if self._is_connection_error(exc):
                    connection_failures_this_call += 1
                LOGGER.warning("LLMPolicy attempt %d/%d failed: %s", attempt + 1, self.max_retries, exc)
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))

        if connection_failures_this_call >= self.max_retries:
            self._consecutive_connection_failures += 1
            if self._consecutive_connection_failures >= self.disable_after_consecutive_connection_failures:
                self._connection_fallback_enabled = True
                LOGGER.error(
                    "LLMPolicy disabled after %d consecutive connection-failed calls; using RandomPolicy fallback for remaining samples",
                    self._consecutive_connection_failures,
                )
        else:
            self._consecutive_connection_failures = 0

        LOGGER.warning("LLMPolicy all retries exhausted, falling back to RandomPolicy")
        plan = self._fallback.select(context, rng)
        for step in plan:
            step["reason"] = "LLM_fallback_to_random"
            step["chain_of_thought"] = ""
        return plan
