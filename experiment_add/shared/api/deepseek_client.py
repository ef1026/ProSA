from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def get_deepseek_api_key(env_name: str = "DEEPSEEK_API_KEY") -> str | None:
    """Read DeepSeek API key from environment only."""
    value = os.environ.get(env_name)
    return value if value else None


def parse_json_response_text(text: str) -> Any:
    """Parse a strict JSON response, stripping common code fences if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


def chat_completion(
    messages: list[dict[str, str]],
    model: str,
    api_key_env: str = "DEEPSEEK_API_KEY",
    base_url: str = "https://api.deepseek.com",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    top_p: float | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Call DeepSeek chat completion and return structured success/error data."""
    api_key = get_deepseek_api_key(api_key_env)
    if not api_key:
        return {"ok": False, "error": f"{api_key_env} is missing", "content": None, "raw": None}

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        payload["top_p"] = top_p

    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        # ``returned_model`` is what the server actually used / aliased to.
        # It is observation-only and must not influence cache keys; it is
        # surfaced here so callers can log requested vs returned model.
        returned_model = raw.get("model") if isinstance(raw, dict) else None
        return {
            "ok": True,
            "error": None,
            "content": content,
            "raw": raw,
            "requested_model": model,
            "returned_model": returned_model,
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "error": f"HTTP {exc.code}: {body[:500]}",
            "content": None,
            "raw": None,
            "requested_model": model,
            "returned_model": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "content": None,
            "raw": None,
            "requested_model": model,
            "returned_model": None,
        }
