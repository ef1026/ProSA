from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiment_add.shared.api.api_cache import build_cache_key, read_cache, write_cache
from experiment_add.shared.api.api_logging import log_api_event
from experiment_add.shared.api.deepseek_client import chat_completion, parse_json_response_text
from experiment_add.shared.data.load_manifest import load_manifest
from experiment_add.shared.text.normalize_text import normalize_answer
from experiment_add.shared.utils.hash_utils import hash_text
from experiment_add.shared.utils.io import atomic_write_json, ensure_dir, read_jsonl, read_text, read_yaml, write_text
from experiment_add.shared.utils.path_manager import PathManager


REQUIRED_DEEPSEEK_KEYS = {
    "generation": ["model", "temperature", "max_tokens", "candidates_per_page", "batch_by_page", "cache_enabled"],
    "answering": ["model", "temperature", "top_p", "max_tokens", "batch_by_page", "cache_enabled"],
}
REQUIRED_QA_ITEM_KEYS = {"question", "answer", "evidence_text", "answer_type"}


def _validate_deepseek_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if config.get("api_provider") != "deepseek":
        reasons.append("api_provider must be deepseek")
    if config.get("api_key_env") != "DEEPSEEK_API_KEY":
        reasons.append("api_key_env must be DEEPSEEK_API_KEY")
    model_policy = config.get("model_policy", {})
    if model_policy.get("experiment_model_family") != "deepseek-v3":
        reasons.append("model_policy.experiment_model_family must be deepseek-v3")
    allowed_models = set(model_policy.get("allowed_api_models", []))
    if allowed_models != {"deepseek-chat"}:
        reasons.append("model_policy.allowed_api_models must contain only deepseek-chat for DeepSeek-V3")
    if model_policy.get("forbid_model_switching") is not True:
        reasons.append("model_policy.forbid_model_switching must be true")
    for section, keys in REQUIRED_DEEPSEEK_KEYS.items():
        value = config.get(section)
        if not isinstance(value, dict):
            reasons.append(f"missing section {section}")
            continue
        if value.get("model_family") != "deepseek-v3":
            reasons.append(f"{section}.model_family must be deepseek-v3")
        if value.get("model") not in allowed_models:
            reasons.append(f"{section}.model must be one of {sorted(allowed_models)}")
        for key in keys:
            if key not in value:
                reasons.append(f"missing {section}.{key}")
    return not reasons, reasons


def _clean_outputs_ready(pm: PathManager) -> tuple[bool, int, list[str]]:
    reasons: list[str] = []
    manifest = load_manifest(pm.page_manifest_debug20)
    manifest_ids = {row["page_id"] for row in manifest}
    aligned_ids: set[str] | None = None
    for pipeline in ("mineru", "ppstructure"):
        path = pm.clean_parse_merged_path(pipeline)
        if not path.exists() or path.stat().st_size == 0:
            reasons.append(f"{pipeline} merged clean parse missing or empty")
            continue
        rows = read_jsonl(path)
        success = [row for row in rows if row.get("parser_status") == "success"]
        non_empty = [row for row in success if str(row.get("page_text", "")).strip()]
        if len(rows) != len(manifest):
            reasons.append(f"{pipeline} row count {len(rows)} != debug manifest {len(manifest)}")
        if len(success) != len(manifest):
            reasons.append(f"{pipeline} success count {len(success)} != debug manifest {len(manifest)}")
        if len(non_empty) != len(manifest):
            reasons.append(f"{pipeline} non-empty page_text count {len(non_empty)} != debug manifest {len(manifest)}")
        ids = {row.get("page_id") for row in success}
        if not ids.issubset(manifest_ids):
            reasons.append(f"{pipeline} contains page_ids outside debug manifest")
        aligned_ids = ids if aligned_ids is None else aligned_ids & ids
    return not reasons, len(aligned_ids or set()), reasons


def _prompt_ready(prompt_path: Path) -> tuple[bool, list[str]]:
    if not prompt_path.exists():
        return False, ["qa_generation_prompt.txt missing"]
    text = read_text(prompt_path)
    required_phrases = [
        "Generate extractive question-answer pairs only from the provided text.",
        "Each answer must be a contiguous short span copied from the text.",
        "Do not use external knowledge.",
        "Do not generate yes/no questions.",
        "Return a strict JSON list.",
        '"question"',
        '"answer"',
        '"evidence_text"',
        '"answer_type"',
    ]
    missing = [phrase for phrase in required_phrases if phrase not in text]
    return not missing, [f"prompt missing phrase: {phrase}" for phrase in missing]


def _build_smoke_messages(prompt: str, page_text: str, candidates_per_page: int) -> list[dict[str, str]]:
    user = (
        f"Generate at most {candidates_per_page} QA candidates from this document text.\n\n"
        f"DOCUMENT TEXT:\n{page_text[:6000]}"
    )
    return [{"role": "system", "content": prompt}, {"role": "user", "content": user}]


def _validate_qa_items(items: Any, source_text: str) -> tuple[bool, list[str]]:
    if not isinstance(items, list):
        return False, ["response is not a JSON list"]
    reasons = []
    norm_source = normalize_answer(source_text)
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            reasons.append(f"item {idx} is not object")
            continue
        missing = REQUIRED_QA_ITEM_KEYS - set(item)
        if missing:
            reasons.append(f"item {idx} missing keys {sorted(missing)}")
        answer = str(item.get("answer", ""))
        evidence_text = str(item.get("evidence_text", ""))
        if normalize_answer(answer) not in norm_source:
            reasons.append(f"item {idx} answer not found in source text")
        if normalize_answer(answer) not in normalize_answer(evidence_text):
            reasons.append(f"item {idx} answer not found in evidence_text")
    return not reasons, reasons


def _run_smoke_test(pm: PathManager, deepseek_cfg: dict[str, Any], prompt_path: Path, allow_external_api: bool) -> tuple[str, list[str]]:
    if not allow_external_api:
        return "SKIPPED", ["external API call not explicitly allowed"]
    mineru_rows = [row for row in read_jsonl(pm.clean_parse_merged_path("mineru")) if row.get("parser_status") == "success" and row.get("page_text")]
    if not mineru_rows:
        return "FAIL", ["no successful MinerU page_text available for smoke test"]
    source = mineru_rows[0]
    prompt = read_text(prompt_path)
    gen_cfg = deepseek_cfg["generation"]
    model = gen_cfg["model"]
    temperature = float(gen_cfg["temperature"])
    prompt_hash = hash_text(prompt)
    input_hash = hash_text(source["page_text"][:6000])
    cache_dir = pm.qa_pairs_dir / "api_cache"
    ensure_dir(cache_dir)
    cache_key = build_cache_key(model, temperature, prompt_hash, input_hash, "qa_generation_smoke_test")
    cached = read_cache(cache_dir, cache_key)
    api_log_path = pm.shared_log_root / "qa_generation_api_log.jsonl"
    if cached is not None:
        result = cached
        log_api_event(api_log_path, {"event": "cache_hit", "task": "qa_generation_smoke_test", "model": model, "cache_key": cache_key})
    else:
        messages = _build_smoke_messages(prompt, source["page_text"], int(gen_cfg["candidates_per_page"]))
        result = chat_completion(
            messages,
            model=model,
            api_key_env=deepseek_cfg["api_key_env"],
            base_url=deepseek_cfg.get("client", {}).get("base_url", "https://api.deepseek.com"),
            temperature=temperature,
            max_tokens=int(gen_cfg["max_tokens"]),
            timeout_seconds=int(deepseek_cfg.get("client", {}).get("timeout_seconds", 60)),
        )
        log_api_event(api_log_path, {"event": "api_call", "task": "qa_generation_smoke_test", "model": model, "ok": result.get("ok"), "cache_key": cache_key})
        if result.get("ok"):
            write_cache(cache_dir, cache_key, result)
    if not result.get("ok"):
        return "FAIL", [str(result.get("error"))]
    try:
        parsed = parse_json_response_text(str(result.get("content") or ""))
    except Exception as exc:
        return "FAIL", [f"response JSON parse failed: {exc}"]
    valid, reasons = _validate_qa_items(parsed, source["page_text"])
    atomic_write_json(
        pm.qa_pairs_dir / "qa_generation_smoke_test.json",
        {"page_id": source.get("page_id"), "items": parsed, "valid": valid, "reasons": reasons},
    )
    return ("OK" if valid else "FAIL"), reasons


def run_preflight(config_path: str | Path, deepseek_config_path: str | Path, smoke_test: bool = False, allow_external_api: bool = False) -> dict[str, Any]:
    pm = PathManager(config_path, create_dirs=True)
    deepseek_cfg = read_yaml(deepseek_config_path)
    prompt_path = pm.project_root / "experiment_add/prompts/qa_generation_prompt.txt"
    report_path = pm.shared_log_root / "qa_generation_preflight_report.md"
    api_key_found = bool(os.environ.get("DEEPSEEK_API_KEY"))
    deepseek_valid, deepseek_reasons = _validate_deepseek_config(deepseek_cfg)
    clean_ready, aligned_pages, clean_reasons = _clean_outputs_ready(pm)
    prompt_ok, prompt_reasons = _prompt_ready(prompt_path)
    ensure_dir(pm.qa_pairs_dir)
    ensure_dir(pm.shared_log_root)
    cache_ready = (pm.qa_pairs_dir / "api_cache").exists() or bool(ensure_dir(pm.qa_pairs_dir / "api_cache"))
    api_log_ready = pm.shared_log_root.exists()

    smoke_status = "SKIPPED"
    smoke_reasons = []
    if smoke_test:
        if not api_key_found:
            smoke_reasons.append("DEEPSEEK_API_KEY missing")
            smoke_status = "SKIPPED"
        else:
            smoke_status, smoke_reasons = _run_smoke_test(pm, deepseek_cfg, prompt_path, allow_external_api)

    smoke_ok = (not smoke_test) or smoke_status == "OK"
    ready = api_key_found and deepseek_valid and clean_ready and prompt_ok and cache_ready and api_log_ready and smoke_ok
    reasons = []
    if not api_key_found:
        reasons.append("DEEPSEEK_API_KEY is missing in this process environment")
    reasons.extend(deepseek_reasons + clean_reasons + prompt_reasons + smoke_reasons)
    enable_external_api = bool(deepseek_cfg.get("runtime", {}).get("enable_external_api", False))

    lines = [
        "# QA Generation Preflight Report",
        "",
        f"- DEEPSEEK_API_KEY: `{'FOUND' if api_key_found else 'MISSING'}`",
        f"- deepseek.yaml valid: `{'YES' if deepseek_valid else 'NO'}`",
        f"- clean parser outputs ready: `{'YES' if clean_ready else 'NO'}`",
        f"- debug20 aligned pages: `{aligned_pages}`",
        f"- qa_generation_prompt ready: `{'YES' if prompt_ok else 'NO'}`",
        f"- api cache ready: `{'YES' if cache_ready else 'NO'}`",
        f"- api logging ready: `{'YES' if api_log_ready else 'NO'}`",
        f"- smoke_test_run: `{'YES' if smoke_test and smoke_status != 'SKIPPED' else 'NO'}`",
        f"- smoke_test_status: `{smoke_status}`",
        f"- ready_for_debug20_qa_generation: `{'YES' if ready else 'NO'}`",
        "",
        "## API Call Setting",
        "",
        f"- deepseek.yaml runtime.enable_external_api: `{enable_external_api}`",
        "- If this is `false`, a later generation command should require explicit user confirmation or an explicit CLI override before calling the API.",
        "",
        "## Reasons / Notes",
        "",
    ]
    lines.extend(f"- {reason}" for reason in reasons) if reasons else lines.append("- No blocking issues found.")
    if smoke_test and smoke_status != "OK":
        lines.append("- Smoke test was requested and did not pass, so debug20 QA generation is not ready yet.")
    if ready:
        lines.extend(
            [
                "",
                "## Next Debug20 QA Generation Command",
                "",
                "Do not run automatically in preflight. Suggested next command:",
                "",
                "```bash",
                "python3 experiment_add/shared/qa_generation/run_qa_generation.py \\",
                "  --config experiment_add/configs/base.yaml \\",
                "  --deepseek-config experiment_add/configs/deepseek.yaml \\",
                "  --debug",
                "```",
            ]
        )
    write_text(report_path, "\n".join(lines) + "\n")
    return {
        "report_path": str(report_path),
        "api_key_found": api_key_found,
        "deepseek_yaml_valid": deepseek_valid,
        "clean_outputs_ready": clean_ready,
        "aligned_pages": aligned_pages,
        "prompt_ready": prompt_ok,
        "cache_ready": cache_ready,
        "api_logging_ready": api_log_ready,
        "smoke_test_status": smoke_status,
        "ready_for_debug20_qa_generation": ready,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DeepSeek / QA generation preflight.")
    parser.add_argument("--config", default="experiment_add/configs/base.yaml")
    parser.add_argument("--deepseek-config", default="experiment_add/configs/deepseek.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-external-api", action="store_true")
    args = parser.parse_args()
    result = run_preflight(args.config, args.deepseek_config, args.smoke_test, args.allow_external_api)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ready_for_debug20_qa_generation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
