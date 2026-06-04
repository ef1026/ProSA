from __future__ import annotations

import argparse
import json
import os
import re
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
from experiment_add.shared.qa_generation.locate_evidence_blocks import locate_evidence_blocks
from experiment_add.shared.text.answer_matching import count_answer_occurrences
from experiment_add.shared.text.normalize_text import normalize_answer
from experiment_add.shared.utils.hash_utils import hash_json, hash_text
from experiment_add.shared.utils.io import atomic_write_jsonl, ensure_dir, read_jsonl, read_text, read_yaml, write_text
from experiment_add.shared.utils.path_manager import PathManager
from experiment_add.shared.utils.retry import retry_call


def _load_clean_outputs(pm: PathManager) -> dict[str, dict[str, dict[str, Any]]]:
    """Load clean parse outputs keyed by pipeline then page_id."""
    return {
        pipeline: {row["page_id"]: row for row in read_jsonl(pm.clean_parse_merged_path(pipeline))}
        for pipeline in ("mineru", "ppstructure")
    }


def _bad_char_ratio(text: str) -> float:
    """Estimate garbled text ratio conservatively."""
    if not text:
        return 1.0
    bad = sum(1 for ch in text if ch == "\ufffd" or (ord(ch) < 32 and ch not in "\n\t\r"))
    return bad / len(text)


def _choose_source(mineru: dict[str, Any], ppstructure: dict[str, Any]) -> tuple[str, str]:
    """Choose generation source text; prefer longer/cleaner, tie to MinerU."""
    candidates = {
        "mineru": str(mineru.get("page_text", "")),
        "ppstructure": str(ppstructure.get("page_text", "")),
    }
    lengths = {k: len(v) for k, v in candidates.items()}
    bad = {k: _bad_char_ratio(v) for k, v in candidates.items()}
    if bad["mineru"] <= bad["ppstructure"] + 0.01 and lengths["mineru"] >= lengths["ppstructure"] * 0.9:
        return "mineru", candidates["mineru"]
    if lengths["ppstructure"] > lengths["mineru"] * 1.1 or bad["ppstructure"] + 0.01 < bad["mineru"]:
        return "ppstructure", candidates["ppstructure"]
    return "mineru", candidates["mineru"]


def _split_lines(text: str) -> list[str]:
    """Return short-ish non-empty text lines for shared-friendly candidate source."""
    lines = []
    for raw in re.split(r"[\n\r]+|(?<=[.!?])\s+", text):
        line = " ".join(raw.split()).strip()
        if 8 <= len(line) <= 220:
            lines.append(line)
    return lines


def _shared_friendly_source(mineru: dict[str, Any], ppstructure: dict[str, Any]) -> tuple[str, str, str]:
    """Build a simple source from text snippets visible in both clean parsers."""
    mineru_text = str(mineru.get("page_text", ""))
    pp_text = str(ppstructure.get("page_text", ""))
    norm_pp = normalize_answer(pp_text)
    selected: list[str] = []
    for line in _split_lines(mineru_text):
        norm_line = normalize_answer(line)
        token_len = len(norm_line.split())
        if not norm_line or token_len > 35:
            continue
        if norm_line in norm_pp:
            selected.append(line)
        if len(selected) >= 60:
            break
    source_text = "\n".join(selected)
    if len(source_text) >= 800:
        return "shared", source_text, "shared_friendly"
    pipeline, fallback = _choose_source(mineru, ppstructure)
    return pipeline, fallback, "full_text_fallback"


def _messages(prompt: str, page_text: str, n: int, fallback: bool = False) -> list[dict[str, str]]:
    fallback_note = ""
    if fallback:
        fallback_note = (
            "\nFallback mode: Generate only questions with short exact-span answers. "
            "Prefer numbers, titles, dataset names, method names, and short table values. "
            "Avoid long answers and answers that may appear differently across OCR outputs.\n"
        )
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Generate exactly up to {n} QA candidates from this document text.{fallback_note}\n\nDOCUMENT TEXT:\n{page_text[:8000]}"},
    ]


def _call_or_cache(
    page_id: str,
    page_text: str,
    prompt: str,
    cfg: dict[str, Any],
    pm: PathManager,
    candidates_per_page: int,
    prompt_version: str,
    task: str = "qa_generation",
    fallback: bool = False,
    source_mode: str = "full_text_fallback",
) -> tuple[str, list[dict[str, Any]], str | None, str, str | None]:
    """Return ``(api_status, candidates, error_message, source, returned_model)``.

    The returned ``returned_model`` is observation-only and is intended to be
    written into per-page metadata and API logs; it is *not* part of the
    cache key (which stays keyed on the requested model so that repeat runs
    with the same configuration are reproducible regardless of provider-side
    routing or aliasing).
    """
    gen_cfg = cfg["generation"]
    model = gen_cfg["model"]
    temperature = float(gen_cfg["temperature"])
    prompt_hash = hash_text(prompt)
    input_hash = hash_json(
        {
            "prompt_version": prompt_version,
            "candidates_per_page": candidates_per_page,
            "page_text_hash": hash_text(page_text[:8000]),
            "source_mode": source_mode,
            "fallback": fallback,
        }
    )
    cache_key = build_cache_key(model, temperature, prompt_hash, input_hash, task)
    cache_dir = pm.qa_pairs_dir / "api_cache"
    ensure_dir(cache_dir)
    api_log_path = pm.shared_log_root / "qa_generation_api_log.jsonl"
    cached = read_cache(cache_dir, cache_key) if gen_cfg.get("cache_enabled", True) else None
    returned_model: str | None
    if cached is not None:
        # Older cache payloads predate ``returned_model`` and yield None.
        returned_model = cached.get("returned_model") if isinstance(cached, dict) else None
        log_api_event(
            api_log_path,
            {
                "event": "cache_hit",
                "task": task,
                "model": model,
                "requested_model": model,
                "returned_model": returned_model,
                "page_id": page_id,
                "cache_key": cache_key,
                "prompt_version": prompt_version,
            },
        )
        content = cached.get("content", "")
        source = "cache"
    else:
        attempt_state: dict[str, Any] = {"returned_model": None}

        def _attempt() -> str:
            result = chat_completion(
                _messages(prompt, page_text, candidates_per_page, fallback=fallback),
                model=model,
                api_key_env=cfg["api_key_env"],
                base_url=cfg.get("client", {}).get("base_url", "https://api.deepseek.com"),
                temperature=temperature,
                max_tokens=int(gen_cfg["max_tokens"]),
                timeout_seconds=int(cfg.get("client", {}).get("timeout_seconds", 60)),
            )
            attempt_returned_model = result.get("returned_model")
            attempt_state["returned_model"] = attempt_returned_model
            log_api_event(
                api_log_path,
                {
                    "event": "api_call",
                    "task": task,
                    "model": model,
                    "requested_model": result.get("requested_model", model),
                    "returned_model": attempt_returned_model,
                    "ok": result.get("ok"),
                    "page_id": page_id,
                    "cache_key": cache_key,
                    "prompt_version": prompt_version,
                },
            )
            if not result.get("ok"):
                raise RuntimeError(result.get("error") or "DeepSeek API failed")
            parsed = parse_json_response_text(str(result.get("content") or ""))
            if not isinstance(parsed, list):
                raise ValueError("DeepSeek response is not a JSON list")
            write_cache(
                cache_dir,
                cache_key,
                {
                    "content": result.get("content"),
                    "model": model,
                    "requested_model": model,
                    "returned_model": attempt_returned_model,
                },
            )
            return str(result.get("content") or "")

        content = retry_call(_attempt, max_retries=2, sleep_seconds=1.0, backoff=True)
        returned_model = attempt_state.get("returned_model")
        source = "api"
    parsed = parse_json_response_text(content)
    if not isinstance(parsed, list):
        raise ValueError("DeepSeek response is not a JSON list")
    return (
        "success",
        [item for item in parsed if isinstance(item, dict)],
        None,
        source,
        returned_model,
    )


def _candidate_reasons(candidate: dict[str, Any], source_text: str, mineru_text: str, pp_text: str) -> list[str]:
    """Mirror the strict filter enough to find zero-QA pages before fallback."""
    reasons = []
    q = str(candidate.get("question", "")).strip()
    a = str(candidate.get("answer", "")).strip()
    if not q or not a:
        reasons.append("empty_question_or_answer")
    n = len(normalize_answer(a).split())
    if n < 2 or n > 15:
        reasons.append("answer_length_out_of_range")
    if normalize_answer(a) not in normalize_answer(source_text):
        reasons.append("answer_not_in_source_text")
    if normalize_answer(a) not in normalize_answer(mineru_text):
        reasons.append("answer_not_in_mineru_text")
    if normalize_answer(a) not in normalize_answer(pp_text):
        reasons.append("answer_not_in_ppstructure_text")
    if count_answer_occurrences(mineru_text, a) > 3 or count_answer_occurrences(pp_text, a) > 3:
        reasons.append("answer_too_frequent")
    evidence = locate_evidence_blocks(a, str(candidate.get("evidence_text", "")), {"page_text": mineru_text, "blocks": []}, {"page_text": pp_text, "blocks": []})
    # This locate call has no blocks in preflight; do not enforce block presence here.
    return reasons


def _estimated_kept_count(row: dict[str, Any], mineru: dict[str, Any], pp: dict[str, Any]) -> int:
    mineru_text = str(mineru.get("page_text", ""))
    pp_text = str(pp.get("page_text", ""))
    source_text = str(row.get("generation_source_text", ""))
    kept_answers: set[str] = set()
    for candidate in row.get("candidates", []):
        if _candidate_reasons(candidate, source_text, mineru_text, pp_text):
            continue
        kept_answers.add(normalize_answer(candidate.get("answer", "")))
    return min(4, len(kept_answers))


def generate_candidates(
    deepseek_config_path: str | Path,
    debug: bool = False,
    prompt_version: str | None = None,
    candidates_per_page: int | None = None,
    output_suffix: str | None = None,
) -> dict[str, Any]:
    """Generate raw QA candidates for debug20 from clean parser outputs only.

    When ``output_suffix`` is provided (e.g. ``"v2_debug"``), all QA pair
    artifacts are written to ``experiment_add/outputs/shared/qa_pairs_<suffix>/``
    instead of the canonical ``qa_pairs/`` directory, so debug/v2 runs do
    not overwrite canonical artifacts consumed by downstream stages.
    """
    cfg_path = Path(deepseek_config_path)
    base_config = cfg_path.parent / "base.yaml"
    pm = PathManager(base_config, create_dirs=True, qa_pairs_suffix=output_suffix)
    cfg = read_yaml(cfg_path)
    if not os.environ.get(cfg.get("api_key_env", "DEEPSEEK_API_KEY")):
        raise RuntimeError("DEEPSEEK_API_KEY is missing; refusing to call API")
    prompt = read_text(pm.project_root / "experiment_add/prompts/qa_generation_prompt.txt")
    gen_cfg = cfg["generation"]
    prompt_version = prompt_version or str(gen_cfg.get("prompt_version", "v1"))
    candidates_per_page = int(candidates_per_page or gen_cfg.get("candidates_per_page", 10))
    fallback_n = int(gen_cfg.get("fallback_candidates_per_zero_qa_page", 0))
    manifest = load_manifest(pm.page_manifest_debug20 if debug else pm.page_manifest_500)
    clean = _load_clean_outputs(pm)
    rows: list[dict[str, Any]] = []
    api_calls = 0
    cache_hits = 0
    failures = 0
    for record in manifest:
        page_id = record["page_id"]
        mineru = clean["mineru"].get(page_id)
        pp = clean["ppstructure"].get(page_id)
        if not mineru or not pp or mineru.get("parser_status") != "success" or pp.get("parser_status") != "success" or not mineru.get("page_text") or not pp.get("page_text"):
            rows.append({"page_id": page_id, "generation_source_pipeline": None, "generation_source_text_hash": None, "candidates": [], "api_status": "skipped", "error_message": "missing successful clean outputs"})
            continue
        source_pipeline, source_text, source_mode = _shared_friendly_source(mineru, pp)
        returned_model: str | None = None
        try:
            status, candidates, error, call_source, returned_model = _call_or_cache(
                page_id,
                source_text,
                prompt,
                cfg,
                pm,
                candidates_per_page=candidates_per_page,
                prompt_version=prompt_version,
                task="qa_generation",
                fallback=False,
                source_mode=source_mode,
            )
            api_calls += 1 if call_source == "api" else 0
            cache_hits += 1 if call_source == "cache" else 0
        except Exception as exc:
            status, candidates, error = "failed", [], str(exc)
            failures += 1
            log_api_event(pm.shared_log_root / "qa_generation_api_log.jsonl", {"event": "page_failed", "task": "qa_generation", "page_id": page_id, "error": error, "requested_model": cfg["generation"]["model"], "returned_model": None})
        rows.append({
            "page_id": page_id,
            "generation_source_pipeline": source_pipeline,
            "generation_source_mode": source_mode,
            "generation_source_text_hash": hash_text(source_text),
            "generation_source_text": source_text,
            "candidates": candidates,
            "api_status": status,
            "error_message": error,
            "requested_model": cfg["generation"]["model"],
            "returned_model": returned_model,
            "prompt_version": prompt_version,
        })
    fallback_pages = []
    fallback_added = 0
    if fallback_n > 0:
        first_pass = list(rows)
        for row in first_pass:
            page_id = row.get("page_id")
            mineru = clean["mineru"].get(page_id, {})
            pp = clean["ppstructure"].get(page_id, {})
            if not mineru or not pp:
                continue
            if _estimated_kept_count(row, mineru, pp) > 0:
                continue
            source_pipeline, source_text, source_mode = _shared_friendly_source(mineru, pp)
            returned_model_fb: str | None = None
            try:
                status, candidates, error, call_source, returned_model_fb = _call_or_cache(
                    str(page_id),
                    source_text,
                    prompt,
                    cfg,
                    pm,
                    candidates_per_page=fallback_n,
                    prompt_version=prompt_version,
                    task="qa_generation_fallback_zero_qa",
                    fallback=True,
                    source_mode=source_mode,
                )
                api_calls += 1 if call_source == "api" else 0
                cache_hits += 1 if call_source == "cache" else 0
                fallback_added += len(candidates)
            except Exception as exc:
                status, candidates, error = "failed", [], str(exc)
                failures += 1
                log_api_event(pm.shared_log_root / "qa_generation_api_log.jsonl", {"event": "page_failed", "task": "qa_generation_fallback_zero_qa", "page_id": page_id, "error": error, "requested_model": cfg["generation"]["model"], "returned_model": None})
            fallback_pages.append(page_id)
            rows.append({
                "page_id": page_id,
                "generation_source_pipeline": source_pipeline,
                "generation_source_mode": f"{source_mode}_fallback_zero_qa",
                "generation_source_text_hash": hash_text(source_text),
                "generation_source_text": source_text,
                "candidates": candidates,
                "api_status": status,
                "error_message": error,
                "fallback_zero_qa": True,
                "requested_model": cfg["generation"]["model"],
                "returned_model": returned_model_fb,
                "prompt_version": prompt_version,
            })
    atomic_write_jsonl(pm.qa_candidates_raw_path, rows)
    total_candidates = sum(len(row.get("candidates", [])) for row in rows)
    summary_path = pm.shared_log_root / ("qa_generation_debug_summary.md" if debug else "qa_generation_summary.md")
    write_text(
        summary_path,
        "\n".join([
            "# QA Generation Summary",
            "",
            f"- mode: `{'debug' if debug else 'full'}`",
            f"- pages: `{len(rows)}`",
            f"- api_calls: `{api_calls}`",
            f"- cache_hits: `{cache_hits}`",
            f"- failed_pages: `{failures}`",
            f"- raw_candidates: `{total_candidates}`",
            f"- prompt_version: `{prompt_version}`",
            f"- candidates_per_page: `{candidates_per_page}`",
            f"- fallback_zero_qa_enabled: `{fallback_n > 0}`",
            f"- fallback_pages_count: `{len(fallback_pages)}`",
            f"- fallback_added_candidates: `{fallback_added}`",
            f"- output: `{pm.qa_candidates_raw_path}`",
        ]) + "\n",
    )
    return {"pages": len(rows), "api_calls": api_calls, "cache_hits": cache_hits, "failed_pages": failures, "raw_candidates": total_candidates, "prompt_version": prompt_version, "candidates_per_page": candidates_per_page, "fallback_pages_count": len(fallback_pages), "fallback_added_candidates": fallback_added, "output": str(pm.qa_candidates_raw_path), "summary": str(summary_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate raw QA candidates from clean parser outputs.")
    parser.add_argument("--config", default="experiment_add/configs/deepseek.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--candidates-per-page", type=int, default=None)
    parser.add_argument(
        "--output-suffix",
        default=None,
        help=(
            "If set, write QA pair artifacts to "
            "experiment_add/outputs/shared/qa_pairs_<suffix>/ instead of "
            "the canonical qa_pairs/ directory. Use this for v2/debug "
            "runs that must not overwrite canonical artifacts."
        ),
    )
    args = parser.parse_args()
    result = generate_candidates(
        args.config,
        debug=args.debug,
        prompt_version=args.prompt_version,
        candidates_per_page=args.candidates_per_page,
        output_suffix=args.output_suffix,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
