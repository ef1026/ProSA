from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
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
from experiment_add.shared.utils.hash_utils import hash_json, hash_text
from experiment_add.shared.utils.io import atomic_write_jsonl, ensure_dir, read_jsonl, read_text, read_yaml, write_text
from experiment_add.shared.utils.path_manager import PathManager
from experiment_add.shared.utils.retry import retry_call


PIPELINES = ("mineru", "ppstructure")
CONDITIONS = ("clean", "area_matched_erasure", "structural_probe", "large_area_erasure")
DEFAULT_PROMPT_VERSION = "v1"
TASK = "qa_answering"


def _resolve_config_path(path: str | Path, project_root: Path, current_config: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidates = [
        project_root / p,
        project_root / "experiment_add" / p,
        current_config.parent / p.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_parse_outputs(pm: PathManager, pipeline: str, condition: str) -> dict[str, dict[str, Any]]:
    if condition == "clean":
        path = pm.clean_parse_merged_path(pipeline)
    else:
        path = pm.perturbed_parse_dir(pipeline, condition) / "merged.jsonl"
    return {str(row.get("page_id", "")): row for row in read_jsonl(path)}


def _group_qa_by_page(qa_rows: list[dict[str, Any]], manifest_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qa_rows:
        page_id = str(row.get("page_id", ""))
        if page_id in manifest_ids:
            grouped[page_id].append(row)
    return {page_id: rows[:4] for page_id, rows in grouped.items() if rows}


def _question_payload(qa_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "qa_id": str(row.get("qa_id", "")),
            "question": str(row.get("question", "")),
        }
        for row in qa_rows
    ]


def _messages(prompt: str, context: str, questions: list[dict[str, str]]) -> list[dict[str, str]]:
    user_payload = {
        "context": context,
        "questions": questions,
    }
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def _answer_item(qa_row: dict[str, Any], pred_answer: str, status: str, supporting_quote: str) -> dict[str, Any]:
    return {
        "qa_id": str(qa_row.get("qa_id", "")),
        "question": str(qa_row.get("question", "")),
        "pred_answer": pred_answer,
        "status": status,
        "supporting_quote": supporting_quote,
    }


def _skipped_batch(
    page_id: str,
    pipeline: str,
    condition: str,
    qa_rows: list[dict[str, Any]],
    parser_status: str,
    api_status: str,
    answer_status: str,
    error_message: str,
    context_truncated: bool = False,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict[str, Any]:
    return {
        "page_id": page_id,
        "pipeline": pipeline,
        "condition": condition,
        "model": None,
        "prompt_version": prompt_version,
        "context_truncated": context_truncated,
        "parser_status": parser_status,
        "api_status": api_status,
        "error_message": error_message,
        "answers": [_answer_item(row, "", answer_status, "") for row in qa_rows],
    }


def _failed_batch(
    page_id: str,
    pipeline: str,
    condition: str,
    qa_rows: list[dict[str, Any]],
    model: str,
    parser_status: str,
    api_status: str,
    error_message: str,
    context_truncated: bool = False,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict[str, Any]:
    return {
        "page_id": page_id,
        "pipeline": pipeline,
        "condition": condition,
        "model": model,
        "prompt_version": prompt_version,
        "context_truncated": context_truncated,
        "parser_status": parser_status,
        "api_status": api_status,
        "error_message": error_message,
        "answers": [_answer_item(row, "", "api_failed", "") for row in qa_rows],
    }


def _normalize_model_answers(qa_rows: list[dict[str, Any]], parsed: Any) -> list[dict[str, Any]]:
    if not isinstance(parsed, list):
        raise ValueError("DeepSeek response is not a JSON list")
    by_id: dict[str, dict[str, Any]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        qa_id = str(item.get("qa_id", ""))
        if qa_id:
            by_id[qa_id] = item

    answers: list[dict[str, Any]] = []
    for row in qa_rows:
        qa_id = str(row.get("qa_id", ""))
        item = by_id.get(qa_id)
        if item is None:
            answers.append(_answer_item(row, "NOT_FOUND", "not_found", ""))
            continue
        raw_status = str(item.get("status", "")).strip().lower()
        status = raw_status if raw_status in {"answered", "not_found"} else "not_found"
        answer = str(item.get("answer", "")).strip()
        quote = str(item.get("supporting_quote", "")).strip()
        if status == "not_found":
            answer = "NOT_FOUND"
            quote = ""
        if status == "answered" and not answer:
            status = "not_found"
            answer = "NOT_FOUND"
            quote = ""
        answers.append(_answer_item(row, answer, status, quote))
    return answers


def _call_or_cache(
    page_id: str,
    pipeline: str,
    condition: str,
    context: str,
    qa_rows: list[dict[str, Any]],
    prompt: str,
    cfg: dict[str, Any],
    pm: PathManager,
    api_log_path: Path,
    prompt_version: str,
) -> tuple[str, list[dict[str, Any]], str | None, str, str | None]:
    """Return ``(api_status, answers, error_message, source, returned_model)``.

    ``returned_model`` is observation-only and is logged + recorded alongside
    each batch; the cache key is keyed on the *requested* model so reruns
    are reproducible regardless of provider-side routing or model aliasing.
    """
    answering_cfg = cfg["answering"]
    client_cfg = cfg.get("client", {})
    model = answering_cfg["model"]
    temperature = float(answering_cfg.get("temperature", 0.0))
    top_p = float(answering_cfg.get("top_p", 1.0))
    questions = _question_payload(qa_rows)
    prompt_hash = hash_text(prompt)
    input_hash = hash_json(
        {
            "task": TASK,
            "prompt_version": prompt_version,
            "prompt_hash": prompt_hash,
            "context_hash": hash_text(context),
            "question_list_hash": hash_json(questions),
            "pipeline": pipeline,
            "condition": condition,
            "page_id": page_id,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
        }
    )
    cache_key = build_cache_key(model, temperature, prompt_hash, input_hash, TASK)
    cache_dir = pm.exp1_output_root / "api_cache" / TASK
    ensure_dir(cache_dir)

    if answering_cfg.get("cache_enabled", True):
        cached = read_cache(cache_dir, cache_key)
        if cached is not None:
            try:
                parsed = parse_json_response_text(str(cached.get("content", "")))
                answers = _normalize_model_answers(qa_rows, parsed)
                cached_returned_model = (
                    cached.get("returned_model") if isinstance(cached, dict) else None
                )
                log_api_event(
                    api_log_path,
                    {
                        "event": "cache_hit",
                        "task": TASK,
                        "model": model,
                        "requested_model": model,
                        "returned_model": cached_returned_model,
                        "pipeline": pipeline,
                        "condition": condition,
                        "page_id": page_id,
                        "cache_key": cache_key,
                    },
                )
                return "success", answers, None, "cache", cached_returned_model
            except Exception:
                log_api_event(
                    api_log_path,
                    {
                        "event": "cache_invalid",
                        "task": TASK,
                        "model": model,
                        "requested_model": model,
                        "returned_model": None,
                        "pipeline": pipeline,
                        "condition": condition,
                        "page_id": page_id,
                        "cache_key": cache_key,
                    },
                )

    attempt_state: dict[str, Any] = {"returned_model": None}

    def _attempt() -> str:
        result = chat_completion(
            _messages(prompt, context, questions),
            model=model,
            api_key_env=cfg.get("api_key_env", "DEEPSEEK_API_KEY"),
            base_url=client_cfg.get("base_url", "https://api.deepseek.com"),
            temperature=temperature,
            top_p=top_p,
            max_tokens=int(answering_cfg.get("max_tokens", 1024)),
            timeout_seconds=min(int(client_cfg.get("timeout_seconds", 60)), 30),
        )
        attempt_returned_model = result.get("returned_model")
        attempt_state["returned_model"] = attempt_returned_model
        log_api_event(
            api_log_path,
            {
                "event": "api_call",
                "task": TASK,
                "model": model,
                "requested_model": result.get("requested_model", model),
                "returned_model": attempt_returned_model,
                "ok": result.get("ok"),
                "pipeline": pipeline,
                "condition": condition,
                "page_id": page_id,
                "cache_key": cache_key,
            },
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "DeepSeek API failed")
        content = str(result.get("content") or "")
        parsed = parse_json_response_text(content)
        _normalize_model_answers(qa_rows, parsed)
        write_cache(
            cache_dir,
            cache_key,
            {
                "content": content,
                "model": model,
                "requested_model": model,
                "returned_model": attempt_returned_model,
            },
        )
        return content

    content = retry_call(
        _attempt,
        max_retries=0,
        sleep_seconds=float(client_cfg.get("retry_backoff_seconds", 1.0)),
        backoff=True,
    )
    parsed = parse_json_response_text(content)
    return (
        "success",
        _normalize_model_answers(qa_rows, parsed),
        None,
        "api",
        attempt_state.get("returned_model"),
    )


def _write_failed_call(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    fieldnames = ["timestamp", "pipeline", "condition", "page_id", "api_status", "error_message"]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def _validate_api_key(api_key: str, api_key_env: str) -> None:
    """Fail fast on placeholder or non-header-safe API keys without logging the key."""
    if not api_key:
        raise RuntimeError(f"{api_key_env} is missing; refusing to call DeepSeek")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            f"{api_key_env} contains non-ASCII characters; set the real sk-... key, not a placeholder."
        ) from exc
    if not api_key.startswith("sk-"):
        raise RuntimeError(f"{api_key_env} does not look like a DeepSeek API key; expected prefix sk-.")


def _is_reusable_existing_record(record: dict[str, Any]) -> bool:
    """Return True for completed records that should not be recomputed."""
    api_status = str(record.get("api_status", ""))
    return api_status in {"success", "cache_hit", "skipped_parser_empty", "skipped_parser_failed", "skipped_context_too_long"}


def _has_api_failed_answer(record: dict[str, Any]) -> bool:
    """Return True if a batch or any answer item is marked as API failed."""
    if str(record.get("api_status", "")) == "api_failed":
        return True
    answers = record.get("answers", [])
    if not isinstance(answers, list):
        return True
    return any(isinstance(item, dict) and str(item.get("status", "")) == "api_failed" for item in answers)


def _is_complete_existing_record(record: dict[str, Any], expected_answer_count: int) -> bool:
    """Return True only when an existing batch is structurally complete."""
    required = {
        "page_id",
        "pipeline",
        "condition",
        "model",
        "prompt_version",
        "context_truncated",
        "parser_status",
        "api_status",
        "error_message",
        "answers",
    }
    if not isinstance(record, dict) or required - set(record):
        return False
    answers = record.get("answers")
    if not isinstance(answers, list) or len(answers) < expected_answer_count:
        return False
    answer_required = {"qa_id", "question", "pred_answer", "status", "supporting_quote"}
    if any(not isinstance(item, dict) or answer_required - set(item) for item in answers):
        return False
    return True


def _should_skip_existing(record: dict[str, Any], expected_answer_count: int) -> bool:
    """Skip only completed non-failed batches."""
    return (
        _is_complete_existing_record(record, expected_answer_count)
        and _is_reusable_existing_record(record)
        and not _has_api_failed_answer(record)
    )


def _summarize_outputs(output_records: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for pipeline in PIPELINES:
        for condition in CONDITIONS:
            rows = output_records.get((pipeline, condition), [])
            key = f"{pipeline}_{condition}"
            summary[key] = {
                "answer_batches": len(rows),
                "answer_items": sum(len(row.get("answers", [])) for row in rows),
            }
    return summary


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# QA Answering Debug Summary",
        "",
        f"- total_page_condition_batches: `{summary['total_page_condition_batches']}`",
        f"- api_called_batches: `{summary['api_called_batches']}`",
        f"- cache_hit_batches: `{summary['cache_hit_batches']}`",
        f"- parser_empty_batches: `{summary['parser_empty_batches']}`",
        f"- parser_failed_batches: `{summary['parser_failed_batches']}`",
        f"- context_too_long_batches: `{summary['context_too_long_batches']}`",
        f"- context_truncated_batches: `{summary['context_truncated_batches']}`",
        f"- api_failed_batches: `{summary['api_failed_batches']}`",
        f"- skipped_existing_batches: `{summary['skipped_existing_batches']}`",
        f"- total_answer_items: `{summary['total_answer_items']}`",
        f"- answered_count: `{summary['answered_count']}`",
        f"- not_found_count: `{summary['not_found_count']}`",
        f"- parser_empty_answer_count: `{summary['parser_empty_answer_count']}`",
        f"- parser_failed_answer_count: `{summary['parser_failed_answer_count']}`",
        f"- api_failed_answer_count: `{summary['api_failed_answer_count']}`",
        f"- summary_path: `{path}`",
        "",
        "## Outputs By Pipeline Condition",
        "",
    ]
    for key, item in summary["outputs_by_pipeline_condition"].items():
        lines.extend(
            [
                f"### {key}",
                "",
                f"- answer_batches: `{item['answer_batches']}`",
                f"- answer_items: `{item['answer_items']}`",
                f"- path: `{item['path']}`",
                "",
            ]
        )
    write_text(path, "\n".join(lines))


def run_qa_answering(config_path: str | Path, debug: bool = False, retry_failed_only: bool = False) -> dict[str, Any]:
    config_path = Path(config_path)
    base_config = config_path.parent / "base.yaml"
    pm = PathManager(base_config, create_dirs=True)
    exp_cfg = read_yaml(config_path)
    deepseek_path = _resolve_config_path(exp_cfg.get("answering", {}).get("provider_config", "configs/deepseek.yaml"), pm.project_root, config_path)
    deepseek_cfg = read_yaml(deepseek_path)

    api_key_env = deepseek_cfg.get("api_key_env", "DEEPSEEK_API_KEY")
    _validate_api_key(os.environ.get(api_key_env, ""), api_key_env)

    prompt_path = pm.project_root / "experiment_add/exp1_qa/prompts/qa_answering_prompt.txt"
    prompt = read_text(prompt_path)
    lowered_prompt = prompt.lower()
    forbidden_terms = ("gold_answer", "evidence_text", "evidence_bbox")
    if any(term in lowered_prompt for term in forbidden_terms):
        raise RuntimeError("QA answering prompt contains forbidden gold/evidence terms")

    manifest = load_manifest(pm.page_manifest_debug20 if debug else pm.page_manifest_500)
    manifest_ids = {row["page_id"] for row in manifest}
    qa_rows = read_jsonl(pm.qa_pairs_shared_path)
    qa_by_page = _group_qa_by_page(qa_rows, manifest_ids)

    context_cfg = exp_cfg.get("context", {})
    max_chars = int(context_cfg.get("max_chars", 30000))
    allow_truncation = bool(context_cfg.get("allow_truncation", False))

    api_log_path = pm.exp1_log_root / "qa_answering_api_log.jsonl"
    failed_log_path = pm.exp1_log_root / "failed_qa_calls.csv"
    summary_path = pm.exp1_log_root / ("qa_answering_debug_summary.md" if debug else "qa_answering_summary.md")
    ensure_dir(pm.exp1_log_root)

    parse_outputs = {
        (pipeline, condition): _load_parse_outputs(pm, pipeline, condition)
        for pipeline in PIPELINES
        for condition in CONDITIONS
    }
    output_records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    answer_status_counts: Counter[str] = Counter()
    model = deepseek_cfg["answering"]["model"]
    prompt_version = str(deepseek_cfg.get("answering", {}).get("qa_answering_prompt_version", DEFAULT_PROMPT_VERSION))

    for pipeline in PIPELINES:
        for condition in CONDITIONS:
            parse_by_page = parse_outputs[(pipeline, condition)]
            out_path = pm.exp1_answers_path(pipeline, condition)
            existing_by_page = {str(row.get("page_id", "")): row for row in read_jsonl(out_path) if isinstance(row, dict)}
            for page_id, page_qas in qa_by_page.items():
                counts["total_page_condition_batches"] += 1
                existing = existing_by_page.get(page_id)
                if existing is not None and _should_skip_existing(existing, len(page_qas)):
                    record = existing
                    counts["skipped_existing_batches"] += 1
                    print(
                        f"[skip-existing] pipeline={pipeline} condition={condition} page_id={page_id} api_status={record.get('api_status')}",
                        flush=True,
                    )
                    output_records[(pipeline, condition)].append(record)
                    for answer in record.get("answers", []):
                        answer_status_counts[str(answer.get("status", ""))] += 1
                    atomic_write_jsonl(out_path, output_records[(pipeline, condition)])
                    continue
                if retry_failed_only:
                    print(
                        f"[retry-failed] pipeline={pipeline} condition={condition} page_id={page_id} api_status={existing.get('api_status') if existing else 'missing'}",
                        flush=True,
                    )

                parsed = parse_by_page.get(page_id)
                if parsed is None:
                    record = _skipped_batch(page_id, pipeline, condition, page_qas, "missing", "skipped_parser_failed", "parser_failed", "parser output missing", prompt_version=prompt_version)
                    counts["parser_failed_batches"] += 1
                else:
                    parser_status = str(parsed.get("parser_status", ""))
                    page_text = str(parsed.get("page_text", "") or "")
                    blocks = parsed.get("blocks", [])
                    if parser_status == "failed":
                        record = _skipped_batch(page_id, pipeline, condition, page_qas, parser_status, "skipped_parser_failed", "parser_failed", "parser output failed", prompt_version=prompt_version)
                        counts["parser_failed_batches"] += 1
                    elif parser_status == "empty" or not page_text.strip() or (not blocks and not page_text.strip()):
                        record = _skipped_batch(page_id, pipeline, condition, page_qas, parser_status or "empty", "skipped_parser_empty", "parser_empty", "parser output is empty", prompt_version=prompt_version)
                        counts["parser_empty_batches"] += 1
                    else:
                        context_truncated = False
                        context = page_text
                        if len(context) > max_chars:
                            if not allow_truncation:
                                record = _failed_batch(
                                    page_id,
                                    pipeline,
                                    condition,
                                    page_qas,
                                    model,
                                    parser_status,
                                    "skipped_context_too_long",
                                    f"context length {len(context)} > max_chars {max_chars}",
                                    prompt_version=prompt_version,
                                )
                                counts["context_too_long_batches"] += 1
                            else:
                                context = context[:max_chars]
                                context_truncated = True
                                counts["context_truncated_batches"] += 1
                        if len(page_text) <= max_chars or allow_truncation:
                            try:
                                print(
                                    f"[api-start] pipeline={pipeline} condition={condition} page_id={page_id} questions={len(page_qas)}",
                                    flush=True,
                                )
                                api_status, answers, error, source, returned_model = _call_or_cache(
                                    page_id,
                                    pipeline,
                                    condition,
                                    context,
                                    page_qas,
                                    prompt,
                                    deepseek_cfg,
                                    pm,
                                    api_log_path,
                                    prompt_version,
                                )
                                counts["api_called_batches"] += 1 if source == "api" else 0
                                counts["cache_hit_batches"] += 1 if source == "cache" else 0
                                print(
                                    f"[api-done] pipeline={pipeline} condition={condition} page_id={page_id} source={source} answers={len(answers)} returned_model={returned_model}",
                                    flush=True,
                                )
                                record = {
                                    "page_id": page_id,
                                    "pipeline": pipeline,
                                    "condition": condition,
                                    "model": model,
                                    "requested_model": model,
                                    "returned_model": returned_model,
                                    "prompt_version": prompt_version,
                                    "context_truncated": context_truncated,
                                    "parser_status": parser_status,
                                    "api_status": api_status,
                                    "error_message": error,
                                    "answers": answers,
                                }
                            except Exception as exc:
                                record = _failed_batch(page_id, pipeline, condition, page_qas, model, parser_status, "api_failed", str(exc), prompt_version=prompt_version)
                                counts["api_called_batches"] += 1
                                counts["api_failed_batches"] += 1
                                print(
                                    f"[api-failed] pipeline={pipeline} condition={condition} page_id={page_id} error={str(exc)[:200]}",
                                    flush=True,
                                )
                                _write_failed_call(
                                    failed_log_path,
                                    {
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "pipeline": pipeline,
                                        "condition": condition,
                                        "page_id": page_id,
                                        "api_status": "api_failed",
                                        "error_message": str(exc),
                                    },
                                )
                output_records[(pipeline, condition)].append(record)
                for answer in record.get("answers", []):
                    answer_status_counts[str(answer.get("status", ""))] += 1
                atomic_write_jsonl(out_path, output_records[(pipeline, condition)])

    outputs = _summarize_outputs(output_records)
    outputs_with_paths = {}
    for pipeline in PIPELINES:
        for condition in CONDITIONS:
            key = f"{pipeline}_{condition}"
            outputs_with_paths[key] = {
                **outputs[key],
                "path": str(pm.exp1_answers_path(pipeline, condition)),
            }

    summary = {
        "total_page_condition_batches": counts["total_page_condition_batches"],
        "api_called_batches": counts["api_called_batches"],
        "cache_hit_batches": counts["cache_hit_batches"],
        "parser_empty_batches": counts["parser_empty_batches"],
        "parser_failed_batches": counts["parser_failed_batches"],
        "context_too_long_batches": counts["context_too_long_batches"],
        "context_truncated_batches": counts["context_truncated_batches"],
        "api_failed_batches": counts["api_failed_batches"],
        "skipped_existing_batches": counts["skipped_existing_batches"],
        "total_answer_items": sum(answer_status_counts.values()),
        "answered_count": answer_status_counts["answered"],
        "not_found_count": answer_status_counts["not_found"],
        "parser_empty_answer_count": answer_status_counts["parser_empty"],
        "parser_failed_answer_count": answer_status_counts["parser_failed"],
        "api_failed_answer_count": answer_status_counts["api_failed"],
        "outputs_by_pipeline_condition": outputs_with_paths,
        "api_log_path": str(api_log_path),
        "failed_log_path": str(failed_log_path),
        "summary_path": str(summary_path),
    }
    _write_summary(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run debug QA answering on clean and perturbed parser outputs.")
    parser.add_argument("--config", default="experiment_add/configs/exp1_qa.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--retry-failed-only", action="store_true", help="Only recompute existing failed/incomplete batches.")
    args = parser.parse_args()
    result = run_qa_answering(args.config, debug=args.debug, retry_failed_only=args.retry_failed_only)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["api_failed_batches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
