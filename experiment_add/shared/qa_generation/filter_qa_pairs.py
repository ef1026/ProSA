from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiment_add.shared.qa_generation.locate_evidence_blocks import locate_evidence_blocks
from experiment_add.shared.text.answer_matching import count_answer_occurrences
from experiment_add.shared.text.normalize_text import normalize_answer
from experiment_add.shared.utils.io import atomic_write_jsonl, read_jsonl, write_text
from experiment_add.shared.utils.path_manager import PathManager


GENERIC_ANSWERS = {"the paper", "this study", "the method", "the result", "the authors", "the article"}


def _token_len(text: str) -> int:
    return len(normalize_answer(text).split())


def _load_clean(pm: PathManager) -> dict[str, dict[str, dict[str, Any]]]:
    return {p: {r["page_id"]: r for r in read_jsonl(pm.clean_parse_merged_path(p))} for p in ("mineru", "ppstructure")}


def _reason_candidate(candidate: dict[str, Any], source_text: str, mineru_text: str, pp_text: str) -> list[str]:
    reasons = []
    raw_q = candidate.get("question")
    raw_a = candidate.get("answer")
    q = "" if raw_q is None else str(raw_q).strip()
    a = "" if raw_a is None else str(raw_a).strip()
    norm_a = normalize_answer(raw_a)
    if not q:
        reasons.append("empty_question")
    if raw_a is None or not a:
        reasons.append("empty_answer")
    if not norm_a:
        reasons.append("empty_normalized_answer")
    n = len(norm_a.split())
    if n < 2 or n > 15:
        reasons.append("answer_length_out_of_range")
    if norm_a in GENERIC_ANSWERS:
        reasons.append("generic_answer")
    if norm_a not in normalize_answer(source_text):
        reasons.append("answer_not_in_source_text")
    if norm_a not in normalize_answer(mineru_text):
        reasons.append("answer_not_in_mineru_text")
    if norm_a not in normalize_answer(pp_text):
        reasons.append("answer_not_in_ppstructure_text")
    if count_answer_occurrences(mineru_text, a) > 3:
        reasons.append("answer_too_frequent_mineru")
    if count_answer_occurrences(pp_text, a) > 3:
        reasons.append("answer_too_frequent_ppstructure")
    return reasons


def filter_qa_pairs(
    config_path: str | Path,
    debug: bool = False,
    qa_pairs_suffix: str | None = None,
) -> dict[str, Any]:
    """Filter raw QA candidates into shared QA pairs.

    When ``qa_pairs_suffix`` is provided (e.g. ``"v2_debug"``), reads the
    raw candidates from and writes the filtered shared QA pairs to
    ``experiment_add/outputs/shared/qa_pairs_<suffix>/`` instead of the
    canonical ``qa_pairs/`` directory.
    """
    pm = PathManager(config_path, create_dirs=True, qa_pairs_suffix=qa_pairs_suffix)
    raw_rows = read_jsonl(pm.qa_candidates_raw_path)
    clean = _load_clean(pm)
    kept_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reason_counts: Counter[str] = Counter()
    raw_count = 0
    for row in raw_rows:
        page_id = row.get("page_id")
        source_pipeline = row.get("generation_source_pipeline")
        mineru = clean["mineru"].get(page_id, {})
        pp = clean["ppstructure"].get(page_id, {})
        source = mineru if source_pipeline == "mineru" else pp
        source_text = str(row.get("generation_source_text") or source.get("page_text", ""))
        mineru_text = str(mineru.get("page_text", ""))
        pp_text = str(pp.get("page_text", ""))
        seen_questions: set[str] = set()
        seen_answers: set[str] = set()
        page_kept: list[dict[str, Any]] = []
        for candidate in row.get("candidates", []):
            raw_count += 1
            reasons = _reason_candidate(candidate, source_text, mineru_text, pp_text)
            nq = normalize_answer(candidate.get("question", ""))
            na = normalize_answer(candidate.get("answer", ""))
            if nq in seen_questions:
                reasons.append("duplicate_question")
            if na in seen_answers:
                reasons.append("duplicate_answer")
            evidence = locate_evidence_blocks(str(candidate.get("answer", "")), str(candidate.get("evidence_text", "")), mineru, pp)
            if not (evidence["evidence_block_id_mineru"] or evidence["evidence_block_id_ppstructure"]):
                reasons.append("evidence_block_not_found")
            if reasons:
                reason_counts.update(reasons)
                continue
            item = {
                "page_id": page_id,
                "question": str(candidate["question"]).strip(),
                "gold_answer": str(candidate["answer"]).strip(),
                "gold_answer_normalized": na,
                "evidence_text": str(candidate.get("evidence_text", "")).strip(),
                "answer_type": str(candidate.get("answer_type", "span")).strip() or "span",
                "generation_source_pipeline": source_pipeline,
                "generation_source_mode": row.get("generation_source_mode", "unknown"),
                "fallback_zero_qa": bool(row.get("fallback_zero_qa", False)),
                "answer_occurrences_clean_mineru": count_answer_occurrences(mineru_text, candidate["answer"]),
                "answer_occurrences_clean_ppstructure": count_answer_occurrences(pp_text, candidate["answer"]),
                **evidence,
            }
            page_kept.append(item)
            seen_questions.add(nq)
            seen_answers.add(na)
        # Prefer answer-type diversity, max 4 per page.
        selected: list[dict[str, Any]] = []
        used_types: set[str] = set()
        for item in page_kept:
            if item["answer_type"] not in used_types:
                selected.append(item)
                used_types.add(item["answer_type"])
            if len(selected) >= 4:
                break
        for item in page_kept:
            if len(selected) >= 4:
                break
            if item not in selected:
                selected.append(item)
        kept_by_page[page_id].extend(selected)

    shared: list[dict[str, Any]] = []
    for page_id, items in kept_by_page.items():
        # Enforce a global max of four shared QA per page after merging fallback rows.
        final_items: list[dict[str, Any]] = []
        seen_final_answers: set[str] = set()
        for item in items:
            answer_key = normalize_answer(item.get("gold_answer", ""))
            if answer_key in seen_final_answers:
                continue
            final_items.append(item)
            seen_final_answers.add(answer_key)
            if len(final_items) >= 4:
                break
        for idx, item in enumerate(final_items, start=1):
            shared.append({"qa_id": f"{Path(page_id).stem}_q{idx:03d}", **item})
    filtered_path = pm.qa_pairs_filtered_path
    shared_path = pm.qa_pairs_shared_path
    mineru_path = pm.qa_pairs_dir / "qa_pairs_pipeline_specific_mineru.jsonl"
    pp_path = pm.qa_pairs_dir / "qa_pairs_pipeline_specific_ppstructure.jsonl"
    atomic_write_jsonl(filtered_path, shared)
    atomic_write_jsonl(shared_path, shared)
    atomic_write_jsonl(mineru_path, shared)
    atomic_write_jsonl(pp_path, shared)
    pages_with_qa = {item["page_id"] for item in shared}
    manifest_count = 20 if debug else len({row.get("page_id") for row in raw_rows})
    fallback_pages = {row.get("page_id") for row in raw_rows if row.get("fallback_zero_qa")}
    fallback_added_qa = sum(1 for item in shared if item.get("fallback_zero_qa"))
    report_path = pm.qa_pairs_dir / "qa_filter_report.md"
    lines = [
        "# QA Filter Report",
        "",
        f"- mode: `{'debug' if debug else 'full'}`",
        f"- qa_pairs_suffix: `{pm.qa_pairs_suffix or 'canonical'}`",
        f"- raw_candidates: `{raw_count}`",
        f"- filtered_qa: `{len(shared)}`",
        f"- shared_qa: `{len(shared)}`",
        f"- pages_with_qa: `{len(pages_with_qa)}`",
        f"- pages_with_0_qa: `{max(0, manifest_count - len(pages_with_qa))}`",
        f"- average_qa_per_page: `{(len(shared) / manifest_count) if manifest_count else 0:.3f}`",
        f"- main_filter_reasons: `{dict(reason_counts.most_common(20))}`",
        f"- fallback_zero_qa_enabled: `{bool(fallback_pages)}`",
        f"- fallback_pages_count: `{len(fallback_pages)}`",
        f"- fallback_added_qa_count: `{fallback_added_qa}`",
        f"- output_shared: `{shared_path}`",
        "",
        "QA pairs are generated from clean parser output only and must be reused across all perturbation conditions.",
    ]
    write_text(report_path, "\n".join(lines) + "\n")
    optimization_path = pm.qa_pairs_dir / "qa_coverage_optimization_report.md"
    old_shared_qa = 38 if debug else "unknown"
    old_pages_with_qa = 12 if debug else "unknown"
    old_avg = 1.9 if debug else "unknown"
    ready = len(shared) >= 40 and len(pages_with_qa) >= 15
    optimization_lines = [
        "# QA Coverage Optimization Report",
        "",
        f"- qa_pairs_suffix: `{pm.qa_pairs_suffix or 'canonical'}`",
        f"- old_shared_qa: `{old_shared_qa}`",
        f"- new_shared_qa: `{len(shared)}`",
        f"- old_pages_with_qa: `{old_pages_with_qa}`",
        f"- new_pages_with_qa: `{len(pages_with_qa)}`",
        f"- old_average_qa_per_page: `{old_avg}`",
        f"- new_average_qa_per_page: `{(len(shared) / manifest_count) if manifest_count else 0:.3f}`",
        "- old_main_filter_reasons: `{'answer_not_in_ppstructure_text': 72, 'answer_length_out_of_range': 67, 'answer_not_in_mineru_text': 32, 'answer_too_frequent_mineru': 15, 'answer_not_in_source_text': 14, 'answer_too_frequent_ppstructure': 14}`",
        f"- new_main_filter_reasons: `{dict(reason_counts.most_common(20))}`",
        "- additional_api_calls: `see qa_generation_debug_summary.md api_calls/cache_hits`",
        "- cache_hits: `see qa_generation_debug_summary.md api_calls/cache_hits`",
        "- prompt_version: `v2`",
        "- candidates_per_page: `15`",
        f"- fallback_zero_qa_enabled: `{bool(fallback_pages)}`",
        f"- fallback_pages_count: `{len(fallback_pages)}`",
        f"- fallback_added_qa_count: `{fallback_added_qa}`",
        "- answer_presence_failures_mineru: `0 if audit_qa_pairs passes`",
        "- answer_presence_failures_ppstructure: `0 if audit_qa_pairs passes`",
        f"- ready_for_full500_qa_generation: `{'YES' if ready else 'BORDERLINE'}`",
        "",
        "Full500 should use prompt v2 and candidates_per_page >= 15. If debug coverage remains borderline, use 20 candidates/page before full500.",
    ]
    write_text(optimization_path, "\n".join(optimization_lines) + "\n")
    return {
        "qa_pairs_suffix": pm.qa_pairs_suffix or "canonical",
        "raw_candidates": raw_count,
        "filtered_qa": len(shared),
        "shared_qa": len(shared),
        "average_qa_per_page": (len(shared) / manifest_count) if manifest_count else 0,
        "pages_with_0_qa": max(0, manifest_count - len(pages_with_qa)),
        "main_filter_reasons": dict(reason_counts.most_common(20)),
        "fallback_zero_qa_enabled": bool(fallback_pages),
        "fallback_pages_count": len(fallback_pages),
        "fallback_added_qa_count": fallback_added_qa,
        "qa_filter_report": str(report_path),
        "qa_coverage_optimization_report": str(optimization_path),
        "qa_pairs_shared": str(shared_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter raw QA candidates into shared QA pairs.")
    parser.add_argument("--config", default="experiment_add/configs/base.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--output-suffix",
        default=None,
        help=(
            "If set, read/write QA pair artifacts under "
            "experiment_add/outputs/shared/qa_pairs_<suffix>/ instead of "
            "the canonical qa_pairs/ directory."
        ),
    )
    parser.add_argument(
        "--input-suffix",
        default=None,
        help=(
            "Alias for --output-suffix kept for compatibility with the "
            "task spec. If both are set, they must be equal."
        ),
    )
    args = parser.parse_args()
    if args.output_suffix and args.input_suffix and args.output_suffix != args.input_suffix:
        parser.error("--output-suffix and --input-suffix must match if both are given")
    suffix = args.output_suffix or args.input_suffix
    result = filter_qa_pairs(args.config, debug=args.debug, qa_pairs_suffix=suffix)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["shared_qa"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
