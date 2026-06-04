"""Non-overlap subset analysis for retrieval, mirroring exp1.

Re-uses exp1's ``evidence_bbox_for_pipeline`` and ``is_non_overlap_subset``
helpers so that the QA subset definition stays identical across exp1 and
exp2; this is what lets the paper say "even on the QA whose evidence does
not geometrically intersect the perturbation footprint, retrieval still
collapses under structural probes".
"""

from __future__ import annotations

from typing import Any

from experiment_add.exp1_qa.metrics.qa_non_overlap_analysis import (
    evidence_bbox_for_pipeline,
    is_non_overlap_subset,
)


def select_non_overlap_qa_ids(
    instances: list[dict[str, Any]],
    qa_by_id: dict[str, dict[str, Any]],
    metadata: dict[tuple[str, str], dict[str, Any]],
    pipeline: str,
    condition: str,
) -> set[str]:
    """Return the set of ``qa_id``s whose evidence bbox does not overlap support."""
    out: set[str] = set()
    for row in instances:
        if row.get("pipeline") != pipeline or row.get("condition") != condition:
            continue
        qa_id = str(row.get("qa_id", ""))
        page_id = str(row.get("page_id", ""))
        qa = qa_by_id.get(qa_id, {})
        evidence = evidence_bbox_for_pipeline(qa, pipeline)
        support = metadata.get((page_id, condition), {}).get("support_bbox")
        if is_non_overlap_subset(evidence, support):
            out.add(qa_id)
    return out
