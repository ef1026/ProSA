# exp2_retrieval

Page-internal retrieval robustness evaluation that closes the RAG loop on
the `experiment_add` shared artifacts (clean / perturbed parser outputs and
shared QA pairs). All inputs are reused; this experiment never re-runs
parsers, never regenerates QA pairs, and never modifies `shared/` or
`exp1_qa/`.

## Pipeline

1. `scripts/01_build_retrieval_corpus.py` &mdash; turn each
   `(pipeline, condition)` parser output into per-page `chunks.jsonl` files
   under `outputs/exp2_retrieval/corpora/`.
2. `scripts/02_build_indexes.py` &mdash; persist BM25 (per-page) plus dense
   embeddings (`sentence-transformers/all-MiniLM-L6-v2`) under
   `outputs/exp2_retrieval/indexes/`.
3. `scripts/03_run_retrieval.py` &mdash; for every QA, run BM25 + dense
   page-internal top-K and write `run.jsonl` files under
   `outputs/exp2_retrieval/retrieval_runs/` with `evidence_hit` /
   `answer_hit` flags pre-computed.
4. `scripts/04_evaluate_retrieval.py` &mdash; aggregate runs into the six CSV
   tables under `outputs/exp2_retrieval/metrics/`.

Audits:

- `scripts/audit_retrieval_corpus.py` &mdash; chunk count distribution, evidence
  / answer coverage on clean (must be >= 95%).
- `scripts/audit_retrieval_metrics.py` &mdash; clean Recall@5 thresholds and
  monotonicity (clean > perturbed).

## Quick start (debug20)

```bash
cd /root/autodl-tmp/ProSA_github_uploaded
conda activate advdoc

python3 experiment_add/exp2_retrieval/scripts/01_build_retrieval_corpus.py --debug
python3 experiment_add/exp2_retrieval/scripts/audit_retrieval_corpus.py --debug
python3 experiment_add/exp2_retrieval/scripts/02_build_indexes.py --debug
python3 experiment_add/exp2_retrieval/scripts/03_run_retrieval.py --debug
python3 experiment_add/exp2_retrieval/scripts/04_evaluate_retrieval.py --debug
python3 experiment_add/exp2_retrieval/scripts/audit_retrieval_metrics.py --debug
```

Drop `--debug` to switch to the full 500-page manifest.

## Dependencies

In the `advdoc` conda environment:

```bash
pip install rank_bm25==0.2.2 sentence-transformers==3.0.1
```

The dense encoder downloads `sentence-transformers/all-MiniLM-L6-v2` to
`/root/autodl-tmp/hf_cache` on first use; set `HF_ENDPOINT=https://hf-mirror.com`
if the default HuggingFace endpoint is slow.

## Outputs

```text
experiment_add/outputs/exp2_retrieval/
  corpora/{pipeline}_{condition}/{chunks,pages}.jsonl + corpus_summary.json
  indexes/{pipeline}_{condition}/{bm25.pkl, dense_embeddings.npz, dense_meta.json, manifest.json}
  retrieval_runs/{pipeline}_{condition}_{retriever}/run.jsonl
  metrics/
    retrieval_metrics_by_pipeline_condition.csv
    retrieval_metrics_by_page.csv
    retrieval_metrics_by_question_type.csv
    retrieval_metrics_non_overlap_subset.csv
    retrieval_correlations.csv
    retrieval_failure_decomposition.csv
experiment_add/logs/exp2_retrieval/
  corpus_*.md, index_*.md, retrieval_run_*.md, retrieval_evaluation_*.md
  audit_corpus_*.md, audit_metrics_*.md
```

## Reuse boundary

- All chunking, retrieval, and evaluation logic lives in this directory.
- `shared/` is **read-only** and used via Python imports
  (`shared.text.*`, `shared.utils.*`, `shared.metrics.*`,
  `shared.data.load_manifest`).
- `exp1_qa/metrics/qa_non_overlap_analysis.py` is imported but never modified
  so that exp1 and exp2 use the same non-overlap subset definition.
