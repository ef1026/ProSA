# Downstream QA and Retrieval Metrics Reproduction

`experiment_add/` contains the downstream experiments used to measure how parser
perturbations propagate into task metrics:

- `exp1_qa`: single-page QA exact match, F1, answer-missing rate, QA drop, and
  failure decomposition.
- `exp2_retrieval`: page-internal retrieval Recall@k, MRR@10, AnswerHit@k,
  drops, per-TOR efficiency, and failure decomposition.

The pipeline is end-to-end: build manifests, parse clean pages, perturb pages,
parse perturbed pages, compute shared parser metrics, generate shared QA pairs,
run QA, run retrieval, then evaluate and audit metrics.

## Environment

Run from the repository root with the `advdoc` environment active:

```bash
conda activate advdoc
export DEEPSEEK_API_KEY=<your-deepseek-api-key>
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export SENTENCE_TRANSFORMERS_HOME=$HF_HOME
export PPOCR_PYTHON=$(conda run -n paddle which python)
```

The pinned environment records are:

```text
experiment_add/env/advdoc_environment.yml
experiment_add/env/advdoc_pip_freeze.txt
experiment_add/env/paddle_environment.yml
experiment_add/env/paddle_pip_freeze.txt
```

## Inputs and Configs

Required images:

```text
data/selected/publaynet/images/
data/selected/doclaynet/images/
```

Core configs:

```text
experiment_add/configs/base.yaml
experiment_add/configs/parser.yaml
experiment_add/configs/perturb.yaml
experiment_add/configs/deepseek.yaml
experiment_add/configs/exp1_qa.yaml
experiment_add/exp2_retrieval/configs/exp2_retrieval.yaml
```

## Preflight

```bash
python3 experiment_add/shared/parsers/parser_env_preflight.py \
  --config experiment_add/configs/parser.yaml --strict
```

## 1. Build Page Manifest

```bash
python3 experiment_add/shared/data/build_page_manifest.py \
  --config experiment_add/configs/base.yaml
```

Debug subset:

```bash
python3 experiment_add/shared/data/build_page_manifest.py \
  --config experiment_add/configs/base.yaml --debug
```

Outputs:

```text
experiment_add/data/page_manifest_500.csv
experiment_add/data/page_manifest_debug20.csv
experiment_add/data/selected_pages_report.md
```

## 2. Clean Parse

```bash
python3 experiment_add/shared/parsers/run_clean_parse.py \
  --config experiment_add/configs/parser.yaml --pipeline mineru
python3 experiment_add/shared/parsers/run_clean_parse.py \
  --config experiment_add/configs/parser.yaml --pipeline ppstructure
python3 experiment_add/shared/parsers/audit_clean_parse.py \
  --config experiment_add/configs/parser.yaml
```

Outputs:

```text
experiment_add/outputs/shared/clean_parse/{mineru,ppstructure}/merged_clean_*.jsonl
experiment_add/outputs/shared/clean_parse/{mineru,ppstructure}/pages/*.json
experiment_add/logs/shared/clean_parse_*audit_report.md
```

## 3. Generate Perturbed Pages

```bash
python3 experiment_add/shared/perturb/select_probe_conditions.py \
  --config experiment_add/configs/perturb.yaml
python3 experiment_add/shared/perturb/generate_perturbed_pages.py \
  --config experiment_add/configs/perturb.yaml
python3 experiment_add/shared/perturb/audit_perturbation.py \
  --config experiment_add/configs/perturb.yaml
```

Outputs:

```text
experiment_add/outputs/shared/perturbed_pages/{area_matched_erasure,structural_probe,large_area_erasure}/images/*.png
experiment_add/outputs/shared/perturbed_pages/merged_perturb_metadata.jsonl
experiment_add/outputs/shared/perturbed_pages/perturb_summary.csv
experiment_add/outputs/shared/perturbed_pages/probe_selection.json
```

## 4. Perturbed Parse

```bash
for COND in area_matched_erasure structural_probe large_area_erasure; do
  for PIPE in mineru ppstructure; do
    python3 experiment_add/shared/parsers/run_perturbed_parse.py \
      --config experiment_add/configs/parser.yaml \
      --pipeline $PIPE --condition $COND
  done
done

python3 experiment_add/shared/parsers/audit_perturbed_parse.py \
  --config experiment_add/configs/parser.yaml
```

Outputs:

```text
experiment_add/outputs/shared/perturbed_parse/{mineru,ppstructure}/{condition}/merged.jsonl
experiment_add/outputs/shared/perturbed_parse/{mineru,ppstructure}/{condition}/pages/*.json
```

## 5. Shared Parser Metrics

```bash
python3 experiment_add/shared/metrics/build_shared_parser_metrics.py \
  --config experiment_add/configs/base.yaml
```

Outputs:

```text
experiment_add/outputs/shared/parser_metrics/parser_metrics_merged.jsonl
experiment_add/outputs/shared/parser_metrics/merged_parser_metrics.csv
```

## 6. Shared QA Pairs

This step calls DeepSeek.

```bash
python3 experiment_add/shared/qa_generation/qa_generation_preflight.py \
  --config experiment_add/configs/deepseek.yaml
python3 experiment_add/shared/qa_generation/generate_qa_candidates.py \
  --config experiment_add/configs/deepseek.yaml \
  --candidates-per-page 20 --prompt-version v2
python3 experiment_add/shared/qa_generation/filter_qa_pairs.py \
  --config experiment_add/configs/deepseek.yaml
python3 experiment_add/shared/qa_generation/audit_qa_pairs.py
```

Outputs:

```text
experiment_add/outputs/shared/qa_pairs/qa_candidates_raw.jsonl
experiment_add/outputs/shared/qa_pairs/qa_pairs_filtered.jsonl
experiment_add/outputs/shared/qa_pairs/qa_pairs_shared.jsonl
experiment_add/outputs/shared/qa_pairs/qa_filter_report.md
```

## 7. Exp1 QA Metrics

This step calls DeepSeek for QA answering. Answers are cached under
`experiment_add/outputs/exp1_qa/api_cache/qa_answering/`.

```bash
python3 experiment_add/exp1_qa/scripts/01_run_qa_answering.py
python3 experiment_add/exp1_qa/scripts/02_evaluate_qa_results.py
python3 experiment_add/exp1_qa/scripts/audit_qa_metrics.py
```

Metric outputs:

```text
experiment_add/outputs/exp1_qa/metrics/qa_metrics_by_pipeline_condition.csv
experiment_add/outputs/exp1_qa/metrics/qa_metrics_by_page.csv
experiment_add/outputs/exp1_qa/metrics/qa_metrics_by_question_type.csv
experiment_add/outputs/exp1_qa/metrics/qa_metrics_non_overlap_subset.csv
experiment_add/outputs/exp1_qa/metrics/qa_metrics_correlations.csv
experiment_add/outputs/exp1_qa/metrics/qa_failure_decomposition.csv
experiment_add/logs/exp1_qa/qa_evaluation_summary.md
```

## 8. Exp2 Retrieval Metrics

```bash
python3 experiment_add/exp2_retrieval/scripts/01_build_retrieval_corpus.py
python3 experiment_add/exp2_retrieval/scripts/audit_retrieval_corpus.py
python3 experiment_add/exp2_retrieval/scripts/02_build_indexes.py
python3 experiment_add/exp2_retrieval/scripts/03_run_retrieval.py
python3 experiment_add/exp2_retrieval/scripts/04_evaluate_retrieval.py
python3 experiment_add/exp2_retrieval/scripts/audit_retrieval_metrics.py
```

Metric outputs:

```text
experiment_add/outputs/exp2_retrieval/metrics/retrieval_metrics_by_pipeline_condition.csv
experiment_add/outputs/exp2_retrieval/metrics/retrieval_metrics_by_page.csv
experiment_add/outputs/exp2_retrieval/metrics/retrieval_metrics_by_question_type.csv
experiment_add/outputs/exp2_retrieval/metrics/retrieval_metrics_non_overlap_subset.csv
experiment_add/outputs/exp2_retrieval/metrics/retrieval_correlations.csv
experiment_add/outputs/exp2_retrieval/metrics/retrieval_failure_decomposition.csv
experiment_add/logs/exp2_retrieval/{corpus,index,retrieval_run,retrieval_evaluation,audit_corpus,audit_metrics}_summary.md
```

## Audit Gate

After evaluation, both audit scripts should finish without warnings:

```bash
python3 experiment_add/exp1_qa/scripts/audit_qa_metrics.py
python3 experiment_add/exp2_retrieval/scripts/audit_retrieval_metrics.py
```

The headline files for paper tables are:

```text
experiment_add/outputs/exp1_qa/metrics/qa_metrics_by_pipeline_condition.csv
experiment_add/outputs/exp2_retrieval/metrics/retrieval_metrics_by_pipeline_condition.csv
```

Generated `data/`, `outputs/`, `logs/`, HF caches, and API caches are not part
of the code release.
