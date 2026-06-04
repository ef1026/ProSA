# ProSA

Official code release for:

> How Do Document Parsers Break? Auditing Structural Vulnerability in Document Intelligence

Paper:
[arXiv:2605.19309](https://arxiv.org/abs/2605.19309) /
[DOI:10.48550/arXiv.2605.19309](https://doi.org/10.48550/arXiv.2605.19309)

Authors: Yue Chen, Yihao Wang, Ziyi Tang, Yongsen Zheng, and Keze Wang.

This repository contains the ProSA experiment code, parser adapters, frozen
attack plans, and downstream QA/retrieval metric pipeline. It intentionally
does not redistribute third-party raw datasets, selected page images, COCO
annotation files, OCR text, parser outputs, logs, model weights, or generated
reports.

## Repository Layout

```text
config/                 Frozen attack plans, Phase 2 replay logs, ID-only eval manifest
data/                   Placeholder only; users prepare local datasets here
parsers/                MinerU and PP-StructureV3 adapters
experiment/             Main ProSA experiment and metric code
experiment_add/         Downstream QA and page-internal retrieval metrics
```

Supported parser values are `mineru` and `ppstructure`. PP-StructureV3 uses
`parsers/_ppstructure_worker.py` as a subprocess entry and requires
`PPOCR_PYTHON`.

## Environment

The experiments use two conda environments:

```bash
conda env create -f experiment_add/env/advdoc_environment.yml
conda activate advdoc
pip install -r experiment_add/env/advdoc_pip_freeze.txt

conda env create -f experiment_add/env/paddle_environment.yml
conda activate paddle
pip install -r experiment_add/env/paddle_pip_freeze.txt
```

For a shell running experiments:

```bash
conda activate advdoc
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export SENTENCE_TRANSFORMERS_HOME=$HF_HOME
export PPOCR_PYTHON=$(conda run -n paddle which python)
export DEEPSEEK_API_KEY=<your-deepseek-api-key>
```

`DEEPSEEK_API_KEY` is required only for live DeepSeek policy and QA runs.

## Data

This repository does not redistribute PubLayNet, DocLayNet, selected page
images, COCO annotation JSON files, OCR text, parser outputs, or generated
dataset manifests containing document text. Users must obtain the datasets
from their official sources and comply with their original licenses.

Prepare local data under:

```text
data/selected/publaynet/selected_600.json
data/selected/publaynet/images/
data/selected/doclaynet/selected_600.json
data/selected/doclaynet/images/
```

`config/shared_eval_set.json` is an ID-only manifest for selecting the same
local pages. It contains page IDs and selection metadata only; it does not
include images, annotations, bounding boxes, OCR text, or parser blocks.

## Smoke Checks

```bash
python3 experiment_add/shared/parsers/parser_env_preflight.py \
  --config experiment_add/configs/parser.yaml

python3 experiment_add/shared/data/build_page_manifest.py \
  --config experiment_add/configs/base.yaml --debug
```

Small main-experiment run:

```bash
python3 experiment/run_experiment.py \
  --run_mode pilot --n_images 10 --phases 1a --parser mineru --seed 42
```

If local datasets are missing, the data loader stops with a message describing
which files must be prepared locally.

## Main ProSA Metrics

Formal Phase 1:

```bash
python3 experiment/run_experiment.py \
  --run_mode formal --phases 0plus,1a,1b,sensitivity \
  --parser mineru --seed 42 \
  --shared_dataset config/shared_eval_set.json \
  --static_plan config/static_attack_plan.json

python3 experiment/run_experiment.py \
  --run_mode formal --phases 0plus,1a,1b,sensitivity \
  --parser ppstructure --seed 42 \
  --shared_dataset config/shared_eval_set.json \
  --static_plan config/static_attack_plan.json
```

Formal Phase 2 can be run live with DeepSeek or replayed from the committed
logs. Replay is preferred for exact paper reproduction:

```bash
python3 experiment/run_experiment.py \
  --run_mode formal --phases 2 --parser mineru --seed 42 \
  --shared_dataset config/shared_eval_set.json \
  --replay_phase2_log config/phase2_attack_log_mineru.json

python3 experiment/run_experiment.py \
  --run_mode formal --phases 2 --parser ppstructure --seed 42 \
  --shared_dataset config/shared_eval_set.json \
  --replay_phase2_log config/phase2_attack_log_ppstructure.json
```

Extract paper-number metrics after the CSVs exist:

```bash
python3 experiment/extract_paper_numbers.py --output_dir experiment/output
```

Generated outputs, logs, caches, and local environments are excluded from the
code release.

## Downstream QA and Retrieval Metrics

Use `experiment_add/README.md` for the downstream pipeline. It regenerates
local parser outputs, perturbed pages, QA pairs, QA answers, retrieval
corpora/runs, metrics, and audit logs under ignored local directories.

## License and Third-party Materials

The ProSA code authored by the paper authors is released under the MIT License.
This license does not cover third-party datasets, parser projects, model
weights, APIs, or local generated outputs. See `THIRD_PARTY_NOTICES.md`.

## Citation

If you use this code, please cite:

```bibtex
@misc{chen2026documentparsersbreak,
  title         = {How Do Document Parsers Break? Auditing Structural Vulnerability in Document Intelligence},
  author        = {Yue Chen and Yihao Wang and Ziyi Tang and Yongsen Zheng and Keze Wang},
  year          = {2026},
  eprint        = {2605.19309},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2605.19309},
  url           = {https://arxiv.org/abs/2605.19309}
}
```
