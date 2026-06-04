# Main ProSA Metrics Reproduction

`experiment/` contains the main ProSA experiment runner, perturbation engine,
policies, parser pool, and metric code used for the paper numbers. Exploratory
analysis and figure-generation code has been removed; this directory now emits
only metric CSVs and `paper_numbers.json`.

Supported parsers:

- `mineru`
- `ppstructure`

## Entry Point

```bash
python3 experiment/run_experiment.py
```

Important options:

| Option | Default | Meaning |
|---|---|---|
| `--run_mode` | `formal` | `pilot` or `formal` |
| `--n_images` | mode default | Override sample count |
| `--seed` | `42` | Random seed |
| `--phases` | `0,0plus,1a,1b,2` | Comma-separated phase list |
| `--parser` | `mineru` | `mineru` or `ppstructure` |
| `--output_dir` | `experiment/output` | Output directory |
| `--shared_dataset` | none | Frozen shared image index |
| `--static_plan` | none | Frozen Phase 1 attack plan |
| `--replay_phase2_log` | none | Replay exported Phase 2 decisions |
| `--deepseek_api_key` | env var | DeepSeek key for live Phase 2 policy runs |

## Environment

From the repository root:

```bash
conda activate advdoc
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export SENTENCE_TRANSFORMERS_HOME=$HF_HOME
export PPOCR_PYTHON=$(conda run -n paddle which python)
```

`PPOCR_PYTHON` is required for `--parser ppstructure`; it points to the Python
executable in the PaddleOCR environment. DeepSeek is required only for live
LLM-based Phase 2 policies:

```bash
export DEEPSEEK_API_KEY=<your-deepseek-api-key>
```

## Smoke Run

```bash
python3 experiment/run_experiment.py \
  --run_mode pilot --n_images 10 --phases 1a --parser mineru --seed 42
```

For PP-StructureV3:

```bash
python3 experiment/run_experiment.py \
  --run_mode pilot --n_images 10 --phases 1a --parser ppstructure --seed 42
```

## Formal Reproduction

Phase 1 uses the frozen shared dataset and static attack plan:

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

Phase 2 replay uses committed decisions and avoids DeepSeek drift:

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

Live Phase 2 is still available by replacing `--replay_phase2_log ...` with:

```bash
--deepseek_api_key "$DEEPSEEK_API_KEY"
```

## Outputs

Main files generated under `experiment/output/`:

```text
phase0plus_global.csv
phase0plus_global_ppstructure.csv
phase1a_anchors.csv
phase1a_anchors_ppstructure.csv
phase1b_sweeps.csv
phase1b_sweeps_ppstructure.csv
phase2_policies.csv
phase2_policies_ppstructure.csv
sensitivity.csv
sensitivity_ppstructure.csv
```

Generate the paper-number sidecar from the Phase 1A CSVs:

```bash
python3 experiment/extract_paper_numbers.py --output_dir experiment/output
```

This writes:

```text
experiment/output/paper_numbers.json
```

## Troubleshooting

- `mineru` import error: verify the `advdoc` environment and pip freeze.
- `paddleocr` import error: verify `PPOCR_PYTHON` points to the `paddle`
  environment Python.
- HuggingFace download stalls: set `HF_ENDPOINT` and `HF_HOME` before running.
- CUDA memory errors: run one parser at a time and keep `parser_workers=1`.
