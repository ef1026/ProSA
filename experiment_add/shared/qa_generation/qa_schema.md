# QA Pair Schema

This schema documents QA pair artifacts shared by experiment 1 and future
experiments. It is a data contract only; no QA generation logic is implemented
in this phase.

## Top-Level Object

```yaml
qa_id: string
image_id: string
source_parser: mineru | ppstructure | string
source_condition: clean
question: string
answer:
  canonical: string
  aliases: list[string]
evidence:
  blocks: list[EvidenceBlock]
  text: string
question_type: factual | table_lookup | layout_reference | string
difficulty: easy | medium | hard | string
metadata:
  created_at: string
  generator: string
  prompt_id: string | null
  notes: string | null
```

## EvidenceBlock

```yaml
block_id: string | integer
bbox: [float, float, float, float]
type: string
text: string
page_idx: integer | null
```

## Rules

- QA pairs should be generated from clean parse artifacts unless an experiment
  explicitly declares another source.
- QA generation/filtering belongs to `shared`, not to `exp1_qa`.
- The canonical shared output file is
  `experiment_add/outputs/shared/qa_pairs/qa_pairs_shared.jsonl`.
- Raw candidates, before filtering, should use
  `experiment_add/outputs/shared/qa_pairs/qa_candidates_raw.jsonl`.
- `answer.canonical` is the primary reference answer.
- `answer.aliases` stores acceptable variants.
- `evidence.blocks` should point back to parser output blocks whenever possible.
- Experiment 1 consumes `qa_pairs_shared.jsonl` and writes QA answering/evaluation
  outputs under `experiment_add/outputs/exp1_qa/qa_runs/` and
  `experiment_add/outputs/exp1_qa/metrics/`.
