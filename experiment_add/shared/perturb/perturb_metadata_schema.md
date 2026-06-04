# Perturbation Metadata Schema

This schema documents metadata for clean and perturbed page artifacts in
`outputs/shared/`. It mirrors the original AdvDoc perturbation concepts without
implementing new perturbation logic.

## Top-Level Object

```yaml
artifact_id: string
image_id: string
condition: clean | area_matched_erasure | structural_probe | large_area_erasure
source_image_path: string
output_image_path: string
mask_path: string | null
random_seed: integer
probe_plan: list[ProbeStep]
metrics:
  TOR: float
  ACR: float | null
  BPO: float | null
  BOC: float | null
  EIR: float | null
metadata:
  created_at: string
  generator: string
  original_config_reference: string | null
  notes: string | null
```

## ProbeStep

```yaml
step: integer
probe_type: string
params: object
target_strategy: anchor | content | random | bridge | global | string
target_location: [integer, integer] | null
target_fallback: boolean | null
reason: string | null
pixels_affected: integer | null
```

## Condition Rules

- `clean` uses the original image and has an empty `probe_plan`.
- `area_matched_erasure` uses the existing P4 erasure family and is matched by
  observed `TOR` during analysis.
- `structural_probe` uses existing structural probe families such as P1, P2,
  P5, P6, or P10.
- `large_area_erasure` uses the existing large P4 erasure family, aligned with
  original configurations such as A08 or A19.

## Compatibility Notes

- `TOR` should be computed with the original `experiment.metrics.compute_tor`.
- `probe_plan` should be compatible with `experiment.engine.AttackEngine.execute`
  `override_plan` steps.
- Perturbed page artifacts should be written under:
  - `experiment_add/outputs/shared/perturbed_pages/area_matched_erasure/`
  - `experiment_add/outputs/shared/perturbed_pages/structural_probe/`
  - `experiment_add/outputs/shared/perturbed_pages/large_area_erasure/`
- Perturbed parse artifacts should be written under:
  - `experiment_add/outputs/shared/perturbed_parse/mineru/`
  - `experiment_add/outputs/shared/perturbed_parse/ppstructure/`
- Parser perturbation metrics should use
  `experiment_add/outputs/shared/parser_metrics/`.
