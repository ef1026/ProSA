# Parser Output Schema

This schema documents the parser output contract expected by `experiment_add`.
It mirrors the original `parsers.base_parser.ParseResult` shape and should stay
compatible with MinerU and PP-Structure outputs.

## Top-Level Object

```yaml
parser_name: string
image_id: string
condition: clean | area_matched_erasure | structural_probe | large_area_erasure
source_image_path: string
parse:
  text: string
  layout: list[LayoutBlock]
  tables: list[object]
  reading_order: list[int]
  raw_output: object
metadata:
  parser_version: string | null
  created_at: string
  error: string | null
```

## LayoutBlock

```yaml
bbox: [float, float, float, float]
type: string
text: string
page_idx: integer | null
confidence: float | null
```

Rules:

- `bbox` uses pixel coordinates in `xyxy` order: `[x1, y1, x2, y2]`.
- `type` should preserve the parser layout type when available.
- `text` should be block-level text, not full-page text.
- `layout` and `raw_output.content_list` should be equivalent at the block level when possible.

## Raw Output Expectations

`raw_output` should preserve the original parser details needed by metrics:

```yaml
content_list:
  - bbox: [float, float, float, float]
    type: string
    text: string
middle_json:
  pdf_info:
    - page_idx: integer
      para_blocks:
        - bbox: [float, float, float, float]
          type: string
          lines:
            - spans:
                - bbox: [float, float, float, float]
                  content: string
                  type: string
```

## Compatibility Notes

- MinerU runs in process through `MinerUParser`.
- PP-Structure runs through `PPStructureParser` and its Paddle worker subprocess.
- Consumers should not assume a top-level `blocks` field; use `parse.layout` or
  `parse.raw_output.content_list`.

## Canonical Storage Paths

Clean parse artifacts:

```text
experiment_add/outputs/shared/clean_parse/mineru/
experiment_add/outputs/shared/clean_parse/ppstructure/
```

Perturbed parse artifacts:

```text
experiment_add/outputs/shared/perturbed_parse/mineru/
experiment_add/outputs/shared/perturbed_parse/ppstructure/
```

Parser-level metrics belong under:

```text
experiment_add/outputs/shared/parser_metrics/
```
