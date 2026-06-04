# Data Placeholder

This repository does not redistribute raw datasets, selected page images, COCO
annotation JSON files, OCR text, parser outputs, or generated manifests
containing document text.

Prepare local data under this directory only after obtaining datasets from
their official sources and confirming that your use complies with their
licenses:

```text
data/selected/publaynet/selected_600.json
data/selected/publaynet/images/
data/selected/doclaynet/selected_600.json
data/selected/doclaynet/images/
```

The tracked `config/shared_eval_set.json` file is an ID-only evaluation
manifest. It selects local files after you prepare the datasets; it does not
include images, annotations, OCR text, or parser blocks.
