# Third-party Notices

The MIT License in this repository applies only to ProSA code authored for the
associated paper. It does not grant rights to redistribute third-party data,
parser projects, model weights, APIs, or local generated outputs.

## Datasets

- PubLayNet annotations are provided by IBM under the Community Data License
  Agreement - Permissive - Version 1.0. PubLayNet images are derived from the
  PubMed Central Open Access Subset, and IBM states that it does not own the
  image copyrights. Users must comply with the PMC Open Access Subset Terms of
  Use and the license of each underlying article.
- DocLayNet is provided under the Community Data License Agreement -
  Permissive - Version 1.0.

This repository does not redistribute PubLayNet or DocLayNet images, COCO
annotation JSON files, OCR text, parser outputs, or selected page manifests
containing source document text.

## Parser and Model Dependencies

MinerU, PaddleOCR/PP-StructureV3, VLM/LLM APIs, downloaded model weights, and
their transitive dependencies are governed by their own licenses and terms of
service. Users are responsible for installing and using them under those terms.

## Generated Outputs

Generated parser outputs, QA pairs, retrieval corpora, logs, metrics, caches,
and figures are local reproduction artifacts. They are ignored by Git and are
not part of this source-code license.
