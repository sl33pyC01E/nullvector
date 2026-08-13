# `forge.multifield_eval`

Production-grade evaluation, immutable generation banks, bounded compilation,
visual QA, benchmarking, and exact replay for
`MultiFieldSpriteDiffusion` checkpoints.

Validation v2 separates hard field/topology safety from diagnostic nearest-
centroid condition adherence. Run `calibrate` first; the authoritative held-out
partition must be 100% hard-valid before evaluating neural outputs. Guide,
source target, family-local condition IDs, and the legal-tuple table are also
validated as fail-closed evaluator inputs.

```text
python -m forge.multifield_eval calibrate --corpus CORPUS [options]
python -m forge.multifield_eval snapshot SOURCE DESTINATION
python -m forge.multifield_eval status --checkpoint SNAPSHOT [--corpus CORPUS]
python -m forge.multifield_eval sample --checkpoint SNAPSHOT [options]
python -m forge.multifield_eval benchmark --checkpoint SNAPSHOT [options]
python -m forge.multifield_eval replay GENERATION_MANIFEST [options]
```

Use `python -m forge.multifield_eval COMMAND --help` for every flag. Evaluation
of a live run begins with `snapshot`; output checkpoints and generation banks
are immutable and are never overwritten.

The detailed contract and operational commands are in
[`docs/multifield_evaluation.md`](../../docs/multifield_evaluation.md).

Published JSON contracts:

- [`multifield_generation_bank.schema.json`](../../shared/schema/multifield_generation_bank.schema.json)
- [`multifield_raw_sample.schema.json`](../../shared/schema/multifield_raw_sample.schema.json)
- [`multifield_benchmark.schema.json`](../../shared/schema/multifield_benchmark.schema.json)
- [`multifield_reference_calibration.schema.json`](../../shared/schema/multifield_reference_calibration.schema.json)
