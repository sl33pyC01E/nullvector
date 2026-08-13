# Multi-field production evaluation and sampling

`forge.multifield_eval` is the immutable evaluation path for the role-conditioned
v2 morphology model. It consumes the EMA model only after a trainer checkpoint
has been atomically published. Temporary files are never candidates for loading.

## Calibrate hard validity without a checkpoint

The validator first has to accept the authoritative data contract it is meant
to police. This CPU-only command validates the complete held-out partition and
writes an immutable calibration report without loading a model or occupying the
GPU:

```powershell
python -m forge.multifield_eval calibrate `
  --corpus data/morphology_32768_4d4f5250.npz `
  --validation-fraction 0.08 --split-seed 0x5A17 `
  --output outputs/multifield_calibration/reference_validation_v2_77dc7313_5e400872_0cc21669.json
```

The report must show `hard_valid == samples`, `hard_valid_rate == 1.0`, and no
hard-gate failures. It also records every gate's pass count, owner-presence
rates, occupancy bounds, minimum scaffold coverage, visible component counts,
and the deliberately diagnostic-only condition-template scores. The same full
calibration report is embedded in every checkpoint benchmark.

The authoritative 32,768-specimen corpus currently proves the v2 contract on
all 2,560 held-out rows: 2,560/2,560 hard-valid, no gate failures, and exactly
one visible component in every row. Core ownership is present in 2,560/2,560;
body ownership in 2,543/2,560; head ownership in 2,517/2,560. Minimum scaffold
coverage is `0.979112`; occupancy spans `0.111111` through `0.341146`. The
diagnostic nearest-centroid exact-match rates are `95.3125%` morphology,
`36.171875%` subtype, `86.210938%` role, and `30.273438%` jointly, directly
showing why those measurements cannot be hard validity gates. The immutable
evidence is
`outputs/multifield_calibration/reference_validation_v2_77dc7313_5e400872_0cc21669.json`.
Its evaluation-source hash is
`0cc21669c837a73007f89148ecc6eabd2027a6dc5f9309f2beee1515d29837b8`.

## Snapshot a live run first

Do not benchmark a pathname that a live trainer may replace between epochs.
Capture a stable, immutable snapshot and evaluate the snapshot:

```powershell
python -m forge.multifield_eval snapshot `
  checkpoints/multifield_production_v2/best.pt `
  checkpoints/evaluation_snapshots/production_epoch_004_best.pt

python -m forge.multifield_eval status `
  --checkpoint checkpoints/evaluation_snapshots/production_epoch_004_best.pt `
  --device cpu --precision fp32
```

The snapshot command opens only a published non-temporary file, verifies its
identity and byte count before and after the copy, flushes a temporary
destination, and publishes with `os.replace`. It never overwrites an existing
snapshot and checks the 100 GiB free-space floor. If no checkpoint has completed
an epoch yet, commands return structured JSON with
`"status": "checkpoint_incomplete"` instead of attempting to deserialize an
unpublished file.

Every load then verifies all of the following before constructing the model:

- checkpoint format and completed epoch/global-step markers;
- architecture and all active categorical vocabularies;
- canonical EMA tensor hash;
- active training-source hash;
- complete corpus SHA-256 and morphology-renderer source provenance;
- exact train/validation split fingerprint;
- train-only legal tuple table and fingerprint;
- versioned anti-leak guide policy reconstructed from the training config.

Changing any one of those values rejects the checkpoint. There is no
`allow-source-change` inference escape hatch.

## Immutable sample banks

The 40-cell stratified bank is five family rows by eight role columns. Cycling
the four family-local subtypes across the role columns covers all five families,
all eight roles, all twenty subtypes, and all forty family/role pairs:

```powershell
python -m forge.multifield_eval sample `
  --checkpoint checkpoints/evaluation_snapshots/production_epoch_004_best.pt `
  --device cuda --precision bf16 `
  --grid stratified --samples-per-condition 2 --batch-size 8 `
  --output-dir outputs/multifield_generation/production_epoch_004
```

Other grids are:

- `fixed`: the exact source bank and base seed recorded by the trainer;
- `stratified`: 40 family/role cells with complete subtype coverage;
- `exhaustive`: every legal subtype crossed with every role, 160 cells before
  repetitions.

Each sample owns an independent device-correct `torch.Generator`. Its unsigned
63-bit seed is a stable hash of the checkpoint seed, grid, condition, source
row, and variation, so changing batch composition cannot merge RNG streams.
Batch size, device type, precision, temperature, runtime versions, and all
seeds are recorded for exact replay.

One through eight variations per condition are accepted. For unusually large
banks, global pairwise diversity uses a fixed-seed, uniform, without-replacement
sample capped at 100,000 pairs and records that policy; all within-condition
pairs are still measured exactly.

The destination must be empty. Existing banks and raw outputs are never
overwritten. For each specimen, publication order is deliberately:

1. `raw/<id>.npz`: aligned neural part/material/emission fields, sanitized
   guide, genes, source targets, conditions, and seeds;
2. `raw/<id>_rgba.png` and `raw/<id>_emission.png`;
3. `raw/<id>.json`: raw hashes, full provenance, and validation;
4. only then, optional bounded compilation under `compiled/`.

The raw categorical output is authoritative. Compilation may only remove tiny
detached speckles by replacing their complete tuple with `(background, void,
off)`. It cannot add a pixel, rewrite a surviving tuple, or exceed the recorded
default 3% pixel delta. Both raw and compiled contact sheets are generated for
nearest-neighbor visual QA.

## Validation contract

Raw validation v2 is intentionally stricter than "the PNG opened":

- exact categorical field, source-target, float32 guide, condition-id, and
  train-only legal-table contracts. Malformed evaluator inputs fail closed;
- shape, dtype, vocabulary bounds, and exact aligned-field hashes;
- train-observed `(part, material, emission)` tuple validity;
- a three-pixel safety margin for structural owner categories and a two-pixel
  margin for the complete visible union;
- exactly one connected component in the complete `part_owner != background`
  union, occupancy in `[0.02, 0.60]`, and graph-scaffold coverage of at least
  `0.45` at radius two;
- a required `core` owner category. `body` and `head` presence remain reported
  but are not hard gates: the single owner field legitimately lets joint,
  terminal, ornament, and other semantic categories overwrite those labels,
  and a small fraction of authoritative specimens therefore lacks an explicit
  body or head owner while retaining a valid connected silhouette;
- target-source silhouette and field similarity diagnostics;
- family, family-local subtype, and family-conditioned role adherence using a
  deterministic template bank built only from the training split. Nearest-
  centroid exact matches and in-distribution thresholds are diagnostics, never
  validity gates;
- exact uniqueness, pairwise silhouette IoU, pairwise categorical Hamming
  diversity, and within-condition diversity when repetitions are requested;
- aggregate and per-family, per-role, and per-subtype acceptance rates;
- independently validated postprocess hashes and delta bounds.

`accepted` is exactly the hard contract/topology result. It is independent of
condition-template classification. This prevents a lossy diagnostic classifier
from rejecting a safe, connected, contract-valid neural sprite while preserving
separate exact morphology, subtype, and role match rates for model evaluation.

Matched epoch-20 evidence confirms this is a validator correction, not changed
model output. The original immutable CPU five-sample bank accepted `1/5`; the
calibrated bank accepts `5/5`. Both banks use checkpoint SHA-256 `5937a855...`,
EMA hash `d672fefa...`, identical conditions, and identical raw field hashes.
Diagnostic adherence is unchanged: morphology `5/5`, subtype `1/5`, role `4/5`,
all three exact jointly `1/5`, and all axes in-distribution `5/5`. The new bank
is
`outputs/multifield_generation/milestone_epoch020_cpu5_calibrated_final`;
exact replay passes `5/5` raw and compiled specimens. The final manifest SHA-256
is `4d8d5f4e32c4d3f8c1aa4255edaf9325059bdaeb2513e5f94bb772d3fe6a3122`.

Legal-tuple-constrained generation should always report tuple validity `1.0`.
Full-mask metrics remain unconstrained and reveal whether the three neural
heads learned tuple compatibility without the sampler's hard table.

## Exact replay

Replay defaults to the device type and precision recorded by the bank:

```powershell
python -m forge.multifield_eval replay `
  outputs/multifield_generation/production_epoch_004/generation_manifest.json `
  --report outputs/multifield_generation/production_epoch_004/replay_report.json
```

Replay rejects a changed checkpoint, EMA, corpus, source tree, legal table,
guide policy, Torch/CUDA/cuDNN runtime, GPU identity, device type, or precision.
It regenerates every raw field with the recorded batching and per-sample seeds,
then requires exact equality for raw fields, guides, genes, source targets,
RGBA, emission, bounded compiled fields, compiled RGBA/emission, artifact byte
sizes, and artifact SHA-256 values. The report is `"exact"` only when every
sample passes every comparison.

## Checkpoint benchmark

```powershell
python -m forge.multifield_eval benchmark `
  --checkpoint checkpoints/evaluation_snapshots/production_epoch_004_best.pt `
  --device cuda --precision bf16 `
  --grid stratified --samples-per-condition 2 `
  --full-mask-examples 256 --full-mask-batch-size 32 `
  --generation-batch-size 8 `
  --output outputs/multifield_benchmarks/production_epoch_004.json
```

The benchmark records the complete held-out ground-truth calibration, full-mask
loss, silhouette IoU, per-field macro IoU and
foreground accuracy, unconstrained joint-tuple validity, condition-preference
rates, raw reverse-diffusion validity, pairwise diversity, per-family/role/
subtype acceptance, throughput, batch latency, seconds per sample, and CUDA
peak allocated/reserved VRAM for both full-mask and generation phases.

The evaluator tests use a one-step, width-32 current-contract smoke checkpoint
on CPU. They cover incomplete/corrupt publication handling, immutable snapshot
copying, strict provenance rejection, all three condition grids, raw-before-
compiled artifact ordering, bounded deltas, exact bank replay, visual contact
sheets, invalid synthetic rejection, 100% held-out hard-valid calibration, and
the complete benchmark report surface.
