# Map Decorator Sparse Locator v3

Status: bounded CUDA calibration complete; quality failed. It is not a production decorator,
a visual-quality claim, or a Godot-integrable asset source.

## Why v3 exists

The v2 foreground-factored model proved that empty-head collapse could be reversed, but its
single presence probability field had two incompatible jobs: rank object locations and make
the sum of thousands of probabilities equal the object count. Bounded calibration recovered
nonempty decals and props, yet spatial IoU/F1 remained weak and foreground density was high.

V3 separates those jobs:

- a contextual spatial tower consumes the authoritative 53-channel map tensor plus the raw
  categorical object logits and ranks legal cells;
- a separate pooled head predicts `log1p(object_count)` for each object head;
- the decoder rounds that independent count into a bounded quota, then uses stable legal
  top-k selection;
- decal/prop collisions are resolved by spatial score and deterministically backfilled where
  mutually exclusive legal cells exist;
- exact foreground cells, a bounded low-probability halo, and explicit positive-versus-hard-
  negative ranking all supervise localization;
- count loss uses the complete eligible crop target and is independent of corruption masks or
  the sum of presence probabilities.

This preserves topology-v2 semantics, hard-empty masks, object legality, and full-map split
authority. It neither edits nor migrates the immutable 3,096-map corpus or its recovered
216-shard foreground index.

## Current evidence

Focused tests cover:

- canonical contract hashing and invalid configuration rejection;
- exact tensor shapes and initialized count priors;
- counts that remain exact even when every presence probability is nearly zero;
- stable tie ordering, legal-count clamping, collision exclusion, and quota backfill;
- non-finite count rejection;
- real topology-v2 foreground crops with gradients reaching both locator and count heads;
- CUDA fail-closed behavior;
- deterministic two-run CPU smoke equality and resealed-report tamper rejection.
- bounded, atomic v3 checkpoints with complete dependency-source, corpus, index, model,
  optimizer, EMA, generator, and CPU RNG provenance;
- exact interrupted-versus-uninterrupted training equivalence plus rejection of fully rehashed
  sidecar and tensor tampering.
- a CUDA-cold real-corpus pilot that resolves the immutable 3,096-map corpus and recovered
  foreground index, trains on deterministic foreground-centered crops, evaluates bounded
  validation and test samples through both raw and EMA models, enforces exact per-head legality,
  and publishes a fully reloadable source/corpus/index-bound checkpoint;
- semantic exact replay of that pilot independent of checkpoint-container byte encoding.
- a source-frozen BF16 CUDA worker plus three-attempt supervisor that records Windows native
  exits, enforces the 100 GiB floor, evaluates every validation/test map through both raw and
  EMA weights, reloads the complete checkpoint/RNG boundary, and publishes quality failures
  as evidence without authorizing integration.

`hard_empty` is an object/effect exclusion, not a terrain-variation exclusion. The pilot
therefore requires decal, prop, and emission to remain zero there while checking `variant`
against its authoritative class legality mask. This distinction is explicit because silently
zeroing variant would erase valid terrain detail.

Run the focused suite without touching CUDA:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
C:\Users\forre\AppData\Local\Programs\Python\Python312\python.exe -m pytest `
  tests\test_map_decorator_production_v3.py -q
```

Create and validate an additive smoke artifact:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.map_decorator_production_v3 smoke `
  --output outputs/map_decorator_production_v3/cpu_smoke_g
python -m forge.map_decorator_production_v3 validate `
  outputs/map_decorator_production_v3/cpu_smoke_g/smoke_report.json `
  --exact-replay
```

Run the bounded real-corpus CPU pilot without touching CUDA:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.map_decorator_production_v3 pilot `
  --corpus outputs/map_decorator_corpus_v1 `
  --index outputs/map_decorator_production_v2/foreground_index_v2 `
  --output outputs/map_decorator_production_v3/real_corpus_cpu_pilot `
  --steps 4 --eval-samples 4
python -m forge.map_decorator_production_v3 validate-pilot `
  outputs/map_decorator_production_v3/real_corpus_cpu_pilot/pilot_report.json `
  --corpus outputs/map_decorator_corpus_v1 `
  --index outputs/map_decorator_production_v2/foreground_index_v2 `
  --exact-replay
```

The pilot is an integration and safety proof, not a visual-quality claim. Four CPU updates are
deliberately insufficient to accept or reject the sparse locator's eventual map quality.

## Full 100-step calibration result

The immutable supervised run is
`outputs/map_decorator_production_v3/cuda_calibration_100step_v1`. It completed on its first
worker attempt despite unrelated GPU contention. Training took `137.7943 s` at `0.7257`
updates/s, full raw-plus-EMA evaluation took `306.3592 s`, peak reserved VRAM was only
`0.633 GiB`, and all hard legality/provenance/checkpoint gates passed.

Quality did not pass. EMA validation decal IoU/F1/rare recall was
`0.003569 / 0.007339 / 0.002809`; prop was
`0.001251 / 0.004430 / 0.0`. On the untouched 24-map test split, EMA decal was
`0.000592 / 0.001213 / 0.000160`, while prop was exactly zero IoU/F1/rare recall. Sparse count
density also under-shot the test split (`0.0307x` decal, `0.0553x` prop). The raw model was
similarly weak, so EMA lag is not the explanation. No v3 checkpoint is integration-eligible.

Evidence identities:

- supervisor report: `d7578644232ccaf894375a964a5f6bbbfd7b919ceb1ff1d29a7fd3e069c38b26`;
- calibration report: `9f68995434ecb35c3edb05e8ba9d16a93f18d40a407e69f9b223805d143d6064`;
- semantic calibration: `745a7951ba1178e15acb20b668935400e30ac1caf1e43fc2a4979aa76ef3e6a7`;
- checkpoint source: `1850b6644a2b4e5242ab613428d0848cb348e4a7a7c9d7525b21ba6646c67ba0`;
- model tensors: `afb61a70c248fc4e906c17efd2f65d3d55b66607cf9da26fd2715d72cdaca63f`;
- EMA tensors: `57551350edf9b6759e50f33994cba955cc5c7408959f6d166fdff5be3d7da8d5`.

Validate the sealed evidence without touching CUDA:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.map_decorator_production_v3 validate-calibration `
  --output outputs/map_decorator_production_v3/cuda_calibration_100step_v1 `
  --corpus outputs/map_decorator_corpus_v1 `
  --index outputs/map_decorator_production_v2/foreground_index_v2
```

## Next authorized boundary

Freeze v3 as failed evidence. A v4 model should supervise object coordinates or coarse spatial
heatmaps directly, use cross-theme hard-negative mining, and calibrate counts separately from
localization. It must start from fresh weights under a new source/contract hash and repeat the
same unchanged validation/test gates. Longer training and runtime integration remain forbidden
until both decal and prop pass every predeclared quality and density gate.
