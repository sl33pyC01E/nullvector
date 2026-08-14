# Map Decorator Sparse Locator v3

Status: CPU-only architecture foundation. It is not a trained decorator, a visual-quality
claim, or a Godot-integrable asset source.

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

## Next authorized boundary

Do not reuse a v2 checkpoint: the locator/count parameter contract is intentionally different.
The v3 checkpoint/resume boundary now exists and is CPU-proven. The next implementation slice
is a fresh 100-step calibration runner with immutable process supervision and full validation/
test evaluation. Run it only in a free GPU window, keep the unchanged v2 quality/density gates,
and compare raw plus EMA full-split object metrics. Longer training and runtime integration
remain forbidden until both decal and prop pass every predeclared quality and density gate.
