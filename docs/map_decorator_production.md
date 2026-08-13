# Production Neural Map Decorator Slice

Status: implemented production corpus/training machinery; production corpus build and CUDA
training are separate process-supervised operations.

## Frozen corpus contract

The v1 corpus contains 3,072 train/validation full maps in an exact Cartesian balance:

- six map themes;
- eight size/aspect profiles: 32², 48², 72², 128², 64×40, 40×64, 96×56, 56×96;
- four objective buckets containing 1, 3, 6, and 10 objectives;
- sixteen full-map identities per stratum, selected as thirteen train and three validation
  identities by the foundation's canonical full-map hash split.

Twenty-four identity-disjoint sentinels are additional test-split maps: one 32², 72²,
128², and 256² map for every theme. Candidate selection is deterministic, bounded to
768 candidates per homogeneous shard, and fails closed if an exact split quota cannot be
filled.

The corpus is 192 homogeneous sixteen-map main shards and 24 homogeneous single-map
sentinel shards. Each shard persists:

- the 53-channel float32 feature tensor;
- variant, decal, prop, and emission targets;
- every per-class legality mask and the hard-empty mask;
- the complete topology-v2 semantic arrays and required points;
- full-map/sample/feature/target/legality/replay hashes and exact source contracts.

Teacher extraction is semantic-only and does not allocate RGB, hazard-frame, or pixel-art
buffers. Tests prove equivalence with the original renderer-derived teacher across all six
themes.

## Failure containment and publication

Use at most two shard workers. Each worker builds or validates exactly one homogeneous
shard and exits. The supervisor checks the shared 100 GiB free-space floor before every
worker launch and shard publication, permits at most three attempts, and records the exact
Windows return code, native-failure label, stdout/stderr hashes, PID, and elapsed time.

`fields.npz` uses uncompressed NPY members inside a ZIP container. This makes individual
arrays bounded and memory-mappable without inflating the rest of a shard. A shard is staged
as a directory containing `fields.npz` and `shard.json`, then published with one directory
rename. Existing complete shards are replay-validated and reused after a supervisor crash;
they are never overwritten.

The complete corpus is published only after fresh isolated validation reconstructs every
map from seed/config and exactly replays semantic arrays, features, targets, legality,
identity, and split. Aggregation rejects any duplicate full-map/sample identity, any
main/sentinel overlap, any split drift, or any stratum imbalance.

## Commands

Estimate before building:

```powershell
python -m forge.map_decorator_production estimate `
  --output outputs/map_decorator_corpus_v1
```

Build and validate with two isolated workers:

```powershell
python -m forge.map_decorator_production build `
  --output outputs/map_decorator_corpus_v1 `
  --workers 2
```

Fresh full-shard replay validation:

```powershell
python -m forge.map_decorator_production validate `
  --corpus outputs/map_decorator_corpus_v1 `
  --verify-shards
```

## CUDA calibration and training

The training supervisor first runs 100 CUDA BF16 calibration steps and gates finite losses,
held-out safety, sentinel safety, throughput, and peak memory. Production training is split
into immutable fresh-process segments of exactly two epochs. Each segment resumes only from
the exact prior checkpoint, runs held-out and all-sentinel gates, embeds the combined
`map_decorator_ml` plus production source manifest in its checkpoint, and publishes one
immutable segment directory. A failed process is retried at most three times with native
crash telemetry.

```powershell
python -m forge.map_decorator_production.supervisor `
  --corpus outputs/map_decorator_corpus_v1 `
  --output outputs/map_decorator_production_v1 `
  --epochs 12 `
  --batch-size 4 `
  --train-steps-per-epoch 256
```

No ForgeLab or gameplay integration is performed by this slice. The deterministic renderer
remains authoritative until later visual-quality and compiled-art gates pass.

