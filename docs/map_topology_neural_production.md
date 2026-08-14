# Neural Map Topology Codec Calibration

Status: accepted representation milestone. This is a discrete map codec, not a
map generator, latent prior, compiled map bank, or Godot runtime integration.

## Purpose

`forge.map_topology_neural_production` trains the categorical topology codec on
the frozen 3,096-map topology-v2 corpus. It reconstructs only terrain, hazard,
and elevation. Walkability is derived from terrain for evaluation; navigation,
zones, authoritative topology masks, mission points, and runtime packs remain
under deterministic authority.

The production reader verifies the complete corpus and exposes 2,496 training,
576 validation, and 24 test-sentinel maps in eight homogeneous shape buckets.
The evaluation census uses 48 stratified validation maps and all 24 sentinels.

## Frozen contract

The bounded calibration uses:

- a fully convolutional scale-four VQ codec;
- width 64, latent width 64, residual depth 2;
- 512 EMA codebook entries with eight-dimensional field embeddings;
- categorical class-balanced terrain, hazard, and elevation losses;
- BF16 CUDA forward/backward with float32 loss and deterministic algorithms;
- at most eight maps and 131,072 cells per update;
- model EMA `0.995`, codebook EMA `0.99`, gradient clip `1.0`;
- a fixed seed and a maximum of 500 calibration updates.

The checkpoint contains the raw and EMA model, optimizer, training generator,
CPU/CUDA RNG states, complete history, and evaluation. Loading is CPU-only,
`weights_only=True`, size-bounded, source/corpus/tensor-contract bound, and
checked against a canonical sidecar. Publication uses a unique staging
directory, atomic directory replacement, and a 100 GiB disk guard both before
and after the run.

The report validator requires exact top-level, safety, runtime, checkpoint, and
claim-boundary key censuses. The claim boundary is derived from the quality
result and cannot be changed by merely recomputing the report hash. It reloads
the checkpoint, compares history/evaluation/model hashes, derives the initial
model hash, and can rerun every metric exactly on CUDA.

## Accepted calibration

The accepted immutable artifact is:

`outputs/map_topology_neural_production/calibration_500step_v2_hardened`

Authority and identity:

- production source SHA-256:
  `1fe97d977aaf0a21e2caa6c75a52ee9a0e519087b8c2c2c1dca7e86806253a50`
- frozen corpus SHA-256:
  `16ed5f3b1a661e2bfc2abe9e16c39e9b8caaecba81f50fed6658cc4f73cffab8`
- tensor contract SHA-256:
  `f0fb16a48fedb94e6cb90abd8f66f997da874bc1938f016b36954f7da5fd6f45`
- report SHA-256:
  `63e2070d5164c5b489e7c9fe3ae751b2a415b0b06a9fd7965880a676810e2d73`
- checkpoint file SHA-256:
  `536d7e54e1da9f35ca9200353774121a59da69d9ea12853a5271b89fe06bce64`
- raw model / EMA state SHA-256:
  `75434dd73d777f0b6253c9d39aa229113b88f67b5928eef95e36d983379bee84` /
  `0d90d210505fbda8fa3a319cc3a6d55ca1094252781159c4447f51bac6121d72`

EMA held-out results:

| split | terrain accuracy | terrain macro IoU | hazard macro recall | elevation accuracy | elevation macro IoU | walkability IoU | code use |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation (48) | 0.775510 | 0.381289 | 0.218114 | 0.559691 | 0.274113 | 0.830467 | 0.148438 |
| test sentinels (24) | 0.806554 | 0.402533 | 0.246821 | 0.612762 | 0.305575 | 0.852473 | 0.142578 |

Loss improved 54.109% between the frozen first/last windows. Every frozen
quality gate passes on both splits, as do all provenance and safety gates. A
fresh checkpoint reload reproduced the complete evaluation exactly. A prior
run under the pre-hardening report source produced byte-identical model and EMA
state hashes and an identical evaluation, demonstrating deterministic training;
it is retained as historical evidence and intentionally fails current source
provenance.

## Commands

Run the accepted schedule only into a new path:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
python -m forge.map_topology_neural_production calibrate `
  --corpus outputs/map_decorator_corpus_v1 `
  --output outputs/map_topology_neural_production/UNIQUE `
  --steps 500 --validation-samples 48 --test-samples 24
```

Exact replay:

```powershell
python -m forge.map_topology_neural_production validate `
  --corpus outputs/map_decorator_corpus_v1 `
  --output outputs/map_topology_neural_production/calibration_500step_v2_hardened
```

## Next boundary

The next additive stage is a masked latent prior trained against this frozen
codec. It must keep the codec frozen, publish raw latent proposals separately,
measure raw validity and repair cost, and pass proposals through the existing
deterministic compiler before any map pack or runtime integration claim.

