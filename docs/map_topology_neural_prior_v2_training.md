# Segmented topology-prior v2 calibration

This additive package trains the multiscale prior-v2 architecture against the
frozen 3,096-map latent corpus. It does not alter v1 checkpoints or the seeded
v1 generation bank.

## Why segments are authoritative

Each segment is an immutable directory written by one fresh process. Its
checkpoint preserves model, EMA, AdamW, mask-generator, CPU RNG, CUDA RNG,
cumulative history, evaluation, and the exact predecessor checkpoint hash.
The next segment refuses config, architecture-source, trainer-source, or corpus
drift. A CPU adversarial test proves two one-step processes produce the exact
same history, model, EMA, evaluation, and free-generation artifacts as an
uninterrupted two-step process.

## Metrics that cannot be hidden

Evaluation reports all six mask modes independently:

- full (100% of valid latent cells hidden)
- high random
- rectangle
- half plane
- corridor
- coarse islands

`full_mask_accuracy` and `full_mask_loss` are exact aliases of the `full` mode
record and are re-derived during validation. A separate six-theme iterative
free-generation sentinel starts from an entirely masked canvas and records
token accuracy, uniqueness, uncertainty identity, and complete token reveal.

Calibration acceptance is deliberately not production acceptance. Every report
sets `production_promotion_allowed=false`; compilation, per-theme topology and
grammar audits, visual review, and runtime integration remain outside this
milestone.

## Bounded CUDA sequence

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m forge.map_topology_neural_prior_v2_training segment `
  --output outputs/map_topology_neural_prior_v2_training/calibration_24/segment_000004

python -m forge.map_topology_neural_prior_v2_training segment `
  --resume outputs/map_topology_neural_prior_v2_training/calibration_24/segment_000004/checkpoint.pt `
  --output outputs/map_topology_neural_prior_v2_training/calibration_24/segment_000008
```

Continue through step 24 using a new destination each time. Never overwrite or
delete a prior segment. Validate any segment independently with:

```powershell
python -m forge.map_topology_neural_prior_v2_training validate PATH_TO_SEGMENT
```

## Frozen 24-update calibration (2026-08-14)

The complete six-segment CUDA BF16 chain is published under
`outputs/map_topology_neural_prior_v2_training/calibration_24`. Every segment
loads in a fresh process, binds its predecessor checkpoint, and independently
passes the strict report/checkpoint validator.

- trainer source SHA256: `fe64780bebd0b98e301e04b1638cb9be09dfb634a5da32ac21b12be81f406acf`
- final report SHA256: `b96d19ad673d17662e50b4c6e70e19cc1ce6161d019df8129bd8e0def73c30af`
- final checkpoint SHA256: `8e1c49ca35912a3cab91a84b90c6337574391023f56834667403b66cb98851f2`
- final raw model SHA256: `2e3f40e419d3f866922ee85ebe1f13c15b3bbbcf46078cc3469e8de9232ff46d`
- final EMA model SHA256: `eba71d2d2e8fe0b2e39c12a5d08849b3f667e76af9aa652fe9202b1c0a25e9b5`
- final-step loss: `5.430913925170898`
- raw full-mask accuracy: validation `0.1373626374`, test `0.2191358025`
- EMA full-mask accuracy: validation `0.0062794349`, test `0.0063543936`
- free-generation uniqueness: `6/6`; all latent tokens revealed
- final-segment runtime: 0.849 s training, 0.619 s evaluation, 536 MiB peak
  reserved VRAM

The conservative calibration gate remains false because the deliberately slow
EMA has not crossed its full-mask thresholds. This is the intended result:
the short run demonstrates learning, deterministic resume, bounded memory, and
valid free generation; it is not a topology-quality or production claim.
