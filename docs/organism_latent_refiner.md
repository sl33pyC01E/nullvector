# Neural organism latent refiner and homeostasis assay

This additive model consumes the frozen hierarchical organism flow prior and
VAE v2. It learns a conditional projection from corrupted coarse/fine latent
fields toward exact or same-family-interpolated organism fields. It never runs
connected-component cleanup, edits raster pixels, or changes the sealed prior.

## Why the model exists

The first prior audit mistakenly treated every disconnected visible component
as an invalid satellite. A full reference audit disproved that assumption:
all 45 authoritative organisms contain 3-39 visible components because the
morphology grammar deliberately includes separated appendage tips, aura cells,
ornaments, and anomaly orbital structures. Reference mean component count is
10.6 and mean non-primary-component share is 4.526%. The generated prior bank
closely matches this at 11.77 and 4.339%.

The refiner therefore has two valid jobs:

1. preserve legitimate multi-component topology on already-valid generated
   organisms;
2. recover cellular, visual, and physiological state after continuous latent
   injury—the neural analogue of homeostasis and a foundation for healing.

## Model

`HierarchicalLatentRefiner` has 4,894,096 parameters and predicts residuals for
the 32x12x12 chassis latent and 16x24x24 cellular latent. Four bidirectionally
coupled cross-scale stages receive the frozen 192-value organism condition and
a continuous corruption-level embedding.

Training targets include exact organism latents and continuous same-family
interpolations. Inputs receive log-uniform correlated corruption from 2.5% to
65%. A frozen VAE decoder adds differentiable supervision for opacity, edges,
RGBA, physiology, and organ roles. The output heads begin near identity, but
with nonzero weights so gradients reach both trunks and the conditioner on the
first update.

Like the prior, training uses fresh-process immutable segments with full
optimizer, EMA, RNG, source, and predecessor lineage. A 32+32 resumed run and
an uninterrupted 64-update run produced identical model, EMA, history, and RNG.

## Frozen refiner v3

Authoritative output: `outputs/organism_latent_refiner/refiner_v3`

- source SHA256: `e3fbbf77eb1ee17a7b4f8fc8964654b35dd431ea72fe5c7d03387d94382ad7e1`
- manifest SHA256: `fcbef4d7acccec12448a0ed97d01b785f1609a5dd0dd4adfa13b5e1f31f06371`
- checkpoint SHA256: `614156e9942901cd39583c24771775e1f5dfc6fcce720a7d5904bf3416ee3e13`
- EMA SHA256: `a77a2476ae1d6c0b4784733309c0e295f2a99eee4f37028069f0eda03a7fad3c`
- 4,096 BF16 updates, batch size 128, eight 512-step processes
- zero retries, crashes, OOMs, or non-finite updates
- loss `0.09461418 -> 0.01941080`
- 2.50 GiB peak reserved VRAM

The 45-organism assay injects 35% correlated injury into both normalized latent
scales and compares clean, corrupted, and refined decodes:

- coarse latent MSE: `0.0700454 -> 0.0022985` (96.7% reduction)
- fine latent MSE: `0.0697944 -> 0.0062878` (91.0% reduction)
- RGBA MAE: `0.0030451 -> 0.0020538` (32.6% reduction)
- alpha MAE: `0.0006339 -> 0.0002957` (53.4% reduction)
- physiology MAE: `0.0011405 -> 0.0007954` (30.3% reduction)
- component-count error: `0.04444 -> 0.02222` (50% reduction)

On the untouched 30-sample prior bank, refinement preserves exact thresholded
component statistics, all organ cores, minimum novelty, and family diversity.
The mean rendered change is only `0.0007607` RGBA L1, as expected for valid
inputs. Every hard and calibrated quality gate passes, and the CPU validator
replays all semantic values and PNGs byte-for-byte.

This model is still a research component rather than an integrated gameplay
system. A future neural cellular automaton can use the same living-cell fields
for local healing, clotting, scarring, fluid transport, and regrowth, while this
global refiner supplies organism-scale priors for severe damage recovery.

## Commands

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m forge.organism_latent_refiner train `
  --output outputs/organism_latent_refiner/NEW_IMMUTABLE_NAME `
  --steps 4096 --segment-steps 512 --batch-size 128 --max-attempts 3

$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.organism_latent_refiner finalize `
  --output outputs/organism_latent_refiner/NEW_IMMUTABLE_NAME

$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.organism_latent_refiner validate `
  outputs/organism_latent_refiner/refiner_v3
```
