# Segmented semantic sprite latent production

This pipeline trains the categorical 48×48 FSQ autoencoder on the frozen 32,768-specimen morphology corpus. It is intentionally separate from the small `sprite_latent` smoke artifacts: those remain immutable representation experiments, while this package owns resumable CUDA production state.

## Authority contract

- Corpus SHA-256: `77dc7313ca6411295bad883f483a6edf4be75016ebfd7c107d0f286d2cb1cd7b`
- Split fingerprint: `5e400872460dc527c01a2a301f006e761abd1621773c5f67b45568d68886007b`
- Train-only legal tuple fingerprint: `0b15074b76ca69ea9a93e0b73db7e5df0b242dc0ecc46c5e842342fb0378948d`
- Native categorical fields: part-owner 17 classes, material 10 classes, emission 4 classes
- Latent geometry: 12×12×6 FSQ digits with levels `8,8,6,5,5,4` (38,400 implicit codes)

The production source hash binds the core codec/corpus/training modules and every module in `forge/sprite_latent_production`. A source edit after training begins makes old checkpoints fail closed.

## Crash and outage containment

The supervisor launches each two-epoch segment in a fresh process, allows at most three attempts, records Windows access violations, and publishes a segment only after the canonical report and immutable checkpoint replay. Checkpoints bind model, EMA, optimizer, RNG, completed history, frozen data provenance, and the exact predecessor checkpoint SHA. Partial calibration epochs are explicitly marked and cannot be resumed into production.

Training uses a production-only flattened categorical loss adapter. It is mathematically equivalent to the core `N,C,H,W` cross-entropy, but routes CUDA NLL through the deterministic `N*H*W,C` kernel. The CPU test suite proves the complete loss and every component match the core loss within `1e-6`; deterministic-algorithm enforcement remains enabled during CUDA work.

Every worker is launched with `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization, deterministic cuDNN, disabled cuDNN benchmarking, and PyTorch deterministic-algorithm enforcement. A worker that reaches a nondeterministic kernel fails instead of silently weakening the replay contract.

The 100 GiB free-disk floor is checked before training and before every segment. The worker also requires at least 4 GiB free CUDA memory. Existing outputs are never overwritten.

## Honest quality verdict

Legal projection is a safety invariant, not a visual-quality claim. A checkpoint is accepted only if all global and worst-family gates pass:

- aligned legal-tuple accuracy ≥ 0.93
- visible tuple accuracy ≥ 0.78
- visible silhouette IoU ≥ 0.90
- worst-family visible tuple accuracy ≥ 0.70
- worst-family silhouette IoU ≥ 0.84
- latent code utilization ≥ 0.015

This prevents a background-only reconstruction from passing on empty-pixel accuracy.

## Commands

```powershell
python -m forge.sprite_latent_production calibrate --steps 100
python -m forge.sprite_latent_production train
python -m forge.sprite_latent_production validate checkpoints/sprite_latent_production_v1/segments/epoch_024/checkpoint.pt
python -m forge.sprite_latent_production validate-manifest checkpoints/sprite_latent_production_v1/production_manifest.json
```

Do not promote a checkpoint merely because training was finite. `production_manifest.json` must have `status: ready` and `gates.full_quality_accepted: true`.

## Accepted run and visual evidence

The additive `checkpoints/sprite_latent_production_v1_1` run completed all 24
epochs in 12 immutable two-epoch workers. Epoch 24 is the accepted best
checkpoint. A balanced held-out visual proof can be built and replayed without
changing the frozen trainer source hash:

```powershell
python -m forge.sprite_latent_showcase build
python -m forge.sprite_latent_showcase validate `
  outputs/sprite_latent_production_showcase_v1/showcase_manifest.json
```

The showcase contains eight validation specimens per family, original and EMA
reconstruction PNG pairs, and the exact projected categorical fields. It binds
the production manifest, checkpoint, EMA state, corpus, split, legal tuple
table, selection, per-sample metrics, and all artifact bytes.
