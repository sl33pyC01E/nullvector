# Conditional hierarchical organism rectified flow

This subsystem supplies the generative prior that the continuous organism VAE
intentionally lacked. It learns a conditional velocity field from Gaussian
noise to the frozen v2 VAE's coarse chassis and fine cellular latent pyramids.
It is additive: the anatomy, physiology, trauma, raster VAE v1/v2, and their
checkpoints remain immutable.

## Model and training

`HierarchicalOrganismFlow` has 13,661,120 trainable parameters. It couples:

- a 32x12x12 coarse stream for chassis, symmetry, and global topology;
- a 16x24x24 fine stream for appendages, individual cells, and palette;
- six cross-scale residual/attention stages with bidirectional coarse/fine
  exchange;
- continuous timestep embeddings and the frozen 192-value VAE condition;
- classifier-free condition dropout and guided midpoint-time Euler sampling;
- posterior perturbation so the prior learns a distribution rather than a
  table of 45 means.

The latent corpus is regenerated from the exact frozen VAE checkpoint and
stores means, log variances, conditions, normalization statistics, identities,
and per-tensor hashes. It contains 45 organisms with family census
11/10/9/8/7. Any upstream, tensor, source, or checkpoint drift fails closed.

Training is split into fresh-process segments. Every immutable segment stores
the full model, EMA, optimizer, CUDA RNG, history, source tree, and latent-corpus
identity. Resume was tested by comparing a 32+32 chain against an uninterrupted
64-update run; model, EMA, history, and RNG were identical.

## Frozen prior v1

Authoritative output: `outputs/organism_latent_flow/prior_v1`

- source SHA256: `27f2178f2090f70d8cbcc0be4c4a94fc7f12134f7af00479ece16e1d0515b9b9`
- manifest SHA256: `f524104372db57cabb4fe4d2fe91b6a72da9f7f39481fa1b8038d38cb4a3854e`
- final checkpoint SHA256: `5b0a8f24e71fbad496c3a4a5f38453eedd501b941529ff491c4d9f27568778f1`
- final EMA SHA256: `3ed1242914f7118f81624a4fd203f7be9296a4b6eaa146390bd9e9ce21baa8f6`
- latent-corpus semantic SHA256: `f7b5c0a9f480613d47e3bdd89764e5308000e2a161b30c19518974d147307862`
- 8,192 BF16 updates in sixteen 512-step processes, batch size 90
- zero failed attempts, retries, non-finite updates, or OOMs
- loss `4.33192587 -> 0.21707524`; last-512 mean `0.21170620`
- 3.06 GiB peak reserved VRAM in the final segment
- roughly 10 minutes total supervisor time

The frozen CPU generation bank contains 30 samples: five families by six
samples. Columns 1-3 vary stochastic noise. Columns 4-6 hold noise fixed while
traversing between same-family parent conditions. It passes byte-exact CPU
generation and PNG replay.

Diagnostics:

- finite generation fraction: `1.0`
- occupancy ratio range: `0.17969 .. 0.31337`
- mean within-family pairwise RGBA L1: `0.07139`
- mean nearest-training RGBA L1: `0.00157`
- generated samples containing decoded organ cores: `1.0`
- family mutation maximum RGBA L1: `0.00433 .. 0.00806`
- strict single-component fraction: `0.0`; maximum raw component count: `39`

The images were visually inspected at native scale. Main silhouettes are
recognizable, diverse, family-conditioned, and far cleaner than the 64-step
calibration. Internal physiological networks remain coherent. Small isolated
alpha islands still appear around otherwise coherent bodies, however. The
strict connectivity quality gate therefore fails. This v1 is a valid trained
prior and research baseline, but it is not approved for runtime promotion.
No deterministic connected-component cleanup is applied or implied.

## Next neural correction

The next aligned stage is a learned latent-manifold refiner. It should denoise
flow endpoints toward valid exact and same-family interpolated latent fields,
with frozen-decoder supervision for occupancy edges and organ continuity. That
keeps cleanup neural, preserves continuous breeding, and directly targets the
satellite-cell failure instead of concealing it after rasterization.

## Commands

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m forge.organism_latent_flow train `
  --output outputs/organism_latent_flow/NEW_IMMUTABLE_NAME `
  --steps 8192 --segment-steps 512 --batch-size 90 --max-attempts 3

$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.organism_latent_flow finalize `
  --output outputs/organism_latent_flow/NEW_IMMUTABLE_NAME

$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.organism_latent_flow validate `
  outputs/organism_latent_flow/prior_v1
```
