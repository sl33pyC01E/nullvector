# Hierarchical organism raster VAE v2

This additive model is the high-capacity successor to the 222 MiB raster VAE
baseline. It keeps the same frozen 45-organism, 48x48 cellular corpus, but uses
separate continuous latent fields for chassis-scale structure and cell-scale
detail. The v1 checkpoint and outputs remain unchanged.

## Architecture

`HierarchicalOrganismRasterVAE` has 35,612,578 trainable parameters and uses:

- a 32x12x12 Gaussian coarse latent for chassis, symmetry, and topology;
- a 16x24x24 Gaussian fine latent for appendages, cellular detail, and palette;
- 128-channel fine and 384-channel coarse residual streams;
- learned PixelShuffle upsampling instead of fixed interpolation;
- FiLM conditioning and channel attention throughout the encoder and decoder;
- a 192-dimensional condition vector derived from family, subtype, role,
  genome traits, and an explicit eight-value identity style vector;
- direct heads for RGBA, occupancy, tissue, material, part ownership, emission,
  eight physiology fields, ten cell-state fields, and eight organ systems with
  explicit none/core/conduit/effector roles.

The source observation remains the strict 74-channel living organism tensor
defined by v1. The added system-role head is supervised from those frozen
fields; it does not alter the authoritative anatomy or physiology assets.

The objective combines occupied-weighted reconstruction, silhouette Dice,
palette and edge preservation, categorical anatomy losses, physiology and
cell-state reconstruction, rare organ-role weighting, family-weighted soft
symmetry, two KL terms with free bits, and coarse occupancy supervision.

## Frozen fit v2

Authoritative output: `outputs/organism_raster_vae_v2/fit_v2`

- source SHA256: `79f7a2c1f30cc11a85f5fd3f57d8f40a7820373b64cfb9a0c2725c27f4f5bff8`
- manifest SHA256: `e162b5e68e22101411622ea5a57c2fc28aad32ac08bb90dd9cfa52a0d3736c13`
- checkpoint SHA256: `3a9673d95a7744e51c18a71249192ce15e840fc57996f1c0c3d67fdb663f69be`
- model-state SHA256: `befd51d3bc983598c527917e84a4eedbdd502e4ce5bfbb385651d937f5955253`
- 2,048 full-corpus BF16 updates, batch size 45
- loss `9.40032768 -> 0.35140005`
- 303.90 seconds training
- 4,992,303,104 bytes peak allocated and 5,349,834,752 bytes peak reserved
- silhouette IoU `0.99882323`
- RGBA MAE `0.00725112`; visible RGB MAE `0.03507374`
- tissue/material/part accuracy `0.99968171 / 0.99955630 / 0.99969137`
- visible physiology MAE `0.05068955`
- visible cell-state MAE `0.05269663`
- system core recall `1.0`
- organ-role member accuracy `1.0`

The reconstruction sheet was visually inspected at native resolution. It
preserves one-pixel appendages, organism palettes, silhouettes, and distinct
core/conduit/effector layouts. The interpolation sheet shows continuous,
non-collapsed transitions between same-family identities. Coarse and fine
mutations produce distinct localized changes while retaining identity.

Fresh CPU validation reloads the checkpoint with `weights_only=True`, verifies
the source and upstream authority, recomputes every metric over all 45 samples,
regenerates all three PNGs byte-for-byte, and verifies fusion and mutation
gates. The frozen fit passes that replay.

`fit_v1` in the same output root is preserved as historical evidence. Its
source predates the explicit organ-role decoder and is not the current v2
authority.

## Claim boundary

This is a high-fidelity continuous representation and neural rasterizer, not a
free generator. It was trained on the same 45 identities it reconstructs, so
near-perfect reconstruction demonstrates capacity and faithful field encoding,
not unseen-organism generalization. It is intentionally not promoted into the
runtime yet.

Increasing autoencoder capacity again is no longer the best next move. The
next model should learn the distribution and dynamics of these latents:

1. a family/role/genome-conditioned diffusion or rectified-flow prior over the
   coarse and fine latent fields for new organisms, fusion, and mutation;
2. a cell-and-bond graph dynamics model for motion, fracture, reconnection,
   projectiles, beams, collision, and polyp consolidation;
3. a neural cellular automaton for fluid diffusion, metabolism, organ damage,
   neural cascades, clotting, scarring, healing, growth, and reproduction;
4. sensory encoders and recurrent policies bound to the physical neural-cell
   graph, so damaged neural structures measurably alter control and behavior.

## Commands

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m forge.organism_raster_vae_v2 smoke `
  --output outputs/organism_raster_vae_v2/NEW_IMMUTABLE_NAME `
  --device cuda --steps 2048

$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.organism_raster_vae_v2 validate `
  outputs/organism_raster_vae_v2/fit_v2
```
