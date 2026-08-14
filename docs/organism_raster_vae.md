# Continuous organism raster VAE

This additive subsystem is the first deliberate replacement of a procedural
rendering stage with a neural one. It learns a continuous Gaussian field for
the existing 48px cellular organisms and decodes both visible RGBA and causal
cell state. It does not replace or mutate the sealed anatomy, physiology, or
trauma banks.

## What one pixel means

Each visible source pixel is a physical cell. The encoder receives a strict
74-channel living observation:

- occupancy (1)
- tissue one-hot, including background (15)
- material one-hot (10)
- anatomical part-owner one-hot (17)
- emission intensity (1)
- eight physiological system weights (8)
- eight system-role fields: core, conduit, or effector (8)
- ten continuous cell-state fields: health, fluid fill, nutrient, energy,
  mass, stiffness, clotting, scar bias, regrowth, and healing class (10)
- premultiplied identity RGBA (4)

The last four channels are part of the living observation because color is an
identity trait. Earlier probes omitted them and correctly exposed appearance
collapse: varied organisms converged to mint bodies with purple cores. The
current contract makes that failure impossible to hide.

## Model

`ContinuousOrganismRasterVAE` is a condition-aware convolutional VAE:

- two encoder scales reduce 48x48 to a 12x12 Gaussian field;
- the frozen smoke model has 12 continuous latent channels per spatial cell;
- family, subtype, role, and 16 normalized genome/symmetry traits condition
  every residual stage through FiLM;
- the decoder directly rasterizes continuous RGBA and also predicts occupancy,
  tissue, material, part ownership, emission, eight physiology channels, and
  ten cell-state channels;
- a soft family-weighted bilateral loss encourages organism-like chassis and
  appendage symmetry without constraining anomaly morphology as strongly;
- KL regularization, free bits, occupied-cell weighting, edge preservation,
  alpha/occupancy agreement, and semantic losses prevent a merely decorative
  image autoencoder.

The posterior mean is a stable organism state. Interpolating two means and
their condition vectors is neural fusion. Applying correlated continuous
perturbations is neural mutation. Both are decoded by the same rasterizer and
are demonstrated as exact-replay visual artifacts.

## Frozen smoke v1

Authoritative output: `outputs/organism_raster_vae/smoke_v1`

- source SHA256: `96a91ae88486a686c7f87202e87331fc57da727918dfae2b2c2efa722371f704`
- manifest SHA256: `c549b50e0dbb3d34648e10434c2a6aea0a1c8cbb3c45832126da25f66fa60d35`
- checkpoint SHA256: `f632f9273d9b509521eaf0df0cea9a9c8c9d2868baa262bce0df6c8ba85dfd0c`
- model state SHA256: `2b9d3424e87a746fd8e890ad21ac6342d8bfcf33d6d56ac3f039220e67bed1a6`
- 45 exact upstream organisms; family census 11/10/9/8/7
- loss `7.44915438 -> 0.47730958` over 1,024 BF16 updates
- silhouette IoU `0.94768065`
- RGBA MAE `0.03384309`
- tissue/material/part accuracy `0.98177 / 0.97951 / 0.98010`
- visible physiology MAE `0.05169355`
- visible cell-state MAE `0.08297454`
- posterior mean absolute magnitude `1.91718`; mean posterior standard deviation
  `0.17913`
- 35.27 seconds training; 222 MiB peak reserved VRAM

All PNGs are rendered on CPU from the frozen checkpoint. Validation reloads the
checkpoint with `weights_only=True`, recomputes every metric over all 45
organisms, regenerates the reconstruction/fusion/mutation sheets byte-for-byte,
and verifies all upstream and source hashes.

This is a representation/rasterizer result, not a free generator. A prior has
not yet been trained over the organism latent distribution, and the checkpoint
is not integrated into the native runtime.

## Route toward neural gameplay

The intended replacement order is incremental and measurable:

1. Train a conditional flow/diffusion prior over these continuous organism
   fields so morphology construction, fusion, and mutation become neural.
2. Train a graph neural dynamics model over cells and bonds for motion,
   collision response, cutting, fracture, reconnection, and polyp fate.
3. Train a neural cellular automaton over the physiology/state channels for
   fluid transport, metabolism, injury cascades, clotting, scarring, healing,
   growth, plant tessellation, and reproduction.
4. Train sensory encoders and recurrent policies whose behavior degrades when
   neural cells or pathways are injured.
5. Train an ecological world model for resource flow, building, cities,
   interspecies strategy, and anomaly powers.

During research, deterministic validators remain outside the playable model so
neural failures are observable. In the final runtime, gameplay decisions and
state transitions can be neural while menu, viewport, HUD, tensor scheduling,
and safety/resource boundaries remain conventional host code. That is the
practical interpretation of “no deterministic gameplay code”: neural behavior
without pretending the executable and GPU kernels cease to be software.

## Commands

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m forge.organism_raster_vae smoke `
  --output outputs/organism_raster_vae/NEW_IMMUTABLE_NAME `
  --device cuda --steps 1024

$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.organism_raster_vae validate `
  outputs/organism_raster_vae/smoke_v1
```
