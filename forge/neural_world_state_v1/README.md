# Neural world state v1

A learned slow-state codec for recurrent world simulation. It compresses biome topology, city materials, elevation, walkability, navigation cost, biomass, minerals, moisture, energy, family, season, development, and disturbance.

The output is a 20×8×8 spatial latent and a 64-value global state. These are the next conditioning inputs for the action-conditioned recurrent frame model.

## Accepted result

- 1,102,990 parameters
- 98.9% terrain accuracy
- 99.0% city foreground IoU
- 96.8% minimum specialized-material recall on the full held-out set
- 0.038 continuous-field MAE
- 0.006 global-condition MAE
- 1.26 ms per encode/decode on RTX 4090
- 28 MiB peak reserved VRAM during the runtime audit

The model is a state codec, not yet a state transition model. Its latents still need end-to-end conditioning against the recurrent action model before the scaffold’s 64-value state input can be retired.
