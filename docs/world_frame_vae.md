# Continuous world-frame VAE

`forge.world_frame_vae` is the first neural raster boundary for the complete
cellular world viewport. It encodes a 256×256 RGB world frame into a continuous
`48×32×32` latent and decodes it back to RGB. Deterministic HUD, menus, controls,
and inspection overlays remain outside the latent.

Training data comes from six independently sealed action-teacher episodes across
six world seeds. Five episodes (300 frames) are training data; the sixth episode
(60 frames) is held out as a complete world-seed split. Frames retain creatures,
fluid/material fields, structures, shadows, sensory geometry, projectiles, and
movement, while controls and cellular authority live in parallel tensors.

The promoted v2 model has 16,913,635 parameters and trained for 4,800 BF16 CUDA
updates with RGB, gradient-edge, Laplacian-edge, multiscale, and low-weight KL
losses. Held-out results:

- RGB MAE: 0.01552;
- RGB MSE: 0.000882;
- PSNR: 30.54 dB;
- edge MAE: 0.01577.

The native game loads compact BF16 EMA weights. `F7` toggles neural reconstruction
for the square world viewport; raw physical scaffold rendering remains the
default comparison view until the VAE is trained on a much broader world corpus.
The VAE output receives only a small viewport-local unsharp reconstruction pass
before nearest-neighbor display scaling. `F8` teacher recording always captures
the pre-VAE physical target, preventing self-distillation from silently replacing
ground truth.

The rejected v1 `32×16×16` latent is preserved under outputs as evidence: it
reached 25.98 dB but visibly smeared thin appendages. It was not promoted.
