# Structured organism raster VAE v3

V3 is a 114,392,108-parameter continuous hierarchical VAE calibration built against morphology v2. It replaces neither the runtime nor the prior VAE yet.

Why v2 was insufficient: its excellent reconstruction scores were measured over only 45 already-converged organism constructions. It learned those fields accurately but could not improve weak family morphology, and its single smooth high-resolution head softened thin appendages.

V3 uses 480 morphology/motion samples: 30 identities across 16 loop phases. Five identities—one per family and all their phases—are excluded from training. The decoder uses three continuous latent scales:

- 6×6 global/chassis latent;
- 12×12 anatomy latent;
- 24×24 fine-cell latent.

It produces a 96×96 RGBA raster plus 48×48 cell-occupancy and tissue heads. High-resolution alpha is a learned residual over the learned cell head; this remains a neural decoder path, not deterministic raster cleanup. The loss directly supervises alpha, silhouette overlap, foreground color, logical cell occupancy, tissue identity, and multiscale edges.

The current bounded calibration is at `outputs/organism_raster_vae_v3/calibration_1200_alpha_scaffold`:

- 1,200 updates in 208.60 seconds on RTX 4090;
- held-out alpha IoU 0.56319;
- held-out RGBA MAE 0.02693;
- held-out foreground RGB MAE 0.06253;
- held-out tissue accuracy 0.95743;
- 2.94 GiB peak allocated VRAM;
- promotion is explicitly disabled.

The contact sheet was visually inspected. It is a viable direction and a large improvement over the first v3 calibration, but thin animal legs and the machine undercarriage still need work. A full run should wait for human approval of morphology v2. Likely next changes are appendage-aware latent tokens, stronger small-component recall, and temporal/identity palette consistency gates.
