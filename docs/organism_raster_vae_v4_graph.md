# Graph-token organism raster VAE v4

V4 is the structural successor experiment to the balanced 114M-parameter VAE v3. It keeps the v3 chassis, organ, fine-cell, anatomy, and global latent hierarchy, then adds graph-owned appendage tokens.

Each appendage token records its kind, bilateral side, segment count, gait phase, root, endpoint, direction, length, bend, paired-part status, locomotor mode, and current contact state. Decoder locations at 12×12 and 24×24 cross-attend to these tokens. A token-owner objective ties each appendage cell to the token that actually owns it.

This mechanism was selected after three scalar appendage-loss continuations failed their combined precision/recall gates. V4 warm-starts exactly from the sealed v3 calibration and initializes its attention gates at zero, so the parent decode is the starting point.

The bounded 100-step calibration is at `outputs/organism_raster_vae_v4_graph/calibration_0100_fresh_replay`:

- 115,421,230 parameters;
- 17.04 seconds of CUDA training;
- held-out alpha IoU improved from 0.56317 to 0.57249;
- appendage neighborhood F1 improved from 0.68562 to 0.69268;
- appendage recall remained 0.91275 versus the 0.92048 parent;
- RGBA MAE improved from 0.02693 to 0.02664;
- exact appendage-token owner accuracy reached 0.50655;
- all bounded quality gates passed;
- production promotion remains disabled.

Checkpoint publication and visual replay run in separate processes. The replay recomputes parent and graph metrics, requires exact equality with the training manifest, verifies every one of the 15 comparison panels independently, and only then publishes the contact sheet.

This is strong evidence for graph tokens as the next model direction, not a finished model. The next stage should train in short resumable segments, expand owner supervision to joints and organs, and evaluate full 16-frame held-out motion loops before any runtime integration.
