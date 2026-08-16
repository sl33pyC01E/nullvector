# Sparse causal world action editor v5

V5 turns the frame-transition model into a copy-preserving neural editor. The frozen world VAE still supplies the 48x32x32 latent raster, while an action-conditioned DiT predicts two coupled quantities: a one-channel causal edit gate and the latent delta inside that gate. The next latent is exactly `current + sigmoid(gate) * delta`.

This addresses the v4 failure directly. The spatial v4 model learned a strong action identity signal, but it repainted stable cells and lost to latent persistence. V5 initializes as an exact copy, supervises the edit support from raw pixel changes, penalizes edits outside that support, upweights changed cells in latent and RGB space, and reports changed-region quality separately.

Control counterfactuals are evaluated on aimed actions (`impact`, `scrape`, `cut`, `beam`, and `projectile`). Local actions such as healing and crafting are expected to ignore cursor direction and are not used to inflate or suppress the control gate.

The raw-pixel persistence score remains in the report as a diagnostic, not a promotion gate: decoding any latent through the VAE incurs its reconstruction floor, while raw persistence bypasses the neural rasterizer entirely. Neural-pipeline acceptance instead requires improvements over latent and refined-VAE persistence globally and within true edit regions, correct action/control preference, and nontrivial edit-mask overlap.
