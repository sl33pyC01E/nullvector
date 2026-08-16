# Native ecology runtime performance

The live native demo keeps the 30 Hz ecology simulation authoritative while
rendering interpolated positions at up to 60 Hz. Neural world presentation is
performed asynchronously and cross-faded between completed VAE/refiner frames;
it never stalls the ecology step or input loop.

The August 2026 optimization pass changed implementation, not simulation
semantics:

- living-body topology and organ capacities are cached with exact array-based
  invalidation;
- fluid exchange, muscle forces, constraints, materials, and cellular sprite
  rasterization use vectorized arrays;
- all organisms and grafted locomotor banks share one batched recurrent neural
  locomotion pass;
- creature sprite work is age-prioritized and budgeted so a normal founder
  scene refreshes every creature within two presentation frames;
- continuous movement is interpolated between ecology ticks;
- field backgrounds, world summaries, VAE frames, and action-DiT futures are
  cached at their natural update rates.

On the 15-founder, 1440x900 reference scene with the CUDA neural stack loaded,
the headless presentation benchmark improved from roughly 3 FPS to 52.3 FPS in
raw cellular mode and 44.6 FPS with the continuous VAE plus learned pixel
refiner enabled. The benchmark is a comparative engineering measurement, not a
hardware-independent performance guarantee.

Focused ecology, anatomy, locomotion, action-teacher, and causal-transition
tests remain the correctness gate. The behavior controller is rebound only
after proving the optimized scaffold produces the exact same 32,768-sample
semantic corpus hash as the validated pre-optimization scaffold.
