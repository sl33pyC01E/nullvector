# Composite neural world V1

Build 1 is a callable DiT + VAE composite backed by the factorized neural
teacher. The Action-DiT advances visual world latents; the world VAE decodes
frames, and a refiner bound to that exact VAE improves held-out MAE by 47.6%;
a compact actor-state student advances causal organism
state; the continuous cell VAE renders posed cellular bodies; and the selected
cellular NCA advances physiology.

This is not yet the final one-model game. Physics projection, scheduling, and
specialist routing remain deterministic. The older unbound pixel refiner stays
excluded; Build 2 uses the newly trained, exact-parent-bound replacement. The
important boundary is now real:
the promoted neural components load together and expose one composite runtime.
The next student distills the remaining locomotion, manipulation, ecology, and
society specialists into recurrent hidden state before mobile work begins.
