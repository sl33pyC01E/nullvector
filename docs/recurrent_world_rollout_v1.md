# Recurrent world rollout evaluation

This evaluator compares the V3 self-fed student with its unchanged recurrent Action-DiT and actor-state parents. Both use the same adapted VAE delta compositor, exact initial frame, external actions, controls, and macro world state.

The untouched sixth world is sampled at 1, 2, 4, 8, 16, and 32 steps. Promotion requires the V3 candidate to beat both initial-frame persistence and its parents at every 4–32 step horizon.
