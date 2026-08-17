# Recurrent world student v3

V3 initializes from the accepted recurrent Action-DiT and actor-state student, then trains both on self-fed four-step windows from the contiguous V8 corpus. Predictions are detached between steps so memory stays bounded while the model still learns from its own rollout distribution.

This stage keeps actions, controls, and macro world state external. Visual latent state and 128-feature organism state are recurrent. One-step held-out improvement is only the first release gate; separate frame-space rollout evaluation determines whether it can replace the Build 3 action path.
