# Action-conditioned teacher trajectories

The native nature stage can now turn ordinary play into synchronized teacher
data for the eventual monolithic action-DiT/VAE student. Press `F8` to start or
seal an episode. Recording is local and bounded to 900 frames per episode.

Each sample contains:

- a 256×256 RGB frame of the cellular world viewport, excluding deterministic HUD;
- the 64-channel cellular/ecology world feature vector;
- normalized movement and cursor-aim controls;
- one of 22 discrete player actions;
- selected-entity identity and exact world tick;
- the timeline transformer's event and calibrated deltas;
- all five counterfactual actions with benefit, risk, population, and resource
  predictions.

Frames are sampled every three ecology ticks, while discrete actions force an
immediate sample so short cuts, beams, grafts, interactions, construction,
abilities, metamorphoses, trades, services, and interventions are not lost.
Archives are compressed NPZ with canonical JSON manifests, exact shapes/dtypes,
artifact SHA-256, semantic named-array SHA-256, source provenance, a 900-frame
hard cap, atomic publication, and a 100 GiB free-disk floor.

The first replayed proof episode contains six synchronized frames and all tensor
streams under `outputs/action_teacher_v1/teacher-proof-v1/`. This recorder is the
dataset boundary between the current validated specialist ensemble and future
reverse distillation; it deliberately records both rendered evidence and the
physical state that produced it.
