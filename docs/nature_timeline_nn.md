# Neural world timeline

`forge.nature_timeline_nn` is the first learned long-horizon ecology layer in the
native nature-stage game. It does not replace the cellular simulation. It reads a
rolling 24-step summary of that authoritative world and predicts:

- the next 64-channel macro-state;
- one of ten ecological/civic event classes;
- calibrated confidence for that event.

The input summarizes population, family balance, lineage and colony activity,
births, deaths, predation, mutation, ten resource fields, climate and season,
seven living-body system capacities, and the current behavioral-intent mixture.
The transformer has 25,304,139 parameters (width 512, eight encoder layers and
eight attention heads). Its deterministic 32,768-sequence causal curriculum
contains resource depletion and renewal, climate shocks, migrations, colonies,
mutations, construction and discovery.

The production model trained for 1,400 BF16 CUDA updates. Its frozen held-out
metrics are 0.07270 state MAE and 91.21% event accuracy. The compact runtime
checkpoint stores BF16 EMA weights and is loaded with exact format, architecture,
and source-provenance checks.

In the game the model observes the live world every 15 simulation ticks. The HUD
shows `NN>EVENT confidence`; the Living Chronicle repeats that forecast beside
the exact cellular and society histories. Region transitions and save restores
clear the temporal memory before beginning a new local forecast sequence.

This is an ensemble milestone toward the eventual action-conditioned monolithic
DiT/VAE student: deterministic systems remain the ground truth and safety rail,
while the learned timeline becomes a measurable source of anticipation and,
later, world-scale planning pressure.
