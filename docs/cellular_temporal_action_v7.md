# Cellular temporal Action-DiT v7

V6 learned where edits belong, but its world image plus 64 global ecology values cannot reveal a selected creature's hidden organ damage, fluids, scars, neural connectivity, graft topology, or limb phase. It also recomputed latent normalization after warm-starting, changing the coordinate system under the inherited editor.

V7 moves the ensemble toward the intended physical neural engine:

- two visual observations expose motion and appendage phase;
- the selected actor's 128-value physiology and 8x32x32 cell/organ field are privileged inputs;
- the model predicts the next VAE latent, actor physiology, and cell field together;
- all new adapters are neutral at initialization, preserving the exact v5 latent editor and exact cellular persistence;
- training will use the frozen parent latent mean/std rather than recomputing them;
- atomic `latest.pt` checkpoints and immutable validation milestones limit interruption loss to one validation interval.

VAE encoding is also published as an immutable, source-bound corpus with one
compressed shard per teacher world. Every shard records its raw trajectory,
VAE checkpoint and EMA identity, tensor shapes/dtypes, byte hash, and semantic
array hash. A restart can therefore reuse validated latents instead of spending
another full pass reconstructing them.

The trainer optimizes one causal transition jointly: sparse next-frame latent,
next 128-value physiology, and next 8-channel cell/organ field. Validation is
split by whole world and scores all three authorities against persistence;
counterfactual actions and aim controls must also be worse than the correct
conditioning. The v5 latent mean/std are frozen exactly instead of recomputed.
Training writes an atomic resumable checkpoint every 500 updates and immutable
best/2,000-update milestones, including optimizer, EMA, and random-generator
state.

The privileged model is an ensemble teacher, not the final product. Once it establishes that physical state transitions are learnable, a recurrent visual student can infer hidden state from frame history; the eventual monolithic Action-DiT/VAE student can then absorb both.
