# Neural ecology counterfactual model

`forge.nature_counterfactual_nn` is the action-conditioned partner to the world
timeline transformer. It evaluates all five physical ecology interventions from
the same 24-step, 64-channel world history and predicts a post-action world
state, ecological benefit, and execution risk.

The production model is a 25,695,554-parameter transformer with width 512,
eight encoder layers, and eight attention heads. Its grouped causal curriculum
contains 6,554 distinct ecology histories and all five interventions for every
history, so held-out ranking never compares an action against a different base
world. Effects are context-sensitive: seed arks matter most under trophic
scarcity, condensers under heat/toxin/water stress, anchors under phase pressure,
wards under predation, and habitat knots when population has outgrown colony
capacity.

The promoted BF16 CUDA run used 2,800 updates. Held-out metrics:

- post-action state MAE: 0.02658;
- benefit MAE: 0.02641;
- risk MAE: 0.00906;
- best-action ranking accuracy: 94.36%.

At runtime the planner evaluates all five actions simultaneously and exposes the
three highest benefit-minus-risk choices. A player choice still passes through
the finite inventory and physical-authority layer: it can modify real fields,
powder deposits, structures, creature velocities/intents, score, and the exact
world event ledger. The neural model advises; it does not mint resources or
bypass topology.
