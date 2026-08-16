# Causally aligned world-action transition DiT v3

This model corrects a causal alignment error in the first action-conditioned
world predictor. The previous corpus associated an action with the rendered frame
after that action had already occurred. V3 learns the deployable transition:

`world[t] + command[t+1] + controls[t+1] + state[t] -> world[t+1]`

The target is the continuous latent field of the frozen high-fidelity world VAE.
The 39.5M-parameter spatial transformer predicts a residual latent transition in
one pass. Training uses changed-region weighting, inverse-frequency action
balancing, edge-aware residual loss, EMA weights, and the multi-seed balanced
physical curriculum.

Held-out evaluation reports persistence, correct-action, wrong-action, and
zero-control errors globally and for every represented action. A model is only
evidence of causal control when the correct command beats both persistence and a
deliberately wrong command on a world seed excluded from training.

The runtime can apply the learned step recurrently. This is the immediate bridge
between the specialist physical scaffold and a future monolithic recurrent
DiT/action/VAE engine; the deterministic simulator remains the teacher rather
than being represented as the final architecture.
