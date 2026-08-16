# Multi-world sparse action editor v6

V6 is the generalization pass for the copy-preserving action DiT. It warm-starts from the localized v5 editor, but trains on 39 complete worlds and never selects a checkpoint using the four final test worlds.

The split is by complete deterministic ecology rather than by shuffled frame. Three whole worlds select the EMA checkpoint every 1,000 updates; four different whole worlds provide the final causal and visual gates. This prevents adjacent frames or repeated organisms from leaking across splits.

V6 removes rare-action magnitude inflation, adds an explicit per-transition edit-magnitude loss, keeps the learned support/delta factorization, and requires global and changed-region improvements over latent persistence. Correct action, aimed-control preference, VAE-pipeline RGB improvement, and edit-mask overlap remain independent gates.
