# Action-conditioned latent world DiT v1

This milestone is the first learned future-frame engine for the native nature stage. It consumes the continuous 48x32x32 world-VAE latent, one of 22 recorded player/world actions, four analog control values, 64 ecological state features, and a flow time. A 39,499,008-parameter spatial DiT predicts the latent velocity toward the world four simulation ticks later.

## Frozen training result

- Corpus: six sealed 60-frame native teacher episodes; five train seeds and one entirely held-out seed.
- Training pairs: 280; held-out pairs: 56; horizon: four ticks.
- Optimization: 5,000 CUDA updates, BF16 autocast, EMA 0.9995.
- Held-out latent MAE: 0.210475 versus 0.342356 for copying the current latent, a 38.52% improvement.
- Held-out decoded RGB MAE: 0.022658 versus 0.026877 for copying the current frame, a 15.70% improvement.
- Compact runtime checkpoint: `game/generated/models/world_latent_dit/action_dit_v1.pt`, 79,032,462 bytes, BF16 weights.
- Source SHA256: `4924c6744ead983174bd3def9bcaab473b08b388c2f9effe13cf5d951cdf8ef5`.
- Corpus SHA256: `461fd0008e0db62e6575d8c8ae75693ab10caa160851f5e145185db267f6b46c`.

## Honest limitations

The model clears both persistence baselines and tracks creature/event motion on the unseen seed, but its decoded static architecture is visibly softer than the target. It is therefore a valid learned dynamics milestone and an optional future-state oracle, not yet the default rasterizer or authoritative simulation. The next visual-quality iteration should explicitly preserve static latent regions while allocating capacity and loss to changed cells, then prove long-rollout stability rather than optimizing only one four-tick jump.

The deterministic ecology remains the teacher and safety oracle. This model is one specialist in the planned ensemble; a later reverse-distilled action-DiT/VAE student may take over only after it matches the scaffold's cellular causality, organ damage, physics, reproduction, societies, and player agency.

The native demo exposes the frozen v1 model through F6. It encodes the current raw world viewport, conditions on the selected action, WASD vector, aim vector, and 64 world-state features, predicts the +4-tick latent, then decodes it through the learned pixel-cell refiner. Predictions are cached for responsiveness, bordered in magenta, and labeled non-authoritative. Teacher recording continues to capture the raw scaffold, preventing the student from recursively training on its own artifacts.

## Residual v2 experiment

A second 39.5M-parameter model replaced eight-step flow integration with a one-pass residual prediction and added changed-region, static-drift, and edge losses. On the identical held-out seed it improved latent MAE by 37.97% and decoded RGB MAE by 10.40% over persistence. It preserved the broad layout but did not beat v1's 38.52% / 15.70%, and visual inspection showed that the dominant softness remained in the shared VAE decoder. The v2 source and evaluation are retained as evidence, but no v2 runtime checkpoint is promoted. The next iteration targets the actual bottleneck with a latent-compatible neural pixel refiner.
