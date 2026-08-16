# Spatial action/world DiT v4

V4 learns a single causal world transition while preserving explicit spatial
control. Its deterministic teacher brackets every command with a setup and
settle frame, centers the acting organism, varies target direction and range,
and balances all 22 action classes across independent worlds.

The 39M-parameter transformer receives four spatial fields alongside the world
VAE latent: actor center, aim point, actor-to-aim ray, and locomotion direction.
Training emphasizes changed pixels and changed latent cells, backpropagates
through the frozen VAE and learned pixel refiner, and contrasts the correct
transition against both a wrong action and a mirrored control.

Promotion is fail-closed. A held-out world must prove all of the following:

- lower latent error than persistence;
- lower error than a wrong action;
- lower error than a mirrored spatial control;
- lower refined RGB error than refined persistence;
- lower raw RGB error than copying the raw prior frame;
- lower changed-region RGB error than refined persistence.

This is an ensemble-stage component toward the eventual monolithic recurrent
action-DiT/VAE engine. The procedural physics remains the authoritative teacher
until the neural transition passes every gate.
