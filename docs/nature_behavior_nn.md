# Neural nature behavior controller

This specialist replaces the deterministic ecology's intent and steering branch
while retaining the scaffold as teacher, validator, and fallback authority.

Each decision observes the creature's family mixture, developmental and ecology
traits, diet, life stage, seven causal organ-system capacities, energy, reserve,
reproduction state, current velocity, colony displacement, appendage/component
inventory, identity phase, ten local resource fields and gradients, and the
twelve nearest living organisms. A 3.56M-parameter transformer predicts one of
twelve ecological intents plus continuous top-down steering and urgency.

The v3 authority is trained on 46,800 decisions from twelve independent world
identities. World IDs divisible by five are held out. Its selected raw model
scores 98.25% held-out intent accuracy, 0.923 directional cosine, and 0.129
direction MAE. It runs batched every three ecology ticks, alongside the separate
4.46M recurrent ground-contact/muscle controller.

The deterministic scaffold remains necessary: it enforces incapacity, physical
metabolism, organ damage, reproduction, collisions, cell/material interactions,
and all validity bounds. This is an ensemble milestone, not the final monolithic
action-conditioned world model.
