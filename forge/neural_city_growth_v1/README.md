# Neural city growth v1

An incremental city model for the world simulation. It receives a 24×24 local city patch plus family, culture, biome, project, resources, site, and growth stage. It predicts the next categorical construction state.

The model controls room and material placement. A separate physical compiler enforces three hard rules: unaffordable actions are exact no-ops, existing construction cannot be repainted by a growth action, and every occupied island is joined to the settlement network.

## Accepted baseline

- 5,674,120 parameters
- 30/30 rollout ticks produced meaningful construction
- 30/30 preserved existing material
- 30/30 formed one accessible toroidal network
- 26/30 expressed the requested utility, garden, or storage material
- 4.73 ms per local growth tick on RTX 4090
- 42 MiB measured runtime allocation

Exact agreement with one procedural blueprint is reported only as a diagnostic. Multiple city layouts can satisfy the same intent.

## Artifacts

- `examples/models/neural_city_growth_v1_ema.pt` — inference-only EMA weights
- `examples/showcase/neural_city_growth_v1.png` — six-step rollouts for all five families
- `examples/showcase/neural_city_growth_v1_report.json` — accepted audit

V1 is a foundation, not the final society engine. Its main known miss is purpose detail: four of thirty rollout actions constructed valid connected rooms without the requested specialized accent material.
