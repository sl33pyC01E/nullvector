# Neural map topology prior v2 foundation

This is an additive architecture foundation. It does not modify or reinterpret
the frozen v1 prior, its checkpoints, or the 48-case seeded generation bank.

## Why v2 exists

The v1 generation audit exposed two concrete training/model mismatches:

1. The v1 mask helper always left at least one valid latent token visible. Free
   generation starts with every valid token masked, so production training never
   saw its actual deployment condition.
2. Four full-resolution 3x3 residual blocks provided only local context. Rooms,
   archipelagos, and anomalies require map-scale grammar, and their v1 proposals
   required roughly 9.6%, 14.2%, and 14.1% deterministic repair respectively.

## New contract

- `full` is a first-class mask mode and masks exactly 100% of valid tokens.
- Other modes cover high-random, rectangles, half-planes, corridors, and coarse
  islands. Every six-sample cycle exercises all six modes.
- Sparse start/exit/objective/spawn sockets expand into exact local-radius fields,
  coordinates, boundary/radial fields, and a clearly labelled soft start-to-exit
  corridor hint. The hint is not represented as an authoritative route.
- A multi-scale encoder/decoder handles odd and rectangular latent shapes without
  cropping. Theme/global/mask conditioning is injected by FiLM at every scale.
- The bottleneck explicitly mixes row, column, and global masked means. A smoke
  gate proves that changing the top-left token changes logits at the bottom-right
  of a 31x47 latent grid.

## Claim boundary and next training gate

The two-step CPU smoke proves deterministic execution, exact replay, full-mask
training, shape support, source/corpus binding, and whole-map influence. It is not
a quality or production checkpoint. It does not compile maps or enter Godot.

A future segmented CUDA calibration must predeclare and report, per theme:

- free-generation condition adherence;
- raw required-terminal and radius-one connectivity;
- mean, p95, and maximum compiler repair fraction;
- room hierarchy, archipelago land/water balance, anomaly radial grammar, arena
  route diversity, cave continuity, and garden tessellation diagnostics;
- uniqueness of raw tokens and compiled maps;
- exact checkpoint resume and fresh-process generation replay.

No v2 checkpoint should be promoted merely for masked reconstruction accuracy.

## Commands

```powershell
python -m forge.map_topology_neural_prior_v2 smoke outputs/map_topology_neural_prior_v2/smoke_next
python -m forge.map_topology_neural_prior_v2 validate outputs/map_topology_neural_prior_v2/smoke_next
python -m pytest tests/test_map_topology_neural_prior_v2.py -q
```

## Frozen foundation result

The current immutable smoke is
`outputs/map_topology_neural_prior_v2/smoke_v2`.

- manifest SHA-256: `f4451e83bccbe97cb595343ec30e22a495f2b2c76a83f562cb64118c5b3d915c`
- source SHA-256: `4f85161b02d14ac580e512eda393f4bd0487a6a10d24d1197237dd5a203fe9c5`
- checkpoint file SHA-256: `9343d20ea9127daae04cb1592a4326024e01aef485c68a4211fffeb429b4117f`
- six of six full-mask proposals were distinct
- opposite-corner logit delta on the 31x47 probe: `0.003397643566131592`
- maximum theme-counterfactual logit delta: `2.3039660453796387`
- exact build/validation replay passed in separate CPU processes
- v1 prior, latent corpus, production trainer, seeded generator, and v2
  foundation passed together: 30 tests

`smoke_v1` predates the final source cleanup and is retained only as stale,
source-bound evidence. `smoke_v2` is the current authority. Neither smoke is a
production-quality topology generator.
