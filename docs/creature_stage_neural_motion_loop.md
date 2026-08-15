# NULLVECTOR // Loop-aware cellular motion successor

## Recovery boundary

This additive branch starts from the sealed rollout update-1,000 authority. It
does not overwrite the original one-frame transformer, the rollout successor,
or any of their checkpoints and reports. The later rollout update 1,500 remains
rejected evidence: it recovered energy but damaged locomotion geometry, motion
balance, and cyclic closure.

The workstation lost power after this branch's CPU smoke had published. The
manifest and checkpoint survived intact and independently exact-replayed after
restart. No CUDA production training has been started for this successor.

## What changes

The loop-aware sampler uses six prediction-fed frames instead of four and
reserves one quarter of eligible loop samples for sequences that cross the
authoritative frame `71 -> 0` boundary. Non-looping actions never wrap. Batches
remain exactly balanced across humanoid, animalian, plantlike, anomaly, and
machine families.

The objective is deliberately rebalanced from the first rollout experiment:

- state-delta weight rises from `0.25` to `1.00`;
- velocity weight rises to `0.55`;
- motion-energy weight falls from `0.25` to `0.10` now that collapse is no
  longer the dominant failure;
- appendage and local graph terms remain explicit;
- model predictions feed all frames after frame zero.

This is intended to repair cyclic seams and the copy-previous failure without
returning to inert motion or overdriving locomotion. The deterministic teacher
is still the validating scaffold; it is not shipped as a claim that the neural
motion problem is solved.

## Frozen CPU smoke

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.creature_stage_neural_motion_loop validate-smoke `
  --output outputs/creature_stage_neural_motion_loop/smoke_cpu_v1
```

The 251,780-parameter smoke trains for eight updates on five forced seam
sequences. It exact-replays six frames per family, including five explicit
`71 -> 0` transitions, keeps every padded cell exactly zero, and produces
nonzero appendage motion. It is an interface, sampler, loss, serialization,
and replay proof—not a visual-quality or production-promotion result.

| evidence | SHA-256 |
| --- | --- |
| loop successor source | `52058a15ac455ed5d128e8f53ca6aed36de10fa11c7a5bc6ecdc18eaf00deb4b` |
| parent rollout source | `045b88495134b8dfdcadbd76a6587f4138ea37a5645e60ddd5d33d3b4bb80856` |
| smoke semantic identity | `4fa225f7ccdd1ac14a025e40e0ef0651c6d748df995a1aa9bbdafb9a5670c0c3` |
| smoke model state | `5b3cf8fb79be8c8d67360167fbda25c0598a5296a7b289c98c4bef28cc7c795b` |
| smoke checkpoint file | `cea7e15a109fd17fe50a791c3e8c591a3587312f45e0b8c71b460da0ce0c3a91` |
| smoke manifest file | `2efda7058d7c43c5da52c94e03f013e7574cbcd296b35b034ceca926505a962d` |

Key deterministic diagnostics are five families, six frames, five
prediction-fed frames, five seam transitions, `1.147786856 px` seam position
MAE, `1.300292134 px` appendage motion, and exact zero outside-cell output.
These uncalibrated smoke magnitudes must not be used to compare this tiny model
against production checkpoints.

## Next guarded step

Production work should initialize a new optimizer and EMA from the frozen
rollout update-1,000 EMA, then run a single bounded segment under the new
six-frame/seam-quota contract. Promotion still requires the full 65-clip,
4,680-frame prediction-fed evaluation matrix and must improve copy-previous,
loop closure, locomotion geometry, and minimum clip energy together. A finite
loss or a visually lively sample alone is insufficient.
