# Map Decorator Public-Entropy Residual v4

Status: corrected hybrid substrate plus one accepted, frozen-core 64-step CUDA residual
calibration. It is not production-integrated or a Godot asset source.

## Why v3 failed

V1 through v3 tried to infer sparse object locations from semantic map features and unrelated
smooth noise. The authoritative renderer actually places objects with a SplitMix64 coordinate
hash keyed by the public map seed, catalog index, and placement rule. Although the global
condition vector contains a low-precision seed embedding, a convolutional network cannot
generalize a pseudorandom 64-bit hash to held-out maps. V3 also capped each object head at 256
cells while a valid 256x256 sentinel can contain thousands. Those are substrate defects, not
evidence that a larger locator needs more epochs.

## V4 substrate

V4 exposes four exact boolean proposal channels—two theme-local decal classes and two prop
classes—computed from the same public map seed and versioned placement rule available at
runtime. Class legality and hard-empty topology masks are applied before neural consumption.
No target array is read. The neural residual tower is responsible only for resolving rare
same-cell proposal conflicts, refining bounded counts, and learning categorical/style context.
The decoder is structurally unable to select an off-proposal, illegal, colliding, or hard-empty
object cell. Its quota cap is 4,096 and is always clamped to actual candidate count.

The deterministic untrained smoke already evaluates all 24 untouched test sentinels and all
six themes. It is an architecture/substrate proof, not a trained-model claim. The separate
full-corpus audit covers every one of the 3,096 immutable maps in process-isolated chunks.

## Current evidence

The visually inspected smoke is
`outputs/map_decorator_production_v4/public_proposal_smoke_v1`. Its target and decoded columns
are visually near-identical across arena, rooms, caves, archipelago, garden, and anomaly. With
zero trained updates, the proposal-conditioned decoder reaches test decal IoU `0.981815` and
prop IoU `0.929559`; both exceed the unchanged calibration gates by a wide margin. This is
credited to the corrected procedural substrate, not neural training.

The full audit is `outputs/map_decorator_production_v4/full_proposal_audit_v1`. Eighteen
isolated workers covered all 216 source shards and all 3,096 maps with zero retries. Proposal
recall is exactly `1.0` for both heads on train, validation, and test. Precision is:

- train: decal `0.976073`, prop `0.994561`;
- validation: decal `0.974956`, prop `0.993338`;
- test: decal `0.977386`, prop `0.996212`.

There are zero missed target cells. The small residual is entirely extra public proposals that
the neural conflict suppressor can learn to reject.

Evidence identities:

- v4 contract: `35b1d91747dbb9133373f08ffb2529ccb053272391b212ac47756bec01c324a1`;
- source: `08b0a22752d8246d2a9524824d33ae367bcf3501289185924a8fb1688f677c16`;
- smoke report: `c6b6a695a6a781cf4d79179a31189bb68f1b8b59fdcc1e0dcab114a5f9ad0df2`;
- smoke contact sheet: `e17aef4471ed3cba38249620426613f634066d0302b12eccf0d8bda1888b5453`;
- full-audit report: `2b5055a5162f982be45b55754ca40d1f417e1b6b985d6f8308527ce227aa9c8f`;
- proposal records: `5494ed55a49e3d9f1b85c0d3b25a1587f1a803127b81f42033ca7abe2429dde2`;
- authoritative sample registry: `ce7d0ec801450f67687ab17091dbf83e5d4cd4da2fd093a45e67cc587e1a3582`.

## Commands

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.map_decorator_production_v4 smoke `
  --corpus outputs/map_decorator_corpus_v1 `
  --index outputs/map_decorator_production_v2/foreground_index_v2 `
  --output outputs/map_decorator_production_v4/public_proposal_smoke_v1 `
  --visually-inspected

python -m forge.map_decorator_production_v4 audit-corpus `
  --corpus outputs/map_decorator_corpus_v1 `
  --index outputs/map_decorator_production_v2/foreground_index_v2 `
  --output outputs/map_decorator_production_v4/full_proposal_audit_v1
```

The residual package now includes an accepted bounded calibration at
`outputs/map_decorator_production_v4_training/cuda_calibration_v2_frozen_core`. Its selected
EMA improves validation/test decal macro-IoU to `0.995097/0.998597` and prop macro-IoU to
`0.989279/0.989276`. Because the categorical core is frozen, variant and emission outputs are
exactly unchanged from the procedural baseline. Independent full-split replay passed over all
576 validation maps and 24 test sentinels. See `docs/map_decorator_v4_residual_training.md` for
the complete gates and identities.

This is still a calibration boundary: production schedules and runtime integration remain
unauthorized. A future schedule must preserve the frozen-core separation, immutable proposal
authority, full-split baseline comparisons, and exact legality/provenance gates.
