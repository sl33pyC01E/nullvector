# NULLVECTOR // Native-cell motion transformer

## Role

The deterministic motion and intervention corpora are causal teachers. This
package is a Stage 2 replacement candidate: a learned recurrent model that
predicts the next local state of every actual native body cell from anatomy,
previous motion, and explicit controls.

It does not rasterize a sprite. It predicts cell dynamics that a later neural
presentation decoder can render. Visual quality therefore cannot hide broken
anatomy or action causality.

## Tensor contract

Every chassis is padded to at most 384 cell tokens. A boolean occupancy mask
excludes padding exactly. The 61 static values per cell contain:

- normalized body x/y, radial distance, and vertical order;
- one-hot tissue and organ identity;
- left, center, or right side and body centrality;
- appendage presence and cyclic appendage identity;
- width, height, asymmetry, symmetry, repair, metabolism, fertility, and bond
  strength genes.

The recurrent state has preceding x/y displacement and preceding x/y
velocity. The four-value target has the same layout for the current frame.
Values use a conservative 12-pixel motion bound.

Conditioning contains five-family identity, one of twenty structural
morphotypes, thirteen motion states, cyclic or bounded action phase, move and
aim vectors, attack and utility scalars, and a none/impact/terminal event.

The teacher is the strict native corpus at
`outputs/creature_stage_motion_corpus_v1_final_a`. Its manifest, binary, and
corpus identities form a separate semantic fingerprint, so model artifacts
fail closed if the native authority changes.

## Architecture

| field | value |
| --- | ---: |
| parameters | 27,409,156 |
| token width | 384 |
| blocks | 10 |
| attention heads | 8 |
| feedforward expansion | 4x |
| condition width | 384 |
| dropout | 0.05 |

Each block has two communication paths:

1. masked global self-attention coordinates organs, paired limbs, and
   whole-body action timing;
2. aggregation over the actual occupied eight-neighbor grid graph carries
   local chassis, joint, and appendage constraints.

Conditioned normalization injects identity, action, phase, and controls in
every block. The bounded output head reapplies the occupancy mask, making every
padded output exactly zero.

The loss combines displacement, velocity, graph-relative motion, temporal
acceleration, and an outside-body penalty. It directly targets the prior
failure modes where locomotion became global expansion/contraction or
appendages flickered independently of their roots.

## Split and deterministic sampling

Each family has four native chassis:

- morphotypes 0 and 1: training (ten chassis total);
- morphotype 2: validation (five chassis);
- morphotype 3: sealed test (five chassis).

Every training batch is family-balanced. A coordinate is a pure function of
seed, update number, and batch slot, selecting chassis, action, and frame
without mutable shuffle state. Interrupted and uninterrupted segments consume
the same curriculum.

## Current CPU evidence

```powershell
python -m forge.creature_stage_neural_motion model-info

$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.creature_stage_neural_motion smoke `
  --output outputs/creature_stage_neural_motion/smoke_cpu_v1_final `
  --steps 4

python -m forge.creature_stage_neural_motion validate-smoke `
  --output outputs/creature_stage_neural_motion/smoke_cpu_v1_final
```

The smoke uses a 251,780-parameter geometry only to prove the full interface
and optimizer path on CPU. It binds all five families in one batch. Fixed-batch
loss fell from 0.116575412 to 0.015344931 in four updates, with finite nonzero
gradients and exact zero outside every chassis.

| evidence | SHA-256 |
| --- | --- |
| model source | `2300cacade824488a69d1f191519e5809222f1de14ecd8d92f64f3ea1f3b5ec5` |
| smoke semantic identity | `829b6ef7a57ffb625e7646791bcf2864d2cf093116e74d403134473c114a1d1b` |
| smoke model state | `41a55d82e033a15ac27f7a7f320f1eb9007a6e0434e74250a39ae4c7b84bbd13` |
| smoke manifest file | `afc9b3add9b6c9cb962fc4e06d3ece22a0cc520ce072ebfaa5dae6f9027eefa0` |
| smoke checkpoint file | `da474777255b5239804d6f20f39d0dce1c4533e20eaca0610cdd133d3a7ab4d1` |

Earlier proof directories created before the checkpoint history registry was
tightened are preserved with `_pre_history_hardening` suffixes and are not
current authorities.

## Prepared production schedule

```powershell
python -m forge.creature_stage_neural_motion prepare-production

$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m forge.creature_stage_neural_motion segment `
  --output outputs/creature_stage_neural_motion/production_v1 `
  --end-step 1000
```

The immutable contract specifies 20,000 family-balanced updates, 1,000-update
checkpoints, AdamW at 2e-4, 1,000-step warmup, gradient clip 1.0, EMA 0.9995,
BF16 autocast with float32 loss, exact optimizer and CPU/CUDA RNG resume, at
least 16 GiB free VRAM, and a planned-output guard preserving 100 GiB free.

Production was not launched during this milestone because an unrelated sprite
training process left about 11.5 GiB VRAM free. The runner correctly refuses
that state rather than crowding or terminating other work.

## Promotion requirements

The CPU proof validates implementation, not learned quality. A separate,
source-bound evaluator now performs recurrent prediction-fed rollouts: only
frame zero comes from the teacher state, and every later state is the model's
own preceding output. This prevents a good one-frame loss from concealing
long-horizon collapse.

```powershell
python -m forge.creature_stage_neural_motion evaluate `
  --checkpoint outputs/creature_stage_neural_motion/production_v1/cell_motion_0020000.pt `
  --output outputs/creature_stage_neural_motion/evaluation_step_20000 `
  --split validation --frames 72 --motions 0 1 2 3 4 5 6 7 8 9 10 11 12 `
  --device cuda

$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.creature_stage_neural_motion validate-evaluation `
  --report outputs/creature_stage_neural_motion/evaluation_step_20000/evaluation_manifest.json `
  --replay
```

The report contains all 65 held-out family/action clips, per-clip evidence,
family and motion macro-aggregates, and exact checkpoint/teacher/source
provenance. It measures position and velocity error, improvement over copying
the previous state, bond-relative coherence, predicted versus target action
energy, appendage/core activity, loop closure, displacement bounds, and exact
zero outside native cells. Aggregate values are recomputed from clip records;
even a fully rehashed nested edit fails validation. The five sealed test
chassis cannot be evaluated without an explicit `--release-sealed-test` flag.

Smoke checkpoints may be evaluated to prove the mechanism, but are never
promotion-eligible. A complete 5-family x 13-motion x 72-frame validation
matrix, all quality gates, and production EMA authority are jointly required.

The current mechanism proof is preserved at
`outputs/creature_stage_neural_motion/evaluation_smoke_full_v1`. It covers 65
clips and 4,680 prediction-fed frames and exact-replays on CPU. Its semantic
identity is
`2a36d1a9e0ccdd0936463a7bf89ec797f692df88d802f9e42ed0a91c51ee2ccb`;
the manifest file SHA-256 is
`235a15ea7302145d58f3c5dda41509b78c815d77c1fac661473199a4792d1a5b`.
As intended, the four-update smoke model is rejected: mean position error is
1.49306437 px, mean velocity error is 1.10997799 px, copy-previous improvement
is -0.01370313, and motion-energy ratio is 3.14874795. It nevertheless remains
finite, bounded, bond-coherent, and exactly zero outside occupied cells. This
distinguishes a sound evaluator/model interface from a trained motion model.

Runtime promotion still additionally requires:

1. a complete segmented checkpoint chain;
2. a promotion-eligible prediction-fed validation report, with sealed test used
   only at final selection;
3. displacement, velocity, loop closure, appendage-root coherence, action
   timing, and long-rollout stability gates;
4. visual clips proving locomotion, breathing, emotes, actions, hits, and
   bounded corpse settling;
5. portable export and numerical parity against EMA weights;
6. native Godot parity plus frame-time and memory measurements;
7. evidence that the neural model improves or faithfully preserves the
   deterministic oracle rather than merely smoothing it.

Only then can it join the ensemble that eventually supervises the monolithic
action-DiT/VAE student.
