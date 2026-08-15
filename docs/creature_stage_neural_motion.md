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

The first immutable production segment is now published at
`outputs/creature_stage_neural_motion/production_v1/cell_motion_0001000.pt`.
It contains step 1,000 EMA state `642f34580da10dbbf5d88f6932a7716c89443878a89398cdda4b9569bff41146`
and model state `2a5879d3658d78688b44963dc5ff2758060d1ddf4e61aaa7f468b2f366226e29`.
The remaining 19 segments are intentionally not inferred from this first
checkpoint; each must resume through the strict optimizer/RNG/checkpoint
chain and pass the same free-disk and free-VRAM guards.

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
provenance. Structural replay is exact across backends; finite metric values
use the manifest-recorded symmetric comparison bound of 5e-5 absolute and
2e-5 relative to accommodate CPU/CUDA kernel rounding. Gates, counts,
integers, identities, and promotion verdicts never receive tolerance. It
measures position and velocity error, improvement over copying
the previous state, bond-relative coherence, predicted versus target action
energy, appendage/core activity, loop closure, displacement bounds, and exact
zero outside native cells. Aggregate values are recomputed from clip records;
even a fully rehashed nested edit fails validation. The five sealed test
chassis cannot be evaluated without an explicit `--release-sealed-test` flag.

Smoke checkpoints may be evaluated to prove the mechanism, but are never
promotion-eligible. A complete 5-family x 13-motion x 72-frame validation
matrix, all quality gates, and production EMA authority are jointly required.

The current mechanism proof is preserved at
`outputs/creature_stage_neural_motion/evaluation_smoke_full_v2`. It covers 65
clips and 4,680 prediction-fed frames and exact-replays on its CPU reference
backend. Its semantic identity is
`a9c6beee6e6f486f6ca6d3fa812a3b85c958dd612eeafbf0a637a3d8b88637e7`;
the manifest file SHA-256 is
`a2e6430013fa18931cabf0748d3c948aaaf9e0d2010055f937ffd14c1366a400`.
As intended, the four-update smoke model is rejected: mean position error is
1.49306437 px, mean velocity error is 1.10997799 px, copy-previous improvement
is -0.01370313, and motion-energy ratio is 3.14874795. It nevertheless remains
finite, bounded, bond-coherent, and exactly zero outside occupied cells. This
distinguishes a sound evaluator/model interface from a trained motion model.

The first production diagnostic is preserved at
`outputs/creature_stage_neural_motion/evaluation_step_1000_v2`. Its CUDA
report then independently replayed on CPU across all 4,680 recurrent frames.
The semantic identity is
`8d67ee24e2e429b342a42dbd4290e57847cf9f95d0d06bc2134088e25bbc1d7b`
and its manifest file SHA-256 is
`1ccee7d7286ddd5de0453de91594d517322ddcf6096047b0c33d07cbb6322fe7`.
It is correctly rejected: step 1,000 is not the final production checkpoint;
position MAE is 1.41918155 px, velocity MAE is 1.10005672 px,
copy-previous improvement is -0.01258480, and energy ratio is 2.82749949.
This is evidence of a valid training/evaluation chain, not acceptable motion
quality. The previous `*_v1` evaluation directories remain historical
backend-exact artifacts and fail current v2 source authority by design.

## Portable ONNX authority

The same checkpoint can be exported as a single-file opset-18 ONNX graph with
dynamic batch size and fixed native-cell dimensions. The game-side caller owns
the recurrent state: it feeds each prediction back as the next frame's state.
No rasterized sprite or flattened animation is embedded in the graph.

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.creature_stage_neural_motion export-onnx `
  --checkpoint outputs/creature_stage_neural_motion/production_v1/cell_motion_0020000.pt `
  --output outputs/creature_stage_neural_motion/onnx_step_20000

python -m forge.creature_stage_neural_motion validate-onnx `
  --output outputs/creature_stage_neural_motion/onnx_step_20000
```

Validation runs ONNX's full checker, rejects external tensor files, verifies
every input/output dtype and dimension, tests batches 1/3/5, and compares
PyTorch with ONNX Runtime on 80 held-out examples spanning all five families,
four representative motions, and four phases. Every validation repeats the
numerical comparison; a rehashed parity claim is insufficient.

The current CPU smoke export at
`outputs/creature_stage_neural_motion/onnx_smoke_v2` is 1,102,737 bytes. Its
ONNX SHA-256 is
`ca07c28bce36bd3b5d96537dd54cd54b29b827e15f953899908944444e621615`,
semantic identity is
`a8bf0d97bf942ec8e13432e6203b1a4cb54644394c1a9383a1c375aeb19c11e2`,
and manifest file SHA-256 is
`67cedf42cfc8c605667630b3cc41a26917053f6953d05ee307e209e689a895bf`.
Maximum absolute error is 4.619e-7 and mean absolute error is 4.74e-8, with
exact zero outside occupied cells. This proves portability of the interface,
not motion quality; production EMA weights must pass the separate recurrent
promotion report before runtime integration.

Runtime promotion still additionally requires:

1. a complete segmented checkpoint chain;
2. a promotion-eligible prediction-fed validation report, with sealed test used
   only at final selection;
3. displacement, velocity, loop closure, appendage-root coherence, action
   timing, and long-rollout stability gates;
4. visual clips proving locomotion, breathing, emotes, actions, hits, and
   bounded corpse settling;
5. a portable export and numerical-parity report against promoted EMA weights;
6. native Godot parity plus frame-time and memory measurements;
7. evidence that the neural model improves or faithfully preserves the
   deterministic oracle rather than merely smoothing it.

Only then can it join the ensemble that eventually supervises the monolithic
action-DiT/VAE student.
