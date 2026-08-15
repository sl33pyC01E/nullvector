# NULLVECTOR // Prediction-fed cellular motion rollout successor

## Why this branch exists

The first native-cell transformer proved the cell/graph/control interface and
learned geometric accuracy quickly. Its held-out recurrent reports also caught
an important failure that one-frame training loss concealed:

| EMA step | position MAE | velocity MAE | aggregate energy ratio | weakest clip |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1.4192 px | 1.1001 px | 2.8275 | 0.3070 |
| 2,000 | 0.6195 px | 0.2426 px | 0.9657 | 0.1814 |
| 3,000 | 0.4972 px | 0.1530 px | 0.6295 | 0.1093 |

Geometry improved while several locomotion/action clips lost amplitude. Blindly
finishing the original 20,000-step schedule would reward the wrong behavior.
The v1 checkpoints and reports remain immutable evidence; this package is an
additive successor initialized from the audited step-3,000 EMA.

## Changed causal training signal

Each update samples one consecutive four-frame sequence for every family.
Frame zero receives the teacher state. Every later frame receives the model's
own preceding prediction, detached between frames to keep memory bounded. The
optimizer sees all four frame losses before one update.

The objective retains position, velocity, and local bond-relative losses, then
adds three anti-collapse terms:

- appendage-weighted position error, using the authoritative chassis cell flag;
- per-creature log-RMS motion-energy matching, with a finite epsilon-smoothed
  gradient at zero;
- a delta loss comparing predicted movement from the model-fed state with the
  teacher's intended movement from its teacher state.

The sampler is a pure function of seed, update, and batch slot. It is exactly
family-balanced and never samples across a clip boundary. Training uses
prediction-fed scheduled sampling without unbounded backpropagation through
time.

## Current CPU authority

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.creature_stage_neural_motion_rollout smoke `
  --output outputs/creature_stage_neural_motion_rollout/smoke_cpu_v1 `
  --steps 8

python -m forge.creature_stage_neural_motion_rollout validate-smoke `
  --output outputs/creature_stage_neural_motion_rollout/smoke_cpu_v1
```

The 251,780-parameter smoke is an interface/gradient proof, not a quality
model. It exact-replays all four prediction-fed frames for five families,
keeps padded cells exactly zero, and emits nonzero appendage motion. Its
energy ratio is 2.807897257, so it is explicitly not promotion evidence.

| evidence | SHA-256 |
| --- | --- |
| rollout source | `045b88495134b8dfdcadbd76a6587f4138ea37a5645e60ddd5d33d3b4bb80856` |
| smoke semantic identity | `529e3b8f1c4699818354b7d2f256de24fd2888eb929a9478f40824e1b87a08c8` |
| smoke model state | `6074311e4f8e253769f2c158357f0d7bb42ccb5fb6b5071ee7866707c825cb44` |
| smoke manifest file | `868c95e59b142f99c6ce907e433fd911e29a55cb62943d409dde5fe1f722267f` |

## Guarded production pilot

The production contract initializes both live and EMA weights from the frozen
v1 step-3,000 EMA. It does not modify or resume the v1 optimizer. The new
schedule uses 500-update immutable segments, AdamW at 8e-5, EMA 0.999, BF16
with float32 losses, a 16 GiB free-VRAM minimum, and the project-wide 100 GiB
free-disk floor.

```powershell
python -m forge.creature_stage_neural_motion_rollout prepare-production

$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m forge.creature_stage_neural_motion_rollout segment --end-update 500
```

No pilot checkpoint is acceptable merely because its training loss is finite.
It must be evaluated on the same 65 held-out clips and 72-frame prediction-fed
matrix as v1. The immediate success criterion is to retain or improve the
step-3,000 geometry while raising the weakest locomotion/action energy ratios
and beating the copy-previous baseline. If those signals do not improve, the
pilot stops without consuming the remaining schedule.

## Update-500 pilot evidence

The first 500-update checkpoint is sealed at
`outputs/creature_stage_neural_motion_rollout/production_v1/cell_motion_rollout_0000500.pt`.
It is 438,984,575 bytes with file SHA-256
`2b0601913ac3fc29f6135c2ef41073667d3bdd51238760dc692baae733813dac`.
Its model state is
`429ee645f2568be65d10a5a764c9cf72e88ac9bf44883bd2448218f164881d68`
and EMA state is
`ae2f83102a40a56880bb1f757428c0627a658640a1992e49210d85c0997dac90`.
The production contract semantic identity is
`8c62aad4bdb20dcb66c179ff799d3723a12a61478bb7107902443611cc47e989`.

The full CUDA report at
`outputs/creature_stage_neural_motion_rollout/evaluation_pilot500_full_v1`
was independently recreated on CPU across all 65 clips and 4,680 recurrent
frames. Its semantic identity is
`610aee770cf9e32d3b3668cc742332083150ab612495c99c9d909d2eeaf9eeb0`;
the manifest file SHA-256 is
`e29eafc3ae67f9dff658a89ace2e98c874909189d6d4a9921a0e107e6c8106ca`;
the evaluator source identity is
`a938415856c20bba1d80782c7df26de0f4373019e34e3eb0756bb3f81981f4ec`.

| measure | v1 step 3,000 | rollout update 500 | interpretation |
| --- | ---: | ---: | --- |
| position MAE | 0.4982 px | 0.4823 px | improved |
| velocity MAE | 0.1546 px | 0.1329 px | improved |
| weakest clip energy | 0.1100 | 0.1450 | improved, still low |
| aggregate energy | 0.6343 | 0.4980 | below calibrated range |
| appendage/core ratio | 1.2993 | 1.7451 | stronger articulation separation |
| copy-previous improvement | -0.0096 | -0.0089 | still fails |

The objective is redistributing energy rather than uniformly shrinking it:
previously overactive humanoid/machine idles moved closer to target, while the
weakest anomaly and locomotion clips gained energy. Nevertheless, locomotion
as a motion class remains underpowered. Update 500 is therefore a useful but
rejected pilot. One more bounded segment is justified to test the trend; the
branch must stop or be recalibrated if aggregate locomotion and the weakest
clip do not continue improving.
