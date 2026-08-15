# NULLVECTOR // Native-cell physiology transformer

## Purpose

This is the second learned subsystem for the rebooted playable layer. The
motion transformer predicts where each physical cell moves. The physiology
transformer predicts what happens to those cells, their organ capacities, and
their top-down fluid puddles after damage, healing, severing, and targeted
organ ablation.

It is trained against the strict native intervention authority at
`outputs/creature_stage_intervention_corpus_v1_final_a`: 20 chassis, all five
families, nine interventions, 180 frames per intervention, 7,304,580 cell
samples, and up to 160 live fluid particles per frame. The learned model never
changes which cells exist in the chassis; cell birth/fracture remains a later
topology model.

## State and conditions

Every organism uses up to 384 actual cell tokens. The 61 immutable features
per cell encode position, tissue, organ identity, side, centrality, appendage,
and eight heritable genes. The recurrent cell state contains displacement,
health, and viability. A separate ten-value organism state contains integrity,
neural, circulation, respiration, digestion, sensory capacity, energy,
hydration, death, and fluid occupancy.

The fluid branch carries 160 ordered puddle tokens with position, velocity,
radius, lifetime, and presence. Positions and velocities remain on the
top-down ground plane—there is no screen-down gravity. Conditioning identifies
family, one of twenty morphotypes, intervention type, continuous time, damage
and heal pulses, and post-event state.

## Architecture

The production geometry has 16,711,701 trainable parameters:

- eight 320-wide cell-transformer blocks;
- masked global attention for organ-to-organ effects;
- explicit eight-neighbor graph aggregation for local tissue propagation;
- a pooled organism-capacity head;
- a four-block, 192-wide recurrent fluid point branch with stable slot queries.

The joint loss supervises cell displacement, health, viability, ten organism
capacities, fluid presence and values, graph-relative health, and exact zero
outside occupied cells. This prevents a low average health error from hiding
dead-organ mistakes or missing leaks.

## Current proof

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.creature_stage_neural_physiology model-info
python -m forge.creature_stage_neural_physiology smoke `
  --output outputs/creature_stage_neural_physiology/smoke_cpu_v1 `
  --steps 4
python -m forge.creature_stage_neural_physiology validate-smoke `
  --output outputs/creature_stage_neural_physiology/smoke_cpu_v1
python -m forge.creature_stage_neural_physiology prepare-production
```

The CPU model is deliberately small and trains a fixed five-family batch of
real post-intervention states. It proves the anatomy/organ/fluid tensor path,
all gradients, deterministic artifact validation, and the substantial
production geometry. It does not claim learned long-rollout quality.

The current four-update proof reduced joint fixed-batch loss from 0.944463015
to 0.508713126 while health loss fell from 0.170996353 to 0.048000634 and
fluid-presence loss from 0.777813792 to 0.468835264. All values and gradients
are finite and predictions remain exactly zero outside occupied cells.

| evidence | SHA-256 |
| --- | --- |
| model source | `e302f78f5989b21c52320cec5ed8ed6c5d8eed2cdf231f442018d37424a2bd68` |
| smoke semantic | `9dc74bbf2975a75e643198cf0de861ef3b249d97d87d1ee2c661eabd871915c9` |
| smoke model state | `539fc853ed644da76cadc2b26ec04f3aee6d84f6b15a6448f863b515bc519d69` |
| smoke manifest file | `154bca63d835330a6af0547a1b749b72609cf05a73df50105a5c8b60ed80b197` |
| smoke checkpoint file | `0f44e86be004bf81dafaefbd3a76bdbcac7059e738687c15a034e85fef922c15` |
| production contract semantic | `081e9eab7a4db17c260b00d145714a15751b43d92295223a998ceb83e69ae52a` |
| production contract file | `e4e7df9f8b92ff1ee0ba9f3962c3cc576d519cff844e15b769b9bd5cefe56588` |

The production contract declares 16,000 family-balanced updates in immutable
800-step segments, EMA 0.9995, AdamW 2e-4, deterministic BF16, and a 16 GiB
free-VRAM gate. CUDA training is intentionally not started while another job
occupies the RTX 4090.

## Promotion gates still required

1. exact segmented optimizer/RNG/EMA resume and native-crash containment;
2. prediction-fed 180-frame validation for all 5 families x 9 interventions;
3. target-organ ablations must causally reduce the correct capacity without
   collapsing unrelated systems;
4. wounds leak and diffuse, healing closes wounds, cuts remove the intended
   regions, and neural destruction changes behavior/death timing;
5. sealed-test success, ONNX parity, and native runtime parity;
6. combined motion+physiology rollouts without chassis escape or spawn-time
   terminal cascades.

Until those gates pass, the deterministic physiology scaffold remains runtime
authority and this model remains a Stage 2 replacement candidate.
