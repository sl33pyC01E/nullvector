# Neural grounded target field

This model removes the remaining call to the procedural locomotion pose from
the grounded neural rollout. It predicts three coupled fields from organism
anatomy, traits, live state, and gait phase:

- periodic appendage terminal targets;
- muscle activation;
- ground-contact state.

The target path uses immutable appendage geometry and an eight-harmonic local
phase code. Live state drives muscles and contacts but cannot feed accumulated
rollout error back into the periodic target. A physical projector still owns
bone length, contact anchors, collision safety, and the vertical 2.5D chassis
lock.

## Accepted runtime

`outputs/creature_stage_neural_target_field_v1/production_6000_v11_tail/runtime.pt`

- 3.78 million parameters
- terminal MAE: 0.074 px
- terminal p95: 0.199 px
- contact F1: 0.9988
- closed-loop advance: 0.801–1.036x the procedural authority
- maximum edge strain: 0.055
- maximum contact slip: 0.016 px
- maximum vertical drift: 2.62 degrees
- maximum loop seam: 0.000114 px

The exhaustive audit runs every reviewed base and grafted chassis twice in an
isolated process. All ten pass. The training bank, rejected checkpoints,
accepted checkpoint, per-chassis results, and aggregate report are additive;
none overwrite earlier evidence.

## Commands

```powershell
# Validate the reusable training bank
python -m forge.creature_stage_neural_target_field_v1.bank validate `
  outputs/creature_stage_neural_target_field_v1/training_bank_v2_modular

# Train a new candidate
python -m forge.creature_stage_neural_target_field_v1 `
  outputs/creature_stage_neural_target_field_v1/CANDIDATE `
  --updates 6000 --device cuda `
  --bank outputs/creature_stage_neural_target_field_v1/training_bank_v2_modular

# Audit all reviewed chassis in isolated workers
python -m forge.creature_stage_neural_target_field_v1.audit `
  outputs/creature_stage_neural_target_field_v1/CANDIDATE/runtime.pt `
  outputs/creature_stage_neural_target_field_v1/CANDIDATE_AUDIT
```

This is a locomotion authority, not a whole-creature world model. Grasping,
feeding, damage, physiology, ecology, and rendering remain separate members of
the current ensemble.
