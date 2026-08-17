# Neural limb pose v1

This stage moves articulated manipulation below the high-level command layer.
The existing grasper network selects an appendage and predicts reach, force,
brace, release, and throw intent. `NeuralLimbPose` then predicts the internal
inverse-muscle curvature for each live appendage chain.

The model consumes:

- current joint positions and velocities;
- normalized segment lengths and cumulative chain position;
- the filtered physical target;
- family and appendage-kind tokens;
- response, actuation/damage capacity, carried load, side, and bend polarity.

It predicts only curvature around an exact root-to-hand chord. The same
full-skeleton PBD projector used by grounded locomotion retains authority over
the chassis root, hand constraint, bone lengths, planted contacts, and 2.5D
orientation. This is a deliberate ensemble-stage neural replacement, not a
claim that deterministic physics has already been removed.

## Accepted checkpoint

`outputs/creature_stage_neural_limb_pose_v1/production_2400_catalog/runtime.pt`

- parameters: 2,343,747;
- held-out pose MAE: 0.01979 px;
- held-out pose p95: 0.07881 px;
- closed-loop node MAE: 0.09890 px;
- maximum projected bone error: 0.00000143 px;
- maximum endpoint velocity: 1.30857 px/frame;
- maximum endpoint acceleration: 1.25751 px/frame squared.

The approved grounded-controller comparison envelope is 2.3321 px/frame and
2.3297 px/frame squared. All training, provenance, closed-loop, bone, and
motion-floor gates pass. The v9 showcase report binds the exact checkpoint and
its metrics to every rendered artifact.

## Commands

```powershell
python -m forge.creature_stage_neural_limb_pose_v1 `
  --output outputs/creature_stage_neural_limb_pose_v1/production_2400_catalog `
  --updates 2400 --device cuda

python -m forge.creature_stage_manipulation_v1.showcase `
  --output outputs/creature_stage_manipulation_v1/showcase_v9_neural_limb
```
