# Neural grounded locomotor controller

This stage replaces direct per-cell pose prediction with a neural policy over the validated grounded physics scaffold.

## Boundary

The policy predicts:

- contact intent for each inherited appendage;
- antagonistic muscle activation for each articulated joint;
- a body-drive diagnostic.

The deterministic scaffold executes those controls using persistent world-space contact anchors, position-based edge constraints, inertia, muscle forces, ground reaction, gravity, and the 2.5D vertical-axis lock. Locomotor behavior is attached to appendage kind rather than family: legs step, roots drag, wheels roll, and anomalies retain field propulsion. A rare graft therefore carries its locomotor mechanics into any host family.

This is an intentional stage-two ensemble boundary. It is substantially safer and more physically meaningful than asking a cell transformer to rediscover constraint solving. A future student can distill the controller and solver trajectories after the ensemble is approved.

## Lineage

The controller is bound to the component-aware checkpoint at `outputs/creature_stage_neural_grounded_components/pilot_0450/component_grounded_motion_0001650.pt`. That checkpoint is itself descended from the runtime-honest cyclic model and ultimately from the sealed rollout update-1000 authority.

Training uses 60 deterministic physics-developed organisms, balanced across five families and deliberately oversampling rare cross-family locomotor grafts. Evaluation remains completely separate: five untouched grafted organisms, one per family, 72 prediction-fed phases each.

The controller pools the parent cell prediction, current cell state, runtime contact features, and cell anatomy into eight appendage tokens. Explicit appendage metadata covers kind, side, segment count, phase, root, and endpoint. Muscle tokens add owner, joint, antagonist, strength, side, and phase features.

## Final reviewed result

Checkpoint:

`outputs/creature_stage_neural_grounded_controller/pilot_0800/grounded_controller_0000800.pt`

Reviewed evaluation:

`outputs/creature_stage_neural_grounded_controller/evaluation_0800_final_verified`

Held-out metrics:

- contact F1 / IoU: 1.000 / 1.000;
- muscle activation MAE: 0.089197;
- cell trajectory MAE: 0.036530 px;
- appendage trajectory MAE: 0.046086 px;
- node trajectory MAE: 0.059675 px;
- loop seam: 0.000107 px;
- maximum planted-contact slip: 0.022963 px;
- maximum edge strain: 0.076020;
- vertical-axis deviation: 2.973 degrees;
- mean travel ratio: 0.980819;
- neural improvement over zero-controller ablation: 89.592%.

All twelve declared gates pass and the result has been visually inspected. Exact replay rebuilds the parent-derived feature corpus, re-predicts raw and EMA policies, reruns controlled and ablated physics, and byte-compares the published cycle archive.

## Rasterization

No rasterizer is used to score or conceal this result. The old 35.6M-parameter VAE remains a provenance-bound baseline only; its visual quality is not accepted as final. Rasterizer development should resume only after human approval of the morphology and physics motion, using structured cell, organ, skeleton, material, and emission fields as conditioning rather than unconstrained RGB reconstruction.

## Commands

```powershell
python -m forge.creature_stage_neural_grounded_controller --output outputs/creature_stage_neural_grounded_controller/pilot_0800 --updates 800 --device cuda
python -m forge.creature_stage_neural_grounded_controller.evaluation --checkpoint outputs/creature_stage_neural_grounded_controller/pilot_0800/grounded_controller_0000800.pt --output outputs/creature_stage_neural_grounded_controller/evaluation_0800_final_verified --device cuda --visually-inspected
```
