# Causal neural grounded feedback v2

This controller replaces the prerecorded locomotion schedule at the gameplay boundary. Every frame it reads current appendage terminal positions and velocities, prior contact state, phase, inherited family and traits, and the live muscle catalog. A 3,503,107-parameter network predicts muscle activation, contact state, and an interpretable drive intention. Contact decisions and muscles execute through the accepted grounded PBD skeleton; the drive head cannot inject ungrounded translation.

The accepted CUDA run is `outputs/creature_stage_neural_grounded_feedback_v2/production_3000_v2/`. Its held-out causal rollout retains 99.04%–100.33% of reference displacement across the five families. Contact F1 is 0.99731, muscle MAE is 0.02173, mean closed-loop node error is 0.00743 px, maximum contact slip is 0.02011 px, maximum strain is 0.10569, vertical drift is 2.9835 degrees, and loop seam error is 0.000107 px. All declared gates pass.

The deterministic pieces are deliberate structural scaffolding: morphology, bone constraints, collision/contact resolution, gravity, and force integration. The neural policy owns live contact selection and muscle commands. This division lets the network be causally tested and ablated without letting a raster predictor silently violate anatomy.

Run focused verification:

```powershell
python -m pytest tests/test_creature_stage_neural_grounded_feedback_v2.py tests/test_creature_stage_manipulation_v1.py -q
```

Rebuild the full integrated gallery:

```powershell
python -m forge.creature_stage_manipulation_v1.showcase --output outputs/creature_stage_manipulation_v1/showcase_v10_full
```
