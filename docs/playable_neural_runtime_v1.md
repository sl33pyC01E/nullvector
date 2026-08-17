# Playable neural runtime v1

This is the source-bound bridge between the promoted neural foundation and the playable nature simulation.

It loads the current composite Action-DiT/VAE path and the promoted locomotion, behavior, colony, society, timeline, and counterfactual specialists. Every artifact is checked against the immutable teacher-ensemble manifest before it is exposed to the game.

The living-world scaffold remains authoritative for topology, material conservation, collisions, wounds, organs, feeding, reproduction, and save data. Neural systems provide rasterization, action-conditioned visual prediction, organism motion, intent, physiology, and multiscale coordination. This is the ensemble stage, not the final monolithic student.

Run a bounded loader/render smoke:

```powershell
python -m forge.playable_neural_runtime_v1 --device cpu
```

The live nature demo now uses this release through `PlayableNeuralRuntime.from_release()`.
