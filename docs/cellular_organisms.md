# Cellular organisms

This subsystem compiles every accepted 48×48 neural categorical sprite into a living, destructible anatomy. One non-aura visible source pixel becomes one physical cell. The source part, material, and emission fields remain immutable authority; the compiler adds tissue, organ, fluid, genome, metabolism, and breakable-bond layers.

Every species receives circulatory, neural, digestive, reproductive, sensory, and integument systems. Family-shaped appendages become named organs. Eyes and a mouth are explicit flagged cells. Blood, hemolymph, sap, phase ichor, or coolant occupies a closed conductive network and leaks after cell or bond failure.

The runtime contract supports spring/constraint physics, fracture, cell ablation, internal-fluid diffusion, bleeding, hunger, feeding, energy conversion, regeneration, reproductive readiness, offspring energy transfer, and deterministic heritable mutation. Aura pixels stay presentation effects and never become collision tissue.

Build and replay:

```powershell
python -m forge.cellular_organism build --generation outputs/production_handoff_v2/final_best_stratified80_bank_attempt1/generation_manifest.json --style outputs/multifield_style/final_best_stratified80_v3/style_manifest.json --output outputs/cellular_organism_v1
python -m forge.cellular_organism validate outputs/cellular_organism_v1/cellular_organism_manifest.json
python -m forge.cellular_organism replay outputs/cellular_organism_v1/cellular_organism_manifest.json
```

The Python `OrganismState` is a deterministic reference, not a gameplay dependency. Native Godot receives a JSON projection and implements the same concepts without Python in the shipped game.
