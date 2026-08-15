# Developmental creature authority

This package is the human-review authority for creature construction and locomotion before another neural motion run is allowed. It is intentionally procedural: the job is to make anatomy understandable, editable, and falsifiable before a model learns it.

## Construction contract

The five creature families are priors, not mutually exclusive classes. A genome stores a five-value family simplex, global continuous traits, a component graph, appendage programs, and optional parent identities. The review bank pairs each base prior with a grafted organism to prove that components and their local traits can cross family boundaries.

Components include chassis regions, sensory crowns, mouths, digestive and respiratory organs, circulators, neural clusters, armor, generators, storage, and anomaly orbitals. Their overlapping developmental fields are normalized at every cell. Cells grown around appendage-only support inherit their nearest component, so no living material is left without trait authority.

The current continuous trait field contains size, symmetry, segmentation, stiffness, elasticity, bone and muscle density, muscle strength, neural density, vascularity, metabolism, regeneration, grip, sensory range, and phase coherence. The fields are designed to become conditioning channels for later neural models.

## Skeleton and muscle contract

Every organism owns a connected load-bearing graph. Each appendage contributes a root joint followed by one to five segments. Paired appendages must be reciprocal, share a type and root component, and occupy opposite chassis sides. Symmetry is therefore a developmental bias at the chassis/appendage level, not a pixel-level hard constraint.

Every articulated appendage joint receives its own antagonistic flexor/extensor pair. The review renderer shows bone in ivory, the two actuator channels in red/cyan, and planted contacts in green. Living cells are skinned to their three nearest skeleton nodes. No animation rotates, flips, or mirrors the whole organism.

The current locomotion authority uses planted and swing targets plus a signed-bend FABRIK reference pose. A recurrent dynamics pass then applies inertia, per-joint antagonistic moments, weighted chassis masses, edge-length constraints, and planted contacts until it reaches a deterministic periodic limit cycle. It exists to establish good motion geometry. A later neural controller should predict actuator contraction, grip, posture, gaze, and action intent; it should not directly teleport raster cells.

## Human-review bank

Build and replay the additive bank:

```powershell
python -m forge.creature_stage_developmental build-review --output outputs/creature_stage_developmental/review_v4
python -m forge.creature_stage_developmental validate-review --output outputs/creature_stage_developmental/review_v4
```

The bank contains:

- `developmental_contact_sheet.png`: base and grafted morphology, cell fields, skeletons, and muscles.
- `developmental_locomotion.gif`: a 72-frame human-review loop.
- `developmental_locomotion.mp4`: the same loop in a compact video artifact.
- `specimens/*.npz`: exact cellular, developmental-field, skeleton, and muscle arrays.
- `review_manifest.json`: strict schema, source binding, exact semantic/frame replay hashes, loop metrics, and artifact hashes.

## Current limits

This is not yet the final creature generator or physics engine. It uses ten authored genomes, a two-dimensional constraint graph, kinematic contact intent, recurrent damped dynamics, and linear skinning. It approximates relative mass, joint moments, skeletal strain, and contact, but does not yet simulate tendon rupture, joint breakage, fluid pressure, or neural injury. Those belong in the next authority iteration after the chassis and locomotion silhouettes are approved. GPU training remains paused until that review is complete.
