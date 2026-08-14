# Cellular ecology v1

This additive forge milestone gives the living cellular organisms a deterministic environment rather than a collection of hand-placed food pellets. It consumes the six strict topology-v2 map packs and the immutable soft-symmetry organism bank.

Each 48×48 map cell carries seven bounded float32 ecological fields: nutrient, moisture, light, temperature, toxicity, energy, and biomass. A five-channel suitability tensor describes the local niche for humanoid, animalian, plantlike, anomaly, and machine organisms. Resource cells are selected deterministically outside every authoritative `decoration_forbidden` cell and outside all hazards. The compiler never mutates terrain, routes, mission clearances, or topology masks.

Family ecology is intentionally asymmetric even when bodies are approximately bilateral. Plantlike organisms prefer light, moisture, growth and nutrients; animalian organisms prefer food, water, temperate safe ground; anomalies can exploit energy and toxic regions; machines favor crystal/charge and drier terrain; humanoids remain generalists. Four candidate resource nodes per family are requested per map, with a hard minimum of three and deterministic spatial separation. Nodes have capacity and biomass-driven regrowth rates so metabolism and reproduction can later draw from carrying capacity instead of infinite food.

Build and verify:

```powershell
python -m forge.cellular_ecology build
python -m forge.cellular_ecology validate outputs/cellular_ecology_v1/cellular_ecology_manifest.json
python -m forge.cellular_ecology replay outputs/cellular_ecology_v1/cellular_ecology_manifest.json
```

The bank is exact-replayable, source-bound to both maps and organisms, schema validated, canonical JSON, and guarded by the repository's 100 GiB free-space floor.

`CellularEcologyLab.tscn` is the native interactive exhibit. It layers the ecological fields over the existing spring-physics organism lab, seeds all 20 resources for the selected biome, regrows depleted nodes, applies temperature/toxicity stress to living cells, and makes local biomass a carrying-capacity requirement for reproduction. `A/D` changes among the six habitats. Motion, tissue fracture, fluid leakage, feeding, and organ failure remain the same physical systems used by the organism and neuromuscular labs; the ecology scene adds no Python runtime dependency.

The current native projection binds the ecology catalog to the exact organism,
motion-v4, connected-physiology, and trauma bundle identities. Resource use is
family-specific instead of a universal food pellet:

- humanoids are broad generalists and can exploit organic or fabricated nodes;
- animalians strongly prefer organic animal/plant niches and reject machine or
  anomaly substrates;
- plantlike cells absorb local light, moisture, nutrients, and compatible
  regenerative nodes without walking toward food;
- anomalies metabolize energetic/toxic gradients and phase-resource nodes;
- machines acquire charge/material from machine and anomaly nodes and prefer
  lower-moisture energetic terrain.

Mobile organisms use their live neural and locomotor capacities to acquire a
weak resource-seeking impulse toward the best compatible nonempty node. Brain,
respiratory, circulatory, or locomotor injury therefore degrades ecological
movement through the same connected organ graph. Reproduction is gated by the
family-specific suitability field and local carrying capacity, which prevents
the earlier universal rapid-reproduction collapse.

The native ecology exhibit now begins with one organism from every family and
runs an independent deterministic behavior state for each. A healthy mobile
organism uses the original family-specific locomotion clip while approaching a
resource, then commits to a complete feeding/tool/field action nearby.
Plantlike organisms alternate stomatal breathing and tropic sway; anomalies
periodically cast field effects; machines perform extraction and diagnostic
actions. The state selector also reads the connected physiology graph:
injury produces a hit reflex, respiratory collapse produces distress, severed
senses or partial neural/locomotor networks produce confusion, and loss of the
brain or heart removes intentional actuation and selects the terminal state.
These are deterministic biological policies driving the existing authored
motion bank, not a claim of a learned behavior model.

The behavior policy now consumes reserve-aware functional physiology rather
than the structural graph alone. Losing the brain produces an incapacitated
sleep state before the delayed terminal cascade; losing respiratory exchange
selects a distress action while the remaining oxygen reserve still supports
consciousness; destroying the digestive conversion core makes a nearby
resource physically unassimilable; and severing sensory effectors collapses
perception range even if the shared neural core survives. Local locomotor and
circulatory delivery still scale every cell's authored appendage stroke, so a
partially damaged leg limps while a disconnected one goes limp. The native
smoke physically lesions heart, lung, gut, brain, and sensory paths and requires
five distinct behavior outcomes, including zero gut intake and an eight-pixel
blind sensory radius.
