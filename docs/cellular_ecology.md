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
