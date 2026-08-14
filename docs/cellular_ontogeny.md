# Cellular ontogeny v1

Ontogeny turns each immutable symmetry-refined adult into a deterministic developmental program. Children no longer need to appear as fully assembled adults: every cell has an exact birth rank, a real earlier bonded parent, a developmental lineage, an activation stage, three morphogen coordinates, and a differentiation time. Bonds activate only after both endpoints exist.

Six connected stages are compiled for all 45 organisms: zygote, gastrula, organ primordia, larval, juvenile, and adult. Organ primordia are prioritized before terminal appendages and weapons. Approximate left/right partners are biased into nearby cohorts, allowing bilateral chassis and appendages to emerge together without forcing perfect frame symmetry. Plant and anomaly bodies retain their family-appropriate irregularity.

The adult cell, organ, bond, fluid, and genome arrays remain immutable. The new bank is an additive growth schedule over those arrays and supports damage interrupting development naturally.

`CellularOntogenyLab.tscn` runs the programs natively inside the living ecology simulation. New organisms begin as small connected zygotes; cells bud from their actual lineage parents, existing springs unfold the adult shape, organs become functional progressively, and normal damage or environmental stress can interrupt the process. Biomass limits carrying capacity and resource cells regrow, so development and reproduction compete for a finite habitat. `A/D` changes habitat, `G` advances one stage, `Shift+G` grows directly to adult, and `X` changes growth speed. Reproduction inherits the same path, so children are no longer instant full-size clones.

```powershell
python -m forge.cellular_ontogeny build
python -m forge.cellular_ontogeny validate outputs/cellular_ontogeny_v1/cellular_ontogeny_manifest.json
python -m forge.cellular_ontogeny replay outputs/cellular_ontogeny_v1/cellular_ontogeny_manifest.json
```
