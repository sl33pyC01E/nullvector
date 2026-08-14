# Structural cellular breeding

The cellular breeding forge makes reproduction structural instead of merely
changing scalar genome traits. It consumes the sealed 36-organism learned
latent evolution bank and emits 45 deterministic physical offspring.

Each offspring has two explicit parents. The forge crosses their 48x48
part/material/emission fields, applies a bounded mutation, requires at least
eight surviving cells from each parent, repairs disconnected raster anatomy,
and re-runs the organ compiler. The resulting child has its own cells, named
organs, eyes, digestive and reproductive systems, conductive fluid network,
spring bonds, health, mass, stiffness, metabolism, and genome.

Coverage is deliberate:

- all 15 unordered pairings of humanoid, animalian, plantlike, anomaly, and
  machine families have exactly three offspring;
- six crossover operators: sagittal, transverse, radial, Voronoi, organ graft,
  and cellular mosaic;
- six mutations: none, budding growth, boundary apoptosis, armor metaplasia,
  bioluminescent shift, and appendage graft;
- 22,933 physical cells, 746 organs, 120 eyes, and 77,829 breakable bonds;
- 627 mutation pixels and 51 explicit connectivity-repair cells;
- exact parent anatomy hashes, operator names, seed, ancestry counts, field
  archive, child anatomy archive, and byte replay.

Build and verify:

```powershell
python -m forge.cellular_breeding build
python -m forge.cellular_breeding validate `
  outputs/cellular_breeding_v1/cellular_breeding_manifest.json
python -m forge.cellular_breeding replay `
  outputs/cellular_breeding_v1/cellular_breeding_manifest.json
python -m forge.cellular_breeding_sync `
  --report outputs/cellular_breeding_sync_report.json
```

Native smoke:

```powershell
C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe --headless `
  --path C:\Users\forre\Documents\neural-game\game `
  res://CellularBreedingLab.tscn -- `
  --cellular-organism-smoke `
  --cellular-organism-report=C:/Users/forre/Documents/neural-game/outputs/cellular_breeding_godot_report.json
```

`CellularBreedingLab.tscn` loads all 45 real child anatomies without Python and
supports the same tear, blast, fluid leakage, feeding, metabolism, healing,
top-down planar motion, surface-fluid diffusion, and reproduction simulation as
the base organism lab. The displayed
parent IDs and operators are source-bound.

The scope remains explicit: the runtime simulates a decoded structural child,
but a later runtime birth clones that child's anatomy and mutates metabolic
traits. Arbitrary live neural decoding is still an offline forge operation;
`runtime_offspring_redecode=false` prevents the UI from claiming otherwise.

## Soft organic symmetry refinement

`forge.cellular_symmetry` is an additive refinement of the 45-child bank. It
does not overwrite the structural breeding output and never deletes or changes
an inherited source cell. Instead, it preferentially completes missing mirror
cells around each child's best bilateral axis:

- chassis completion is strongest for machines and humanoids;
- paired appendages receive a separate owner-aware mirror rule (`left` becomes
  `right`, and vice versa);
- animals and plants receive moderate pressure;
- anomalies retain the weakest prior and lowest growth cap;
- unpaired weapons, emitters, and ornament remain lightly constrained;
- perfect bilateral symmetry is never a hard gate.

Across the current 45 children, mean weighted chassis/appendage/silhouette
symmetry rises from `0.5370884` to `0.6842649`. All 45 improve without reaching
perfect symmetry. The refinement adds 2,535 symmetry cells and 200 connectivity
cells, then re-decodes 25,668 physical cells, 748 organs, 122 eyes, and 85,357
bonds. The before/after/difference sheet is
`outputs/cellular_breeding_symmetry_v1/cellular_symmetry_contact_sheet.png`.

```powershell
python -m forge.cellular_symmetry build
python -m forge.cellular_symmetry validate `
  outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json
python -m forge.cellular_symmetry replay `
  outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json
python -m forge.cellular_symmetry_sync `
  --report outputs/cellular_symmetry_sync_report.json
```

The native `SymmetricOrganismLab.tscn` loads the refined organisms while the
original `CellularBreedingLab.tscn` remains available for direct comparison.
