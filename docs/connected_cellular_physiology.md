# Connected cellular physiology

`forge.cellular_physiology` adds overlapping functional systems to the same
pixel cells and breakable bonds used by the symmetry-refined organism and
neuromuscular motion banks. It does not replace or repaint anatomy.

Every one of the 45 organisms receives eight explicit systems:

- circulation: heart core plus vascular conduits;
- respiration: family-specific lungs, gills, stomata, phase exchange, or
  coolant radiators connected back to the heart;
- digestion: mouth and digestive vacuole linked into circulation;
- neural: brain cells and their life-support connection;
- sensory: eyes/effectors connected to the brain;
- locomotion: brain core, motor paths, and appendage effectors;
- reproduction: reproductive nexus connected to circulation;
- immune/repair: stem or immune seed cells and their vascular route.

Membership overlaps deliberately: one pixel cell may be structural tissue, a
vascular conduit, and part of a neural or respiratory route. Each system stores
core, conduit, and exchange/effector roles. Runtime capacity combines surviving
weighted cells, live-bond reachability from the system core, and explicit
dependencies. A destroyed heart therefore collapses circulation and the
dependent respiratory, digestive, neural, locomotor, reproductive and repair
capacities. Brain loss removes deliberate motion and sensing without pretending
that the gut or heart vanished. Gut loss stops nutrient conversion. Respiratory
loss depletes oxygen and causes secondary neural injury.

Build, validate, and replay without CUDA:

```powershell
python -m forge.cellular_physiology build
python -m forge.cellular_physiology validate `
  outputs/cellular_physiology_v1/cellular_physiology_manifest.json
python -m forge.cellular_physiology replay `
  outputs/cellular_physiology_v1/cellular_physiology_manifest.json
```

The deterministic `PhysiologyState` is the reference capacity/damage model.
The native projection is built with:

```powershell
python -m forge.cellular_physiology_sync `
  --destination game/generated/cellular_physiology/v3 `
  --report outputs/cellular_physiology_v3_sync_report.json
```

`CellularMotionLab.tscn` consumes that projection alongside the attachment-root
motion v2 bank. Its live status panel reports brain, heart, lung, gut, and
oxygen capacity. Digestion and tissue regeneration are scaled by connected gut,
circulation, and repair capacity; respiratory failure depletes oxygen and
injures neural tissue; low circulation causes systemic damage; neural and
locomotor capacity scale intentional spring forces; reproduction requires a
functional reproductive system. The headless native smoke destroys the brain
core of a diagnostic offspring and proves that neural/locomotor capacity reaches
zero while circulation remains functional. Python is not required at runtime.
