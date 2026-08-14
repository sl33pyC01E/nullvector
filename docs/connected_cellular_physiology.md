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
- immune/repair: a distinct repair core, routed conduits, and two repair
  effectors, spatially separated from every primary organ core.

Membership overlaps deliberately: one pixel cell may be structural tissue, a
vascular conduit, and part of a neural or respiratory route. Each system stores
core, conduit, and exchange/effector roles. Runtime capacity combines surviving
weighted cells, live-bond reachability from the system core, and explicit
dependencies. A destroyed heart therefore collapses circulation and the
dependent respiratory, digestive, neural, locomotor, reproductive and repair
capacities. Brain loss removes deliberate motion and sensing without pretending
that the gut or heart vanished. Gut loss stops nutrient conversion. Respiratory
loss depletes oxygen and causes secondary neural injury.

Physiology v2 makes those paths literal. A system may traverse only cells
declared as its core, conduit, or effector; arbitrary healthy chassis cells can
no longer bridge a cut vessel or motor tract. The reference simulator also
computes a widest-path signal for every member and a per-cell delivery field.
Local healing therefore needs immune, circulation, and digestion delivery at
the wound rather than merely three healthy global counters.

Build, validate, and replay without CUDA:

```powershell
python -m forge.cellular_physiology build
python -m forge.cellular_physiology validate `
  outputs/cellular_physiology_v4/cellular_physiology_manifest.json
python -m forge.cellular_physiology replay `
  outputs/cellular_physiology_v4/cellular_physiology_manifest.json
```

The deterministic `PhysiologyState` is the reference capacity/damage model.
The native projection is built with:

```powershell
python -m forge.cellular_physiology_sync `
  --destination game/generated/cellular_physiology/v11 `
  --report outputs/cellular_physiology_native_sync_v11.json
```

`CellularMotionLab.tscn` consumes that projection alongside the attachment-root
motion v2 bank. Its live status panel reports brain, heart, lung, gut, and
oxygen capacity. Digestion and tissue regeneration are scaled by connected gut,
circulation, and repair capacity; respiratory failure depletes oxygen and
injures neural tissue; low circulation causes systemic damage; neural and
locomotor capacity scale intentional spring forces; reproduction requires a
functional reproductive system. Locomotor force is gated again at each cell by
its graded living motor-system route, so crushing one tract progressively
weakens its downstream appendage and severing it can make that appendage go
limp without freezing an intact limb. Wound clotting also requires local
circulatory and immune delivery at that pixel; a detached fragment cannot
borrow the main body's global capacity. The headless native smoke now executes
all eight organ-core failures on every one of the 45 organisms: 360 failure
cases plus 180 explicit heart, respiratory, digestive, and neural cascade
signatures. It requires the exact source census across all five families
(11 humanoids, 10 animalians, 9 plantlikes, 8 anomalies, and 7 machines) and
proves that non-circulatory brain, lung, and gut injuries retain a living
circulation while their actual dependent capacities fail. It also proves member-restricted
routing and progressive root-to-tip appendage coordinates. Python is not
required at runtime.

Physiology v3 also makes internal fluid authoritative. Circulatory
widest-path delivery is limited by each cell's blood, sap, phase ichor, or
coolant relative to that organism's healthy baseline fill. A drained but still
living vessel therefore transmits a graded perfusion signal; downstream organ
capacity, appendage force, clotting, and repair decline before tissue death.
Fluid diffusion can refill an intact route, while an open wound continues to
lose pressure until local clotting seals it.

Native homeostasis adds reserves without weakening structural damage rules.
The connected v3 organ graph still decides which heart, lung, gut, brain,
sensory, motor, reproductive, and repair cells survive and remain reachable.
Its raw capacities feed a functional state that also tracks oxygen, cellular
energy, and circulatory shock. Brain-core loss therefore incapacitates at once;
lung-core loss begins with intact consciousness but exhausts oxygen over time;
gut-core loss immediately blocks reproductive energy while leaving circulation
alive; and heart-core loss removes perfusion and accumulates shock. Functional
locomotion drives the authored appendage animation, while functional digestion,
immune delivery, circulation, and reproduction gate feeding conversion,
healing/scarring, clotting, and offspring. The Godot smoke records and verifies
this four-lesion time course in addition to the original 360 structural core
failures.

Cell connectivity is now exercised separately from cell death. For every
identity, the native smoke selects one living routed member in circulation,
respiration, digestion, neural, sensory, locomotor, reproductive, and immune systems,
cuts every physical bond at that member, and requires local delivery to fall
from nonzero to exactly zero while the isolated cell remains alive. All 360
identity/system cases must also reduce the appropriate whole-system capacity.
This proves a severed vessel, airway/exchange path, gut route, nerve, sensory
connection, motor tract, reproductive route, or repair conduit cannot continue
functioning as an invisible global counter. Physiology v4 expands this to all
360 identity/system severance cases. The immune organ now has a distinct core,
at least five conduits, and two effectors; its core never overlaps any primary
organ core, and cutting a living immune member reduces repair delivery while
the isolated cell itself remains alive.
