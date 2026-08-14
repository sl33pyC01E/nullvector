# Cellular trauma, repair, and fragment fate

`forge.cellular_trauma` extends the 45 symmetry-refined, connected-physiology
organisms without repainting or replacing a cell. Every physical pixel receives
a deterministic healing class plus clotting, scarring, and regrowth weights;
every spring bond receives repair and weak magnetic-reconnection weights.

The reference and native contracts now model:

- diffuse internal-fluid loss from damaged cells and open bond endpoints;
- circulation/immune-dependent clot accumulation that progressively seals a
  leak instead of stopping it instantly;
- energy-consuming health recovery with local scar deposition;
- scars as a persistent visual and mechanical state rather than erased damage;
- weak attraction between the two living endpoints of a recently torn bond;
- distance- and time-bounded reconnection that leaves a scar;
- persistent detached-component age and family-specific terminal fate;
- humanoid and animalian fragments degrading into biomass;
- plantlike fragments becoming polyps, anomaly fragments becoming phase
  polyps, and machine fragments becoming autonomous module polyps when large
  enough;
- the existing connected heart, respiratory, digestive, brain, sensory,
  locomotor, reproductive, and immune capacities continuing to respond to the
  same live cells and bonds.

Family profiles are tendencies rather than geometry rules. Humanoids clot
quickly and scar strongly but have a short reconnection window. Plants regrow
well and can preserve small fragments. Anomalies have the longest phase
reconnection window. Machines magnetize strongly and preserve sufficiently
large modules. The underlying morphology and symmetry banks remain immutable.

Build and exact replay:

```powershell
python -m forge.cellular_trauma build
python -m forge.cellular_trauma validate outputs/cellular_trauma_v3/cellular_trauma_manifest.json
python -m forge.cellular_trauma replay outputs/cellular_trauma_v3/cellular_trauma_manifest.json
```

Native projection:

```powershell
python -m forge.cellular_trauma_sync \
  --destination game/generated/cellular_trauma/v7 \
  --report outputs/cellular_trauma_sync_report.json
```

`CellularMotionLab.tscn` consumes the resulting JSON directly. The native smoke
proves a torn bond is magnetically rejoined with scar tissue, proves a detached
plant organ crosses its reconnection window into a polyp, and still proves the
brain-damage locomotor cascade across the connected physiology overlay. No
Python runtime is needed by Godot.
