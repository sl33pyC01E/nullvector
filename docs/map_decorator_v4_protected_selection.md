# Protected rare-proposal selection

The 100-step v4 residual calibration improved ten of twelve sparse-object metrics but narrowly reduced rare decal recall. This package repairs that failure without retraining, changing the public proposal substrate, or weakening the gate.

The protected selector derives the rare decal class from the frozen procedural baseline report. It may restore that class only where the trained EMA decoder chose no decal, chose no prop, the exact public class proposal exists, class legality is true, and the cell is not hard-empty. It never overwrites a neural object choice. The ordinary v4 legality and off-proposal assertions run again after restoration.

Acceptance compares the resulting full 576-map validation and 24-map test metrics against both the procedural baseline and the trained EMA. Every decal/prop IoU, F1, and rare recall metric must be non-regressing against each reference, with at least one strict improvement. The audit also produces a six-theme target/trained/protected contact sheet.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m forge.map_decorator_production_v4_selection build `
  --calibration outputs/map_decorator_production_v4_calibration/calibration_100step_v1 `
  --corpus outputs/map_decorator_corpus_v1 `
  --index outputs/map_decorator_production_v2/foreground_index_v2 `
  --output outputs/map_decorator_production_v4_selection/protected_selection_audit_v1
```

The audit authorizes the selection policy as a quality candidate. It does not silently integrate the checkpoint into Godot or replace prior map artifacts.
