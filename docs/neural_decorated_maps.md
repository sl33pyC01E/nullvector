# Neural decorated map bank

This additive compiler applies the accepted v4 EMA decorator and protected rare-proposal selector to the six topology-v2 ForgeLab maps. It never alters terrain, hazards, navigation, points, protected backbone, required clearance, or decoration-forbidden arrays.

Only the two accepted sparse object heads cross the runtime boundary: neural decal and prop classes resolve through the authoritative theme catalog. The weak, unaccepted neural variant and emission heads are retained only in provenance diagnostics. Terrain micro-patterns use the deterministic semantic variant, and emission is recomputed from exact terrain capability plus the selected object classes. This hybrid authority is explicit and fail-closed rather than silently presenting weak neural heads as production quality. Hazards remain driven by the original map semantics.

Compilation runs each map inference and render twice and requires exact field and pixel replay. The runtime projection is a nearest-filtered PNG atlas plus JSON index; checkpoints, NumPy, Python, and CUDA never enter the shipped game assets. The accompanying compressed field archive is audit-only.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m forge.neural_decorated_maps build `
  --selection-audit outputs/map_decorator_production_v4_selection/protected_selection_audit_v1 `
  --calibration outputs/map_decorator_production_v4_calibration/calibration_100step_v1 `
  --corpus outputs/map_decorator_corpus_v1 `
  --index outputs/map_decorator_production_v2/foreground_index_v2 `
  --maps outputs/maps_v2_forge_lab `
  --output outputs/neural_decorated_maps_v1
```

## Native Godot lab

The verified bank is projected into a deliberately small runtime bundle: one
canonical JSON catalog and one nearest-filtered PNG atlas. The projection
revalidates the source bank, runs twice, compares exact bytes, enforces the
100 GiB disk floor, and publishes atomically. It ships no checkpoint, NumPy
archive, Python module, or CUDA dependency.

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.neural_decorated_map_sync `
  --source outputs/neural_decorated_maps_v1_verified `
  --report outputs/neural_decorated_map_sync_report_v1_1.json

C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe `
  --headless --path C:\Users\forre\Documents\neural-game\game --import

C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe `
  --headless --path C:\Users\forre\Documents\neural-game\game `
  res://NeuralDecoratedMapLab.tscn -- `
  --neural-decorated-map-smoke `
  --neural-decorated-map-report=C:/Users/forre/Documents/neural-game/outputs/neural_decorated_map_godot_report_v1_1.json
```

`game/NeuralDecoratedMapLab.tscn` is additive; `Arena.tscn` remains the main
scene. The lab exhaustively constructs all 90 atlas frame regions across six
themes and eight layer views. Its smoke gate checks atlas identity and bounds,
nearest filtering, exact field authority, runtime census, and the Python-free
boundary. Arrow keys switch themes and layers; space pauses animated hazards.
