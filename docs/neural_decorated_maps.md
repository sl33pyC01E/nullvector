# Neural decorated map bank

This additive compiler applies the accepted v4 EMA decorator and protected rare-proposal selector to the six topology-v2 ForgeLab maps. It never alters terrain, hazards, navigation, points, protected backbone, required clearance, or decoration-forbidden arrays.

The selected variant controls 8px terrain micro-patterns. Selected decal and prop classes resolve through the authoritative theme catalog, while selected emission levels scale additive terrain and object light. Hazards remain driven by the original map semantics.

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
