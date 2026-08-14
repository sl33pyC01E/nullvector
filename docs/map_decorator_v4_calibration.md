# Map decorator v4 CUDA calibration

This isolated package measures whether the proposal-aware neural residual can improve the immutable v4 public-entropy map decorator without sacrificing its near-exact sparse object placement.

The worker evaluates the untouched initialized model on all 576 validation maps and all 24 full-size test sentinels before the first optimizer update. It then runs a bounded BF16 CUDA schedule and evaluates both raw and EMA weights on the same splits. Decal and prop foreground macro-IoU, foreground F1, and rare-class recall must each remain within `1e-7` of baseline for both raw and EMA models, with at least one strict improvement. Legality, hard-empty behavior, proposal confinement, collision avoidance, source provenance, and checkpoint reload are independent hard gates.

The calibration is deliberately not a runtime-integration authorization. A quality failure is retained as valid experimental evidence, while no failed model is promoted.

Run through the access-violation-aware supervisor:

```powershell
python -m forge.map_decorator_production_v4_calibration calibrate `
  --corpus outputs/map_decorator_corpus_v1 `
  --index outputs/map_decorator_production_v2/foreground_index_v2 `
  --output outputs/map_decorator_production_v4_calibration/calibration_100step_v1 `
  --steps 100 --base-channels 48
```

Validate the immutable report and checkpoint closure:

```powershell
python -m forge.map_decorator_production_v4_calibration validate `
  --corpus outputs/map_decorator_corpus_v1 `
  --index outputs/map_decorator_production_v2/foreground_index_v2 `
  --output outputs/map_decorator_production_v4_calibration/calibration_100step_v1
```
