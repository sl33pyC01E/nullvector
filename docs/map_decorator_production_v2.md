# Map Decorator Production v2

V2 is an experimental foreground-factored map decoration model. It is not a
production-integrable decorator yet. Its immutable topology-v2 corpus and
foreground index are valid, but every bounded 100-step calibration failed the
predeclared held-out quality gate. No long training run or Godot integration is
authorized from these results.

## What is robust now

- The 3,096-map corpus is split by full-map identity and the 216-shard
  foreground index validates with no duplicate or split leakage.
- Object heads explicitly factor presence from foreground type.
- EMA uses a bias-safe, checkpointed inverse-time warm start. Its update count
  must equal `global_step`, so interrupted two-epoch segments resume exactly.
- Half of every training batch is deterministically fully masked, matching the
  actual generation boundary rather than training only on partial corruption.
- Sparse object decoding uses the learned count-calibration signal: summed
  legal presence probabilities determine per-head quotas, strongest legal
  cells are selected, collisions are resolved by confidence, and lost quota is
  backfilled without decal/prop overlap.
- Calibration records both live-model and EMA full-split metrics. Neither can
  hide collapse behind empty accuracy, legality, or a composite score.

## Calibration conclusion

The final count-aware run recovered nonempty predictions, proving that the
earlier empty output was partly a decoder mismatch. On validation it reached
decal IoU/F1/rare recall `0.02385 / 0.04659 / 0.10273` and prop
`0.00751 / 0.01667 / 0.10587`. Test behavior was similar. However both heads
over-predicted foreground density, decal F1 remained below its gate, and prop
IoU/F1 remained weak. Raw and EMA metrics closely agreed, ruling out EMA as the
remaining cause.

The correct handoff is therefore `calibration_failed_quality`, not a weak model
promotion. The next model revision should improve sparse localization and
count calibration on full maps without tuning thresholds against the held-out
or sentinel splits. Any new revision needs a new source hash, fresh bounded
calibration, and the same unchanged quality gates before longer CUDA training.

## Verification

```powershell
C:\Users\forre\AppData\Local\Programs\Python\Python312\python.exe -m pytest `
  tests\test_map_decorator_production_v2.py `
  tests\test_map_decorator_production_v2_recovery.py -q
```

The comparison evidence is preserved at
`outputs/map_decorator_production_v2/calibration_comparison_20260813.json`.
