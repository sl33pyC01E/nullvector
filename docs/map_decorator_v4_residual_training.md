# Map decorator v4 residual calibration

This package trains a small neural residual around the deterministic v4 public-entropy proposal substrate. It does not ask a neural network to rediscover SplitMix64 proposal locations. Public proposals remain immutable inference inputs, and the decoder still makes off-proposal, illegal, hard-empty, or colliding object placement structurally impossible.

The loss has four distinct responsibilities:

- variant and emission categorical refinement;
- presence classification only on public proposal cells, with extra proposals weighted more heavily;
- object type classification only at authoritative foreground cells;
- log-count calibration and a small residual penalty around the v4 proposal priors.

The CPU smoke is a reproducibility proof, not a visual-quality claim. It performs two updates both continuously and across a serialized step-one interruption. Model tensors, EMA tensors, optimizer state, generator state, CPU RNG state, metrics, authority identities, and source contracts must replay exactly. Validation independently recomputes both steps and rejects a report even when altered metrics are given a new valid self-hash.

The bounded CUDA calibration freezes the categorical core and trains only the decal and prop residual towers. This separation is deliberate: the first diagnostic run showed that allowing the shared categorical core to move improved objects while collapsing emission. That run remains historical evidence and is not an accepted artifact. The frozen-core run preserves variant and emission predictions exactly while retaining the object gains.

Build and validate the immutable proof:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.map_decorator_production_v4_training smoke `
  --corpus outputs/map_decorator_corpus_v1 `
  --index outputs/map_decorator_production_v2/foreground_index_v2 `
  --output outputs/map_decorator_production_v4_training/cpu_resume_smoke_v3

python -m forge.map_decorator_production_v4_training validate `
  --corpus outputs/map_decorator_corpus_v1 `
  --index outputs/map_decorator_production_v2/foreground_index_v2 `
  --output outputs/map_decorator_production_v4_training/cpu_resume_smoke_v3
```

Run and independently replay the bounded calibration:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
python -m forge.map_decorator_production_v4_training calibrate `
  --corpus outputs/map_decorator_corpus_v1 `
  --index outputs/map_decorator_production_v2/foreground_index_v2 `
  --output outputs/map_decorator_production_v4_training/cuda_calibration_v2_frozen_core `
  --steps 64

python -m forge.map_decorator_production_v4_training validate-calibration `
  --corpus outputs/map_decorator_corpus_v1 `
  --index outputs/map_decorator_production_v2/foreground_index_v2 `
  --output outputs/map_decorator_production_v4_training/cuda_calibration_v2_frozen_core
```

## Accepted evidence

The frozen-core calibration is accepted as a bounded experiment and exact-replayed over all 576 validation maps and all 24 full-size test sentinels. All evaluations have hard legality `1.0`, zero immutable semantic changes, zero provenance failures, and the exact expected sample registries.

Foreground macro-IoU, procedural baseline to selected EMA:

| Split | Decal | Prop | Variant | Emission |
| --- | ---: | ---: | ---: | ---: |
| validation | `0.984163 -> 0.995097` | `0.937811 -> 0.989279` | `0.051981 -> 0.051981` | `0.158816 -> 0.158816` |
| test | `0.986561 -> 0.998597` | `0.929559 -> 0.989276` | `0.051244 -> 0.051244` | `0.151344 -> 0.151344` |

The small test decal rare-class recall change (`1.0 -> 0.999680`) remains inside the predeclared `0.002` non-regression tolerance. Prop rare recall improves from `0.925816` to `0.997033`. The report selects EMA, not raw weights.

Evidence identities:

- report file SHA256: `3c107030c3d1c89bbfcabe8a565c117c60ab4d978d7ccd62a317d905917cc7e9`;
- canonical report SHA256: `b12b795faaafec25ebf44f71e80cba016075d2d60097aff6259ecdf7fa255c62`;
- checkpoint SHA256: `ad73a8112f61512abcb07e66ebb09d83327cae2333f70efde94f844dfa4fc86c`;
- training source SHA256: `cbbce630f30e6ca35d9bf63a638109b60075704407fab0120e11e8566510fa81`;
- training contract SHA256: `1e34c1e132161992f4c78e3ed342785825774e5a2f50a9134a52f8e19e985a2f`.

This does not authorize a production schedule or Godot integration. The procedural proposal substrate retains credit for the high baseline; the neural residual has demonstrated bounded conflict suppression only.

The earlier `map_decorator_production_v4_calibration/calibration_100step_v1` and its protected-selection derivative predate the frozen-core contract. They are intentionally source-stale historical diagnostics and must not be promoted. Protected-selection work must be rebound to this accepted checkpoint before its audit can become current again.
