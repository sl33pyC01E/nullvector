# Neural map latent-prior calibration

`forge.map_topology_neural_prior_training` trains the compact masked-token prior
over the frozen 3,096-map latent corpus. It is an isolated research stage: it
does not publish raw maps, invoke the deterministic topology compiler, repair
generated maps, or modify Godot assets.

## Frozen authorities

- Latent corpus: `outputs/map_topology_neural_prior_corpus/v1`
- Latent corpus manifest file SHA-256:
  `12ae282fe1d89f4b8f5c87d0d5acf1a8eddf7ab15cd2d32031a6bf7ba1cc3b96`
- Latent corpus semantic SHA-256:
  `01df481c1b3300e41c0e9a70153679e48a9483fd30b0ac3b4e800cff3d198359`
- Latent corpus identity SHA-256:
  `bbcce0606f12d04d53e15e50c16852a8ee3d0e7262146e4c85c5965cf10f4d56`
- Frozen prior source SHA-256:
  `76fcbce48e1ce20f5e1f28c20a38cc9c9d8c98be2cedccd221e7f95bb6145e15`
- Training implementation SHA-256:
  `b54ca3e2a863b9abba8731b8286f86c688e600c26f99e84d6ffa0829e42dc8b1`

Training reads only the 2,496 train-split samples. Evaluation is fixed to a
theme/shape-stratified 48-map validation registry and all 24 test sentinels.
The split and sample registries are hashed into the checkpoint and report.

## Selected calibration

The selected bounded calibration is:

`outputs/map_topology_neural_prior_training/calibration_500step_v1`

- Report SHA-256:
  `48ac41a3e38058bb791f9d8df5e0fb2f508d17722a852392d2d97765a6cfd4d5`
- Checkpoint SHA-256:
  `bb18b1c98474abe51e852a46ebe6c47de773814657d27302ec177ff495cd0475`
- EMA validation accuracy: `0.22821403752605976`
- EMA validation macro structured-mask accuracy: `0.2240508651342269`
- EMA test accuracy: `0.23404997094712376`
- EMA test macro structured-mask accuracy: `0.23824324597045116`
- Relative loss improvement: `0.5231774868676989`
- Peak reserved CUDA memory: `419,430,400` bytes
- Training time on the RTX 4090: `11.552625800002716` seconds

Every predeclared gate passed and a separate fresh-process metric replay matched
exactly. The 200-step candidate also passed, but the 500-step candidate improved
both held-out splits and is the selected checkpoint.

The two-step `probe_2step_v2` is pipeline evidence only. `probe_2step_v1` is
preserved as stale evidence from a validator-ordering defect discovered during
publication; it must not be promoted.

## Commands

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'

python -m forge.map_topology_neural_prior_training calibrate `
  --corpus outputs/map_decorator_corpus_v1 `
  --latent-corpus outputs/map_topology_neural_prior_corpus/v1 `
  --output outputs/map_topology_neural_prior_training/calibration_500step_v1 `
  --steps 500 --validation-samples 48 --test-samples 24

python -m forge.map_topology_neural_prior_training validate `
  --corpus outputs/map_decorator_corpus_v1 `
  --latent-corpus outputs/map_topology_neural_prior_corpus/v1 `
  --output outputs/map_topology_neural_prior_training/calibration_500step_v1
```

Publication is immutable and enforces the 100 GiB disk floor. Checkpoints use
bounded `weights_only` loading, exact source/corpus/config binding, canonical
sidecars, finite tensor checks, model/EMA hashes, RNG state capture, and exact
history/evaluation schemas.

## Claim boundary and next stage

This checkpoint demonstrates masked latent reconstruction under fixed
conditioning. It is not yet evidence of useful free generation or valid map
topology. The next isolated stage must:

1. sample a theme-balanced raw latent bank from the selected EMA;
2. decode through the frozen VQ codec without changing the raw samples;
3. compile/repair through the deterministic topology contract;
4. report pre-repair validity, repair-cell fraction, mission connectivity,
   diversity, exact replay, and held-out condition adherence;
5. reject promotion if generation collapses or repair cost exceeds calibrated
   procedural-reference bounds.

