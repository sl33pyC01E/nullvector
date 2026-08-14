# Map decorator v4 residual-training foundation

This package trains a small neural residual around the deterministic v4 public-entropy proposal substrate. It does not ask a neural network to rediscover SplitMix64 proposal locations. Public proposals remain immutable inference inputs, and the decoder still makes off-proposal, illegal, hard-empty, or colliding object placement structurally impossible.

The loss has four distinct responsibilities:

- variant and emission categorical refinement;
- presence classification only on public proposal cells, with extra proposals weighted more heavily;
- object type classification only at authoritative foreground cells;
- log-count calibration and a small residual penalty around the v4 proposal priors.

The CPU smoke is a reproducibility proof, not a visual-quality claim. It performs two updates both continuously and across a serialized step-one interruption. Model tensors, EMA tensors, optimizer state, generator state, CPU RNG state, metrics, authority identities, and source contracts must replay exactly. Validation independently recomputes both steps and rejects a report even when altered metrics are given a new valid self-hash.

Build and validate the immutable proof:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m forge.map_decorator_production_v4_training smoke `
  --corpus outputs/map_decorator_corpus_v1 `
  --index outputs/map_decorator_production_v2/foreground_index_v2 `
  --output outputs/map_decorator_production_v4_training/cpu_resume_smoke_v1

python -m forge.map_decorator_production_v4_training validate `
  --corpus outputs/map_decorator_corpus_v1 `
  --index outputs/map_decorator_production_v2/foreground_index_v2 `
  --output outputs/map_decorator_production_v4_training/cpu_resume_smoke_v1
```

Production or CUDA training is intentionally not authorized by this foundation contract. A later calibration package must compare raw and EMA outputs against the unchanged v4 procedural baseline on the same validation and full-size test sentinels. Every object head must be non-regressing, and hard legality/provenance gates remain separate from quality metrics.
