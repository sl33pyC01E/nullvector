# Multi-field morphology diffusion training

`forge.train_multifield` trains the 48 px graph-guided diffusion model over
three aligned categorical fields: part ownership, material, and emission.

## Anti-leakage guide policy

Production training defaults to `scaffold_only` (`scaffold-guide-policy-v1`).
The exact target-derived `silhouette`, `body`, and `core` guide channels are
zeroed before the model or accelerator sees a batch. The retained skeleton,
joints, sockets, horizontal-position, and root-distance channels are legitimate
construction constraints. Skeleton heatmaps are thickened one pixel, then only
training batches receive explicit-RNG channel dropout and spatial jitter.
Validation and generation use the same sanitized scaffold without augmentation.

`--guide-policy full_debug` exposes all corpus guide channels and is strictly a
diagnostic overfit mode. Its metrics are not comparable to production runs.

## Smoke checks

Run the complete CPU path (two train batches, one validation batch, one sampled
specimen, atomic latest/best checkpoints):

```powershell
python -m forge.train_multifield `
  --corpus data/morphology_32768_4d4f5250.npz `
  --smoke --device cpu `
  --output-dir outputs/multifield_smoke_cpu `
  --checkpoint-dir checkpoints/multifield_smoke_cpu
```

Run the same path through CUDA and BF16 on the RTX 4090:

```powershell
python -m forge.train_multifield `
  --corpus data/morphology_32768_4d4f5250.npz `
  --smoke --device cuda --precision bf16 `
  --output-dir outputs/multifield_smoke_cuda `
  --checkpoint-dir checkpoints/multifield_smoke_cuda
```

## Production baseline

The production geometry was exercised for a complete CUDA BF16
forward/backward/clip/fused-AdamW step on the RTX 4090. Width 192 and batch 96
use 23,778,327 parameters, 7.374 GiB peak allocated / 7.900 GiB peak reserved,
and 0.643 seconds per measured step. This leaves ample VRAM for stable desktop
operation while providing substantially more capacity than the width-96 smoke
model. The recommended first run is intentionally not started by the smoke
suite:

```powershell
python -m forge.train_multifield `
  --corpus data/morphology_32768_4d4f5250.npz `
  --device cuda --precision bf16 `
  --epochs 72 --batch-size 96 --width 192 --diffusion-steps 16 `
  --warmup-steps 500 `
  --generation-eval-count 16 --generation-eval-interval 4
```

At 30,208 training specimens this is 315 optimizer steps per epoch and 22,680
steps total. The measured training-step floor is about 4.1 hours; full-mask
validation, fixed-bank generation, corpus loading, and atomic checkpoint writes
add overhead. For a supervised first tranche, append `--stop-after-epoch 12`,
inspect the report, then exact-resume without that flag.

The intended first production checkpoint is selected by a stable composite of
every-epoch, full-mask validation metrics. Sparse unconditional generation is
also sampled on a fixed specimen/seed bank at the configured interval, but it
does not silently disappear into the checkpoint selection score.

## Exact resume

```powershell
python -m forge.train_multifield --resume checkpoints/multifield/latest.pt
```

For a planned maintenance boundary without changing the cosine schedule, add
`--stop-after-epoch 8`; resume later with the command above. A regression test
verifies that a two-epoch CPU run interrupted at this boundary has the same EMA
hash, metrics, global step, and training-generator state as an uninterrupted
run.

The checkpoint stores raw and EMA models, optimizer, scheduler, AMP scaler,
Python/NumPy/CPU/CUDA RNG state, DataLoader shuffle generator, corruption and
augmentation generator, corpus SHA-256, split fingerprint, train-only legal
tuple table, class weights, fixed validation seeds, architecture, guide policy,
history, canonical EMA hash, and training-source hash. Resume rejects a changed
corpus, split, source tree, architecture, or training invariant. The planned
epoch count is therefore chosen at the beginning; changing it creates a new
run, not an exact resume. `--allow-source-change` is an explicit reproducibility
escape hatch and should only be used after reviewing the code change.

CUDA deterministic algorithms, deterministic cuDNN, TF32-off matmuls, and the
required cuBLAS workspace mode are enabled by default. The checkpoint also
records PyTorch/CUDA/cuDNN/GPU identity. `--allow-nondeterministic` is available
for profiling only and becomes a recorded resume invariant.

Both `latest.pt` and `best.pt` are written to temporary files, flushed, and
published with `os.replace`. Every checkpoint write checks the 100 GiB free-disk
floor first.

## Metrics

Every epoch reports weighted loss plus full-mask:

- silhouette IoU;
- foreground part macro IoU;
- material and emission macro IoU;
- all-pixel and foreground accuracies for each field;
- train-observed joint `(part, material, emission)` tuple validity;
- aggregate condition-preference rate and foreground NLL margin versus the best of three
  deterministic wrong conditions, plus separate morphology/subtype, role, and
  gene preference diagnostics. The role-only diagnostic is an explicit guard
  against a corpus whose renderer accidentally ignores its role label.

Scheduled reverse-diffusion evaluation uses a deterministic greedy bank that
covers all five morphologies and all eight roles before adding repeated
conditions (when the requested bank has at least eight specimens), and reports
the same shape/field/tuple metrics with a `generation_` prefix.
Legal-tuple-constrained samples must have
joint tuple validity 1.0; the full-mask argmax metric remains unconstrained and
reveals whether the neural heads themselves learned compatibility.
