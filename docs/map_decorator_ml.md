# Topology-Locked Categorical Map Decorator ML

Status: CPU-tested model/training foundation. A process-isolated production corpus and
CUDA-training slice is implemented in `forge.map_decorator_production`; see
`docs/map_decorator_production.md`. It is not a playable renderer.

## Boundary

`forge.map_decorator_ml` consumes the versioned 53-channel feature tensor from
`forge.map_decorator` and predicts only four cell-resolution fields:

| head | classes | meaning |
| --- | ---: | --- |
| variant | 8 | terrain micro-variant |
| decal | 3 | empty plus up to two theme-local decals |
| prop | 3 | empty plus up to two non-colliding theme-local props |
| emission | 4 | off, low, medium, high |

Terrain, walkability, hazard, elevation, zones, navigation costs, mission
points, collision, and topology masks are never outputs. The teacher loader
accepts only strict map schema `2.0.0` packs through
`forge.maps.io.load_map_pack()`. It reads `protected_backbone`,
`required_clearance`, and `decoration_forbidden` directly from the returned
`MapData`; there is no legacy migration or inferred-mask path.

## Model

`CategoricalRefinementUNet` is a two-scale depthwise residual U-Net. A theme
embedding, global map condition vector, and refinement level are injected with
FiLM at every scale. The state input contains one-hot categorical values,
explicit masked indicators, and the refinement level. Masked categories have
zero one-hot state, so class zero is not confused with an unknown value.

The default production-shaped configuration uses 48 base channels and 96
condition channels. Smoke tests use 4-8 base channels. Inputs from 32 through
256 cells on either axis are right/bottom padded to a multiple of four and
cropped back to the exact original top-left extent. Categorical grids are never
resized or interpolated.

## Legality during prediction

Legal-class masks are applied to logits before every loss prediction and every
sampling draw. They are not a final clamping pass.

- Variant logits retain the contract's eight legal micro-variants.
- Decal and prop logits use exact theme/terrain/topology masks.
- Decal and prop are decoded as one joint categorical choice: empty, one decal,
  or one prop for both training metrics and sampling. Two object classes cannot
  occupy one cell.
- After variant and object selection, the emission mask is rebuilt from those
  selected fields. Emission is then drawn only from the newly conditional legal
  set.
- Decal, prop, and emission are class zero on every hard-empty cell at every
  refinement step.
- Every intermediate step and the final candidate are passed through the
  rejecting decoration validator. A failure aborts sampling and is never
  repaired silently.

The sampler uses a dedicated seeded `torch.Generator`, deterministic algorithms,
stable row-major confidence tie-breaking, and parallel masked refinement. Same
model/checkpoint, source map, feature seed, sampler configuration, and generation
seed replay byte-identically on the tested CPU path.

The smoke trainer additionally seeds the CPU module-initialization generator and
uses deterministic algorithms. Two clean smoke rebuilds therefore match in
training history, model tensor hash, EMA tensor hash, and prediction field hash;
wall-clock timing and container-file bytes are not used as rebuild identities.

## Teacher adapter and splits

`load_teacher_sample()` performs the following strict sequence:

1. Load and fully validate a v2 map pack, including hashes and deterministic map
   replay.
2. Encode the 53 channels with the three persisted masks.
3. Run the deterministic map-art forge and project its variant/object semantics
   into the safe theme-local catalog.
4. Exclude colliding props and hard-empty placements. If the legacy teacher put
   a decal and prop on one cell, stable source-catalog order chooses one.
5. Derive semantic emission levels and rebuild conditional legal masks.
6. Reject the sample unless every target is legal.

The split key hashes the complete source map identity: schema, map ID, seed,
theme, dimensions, generator/config, semantic-array hash, and topology-mask hash.
Crop coordinates and feature-noise seeds are deliberately excluded. Thus every
crop, guide view, or noise realization from one complete map receives the same
80/10/10 train/validation/test assignment. Sample and corpus identities still
include crop/noise/target hashes for provenance and duplicate rejection.

## Training and selection

Training applies independently seeded categorical corruption, with a shared
mask for the mutually exclusive decal/prop state. When foreground exists, the
corruptor guarantees at least one foreground decal, prop, and emission target is
selected in each applicable batch item. Cross-entropy is evaluated on corrupted
valid cells only. Per-batch inverse-square-root class weighting and a foreground-
aware empty-class cap prevent sparse empty targets from dominating.

Metrics report confusion matrices, macro IoU, foreground macro IoU, foreground
F1, and rare-class recall per head. Empty accuracy is explicitly absent from
the selection score, and rare-class recall excludes the empty class.

EMA state, optimizer state, explicit training-generator state, model/training
configuration, corpus identity, feature/catalog/model contracts, every package
source-file hash, model tensor hash, and EMA tensor hash are stored in atomic
checkpoints. Resume requires an exact contract match. Checkpoint startup never
deletes or rotates existing files.

## Artifacts and CPU smoke

Run the isolated smoke path after v2 map packs exist:

```powershell
python -m forge.map_decorator_ml `
  --packs outputs/maps_v2_forge_lab `
  --output outputs/map_decorator_ml_smoke `
  --maps 2 --train-steps 2 --refinement-steps 4 `
  --base-channels 8 --condition-channels 16 --threads 2
```

It refuses to start if CUDA is already initialized and verifies CUDA remains
uninitialized before publication. The output tree is built under a unique
staging directory and published with one rename. It contains an atomic
checkpoint plus hash sidecar, raw prediction fields plus manifest, and a smoke
report. Reload validation binds the source semantics/topology masks, encoded
features, catalog/model contract, checkpoint/source/corpus hashes, array
descriptors, sampler configuration, recorded validation, and all raw field
hashes. The prediction manifest explicitly selects the checkpoint's EMA weight
set and binds its tensor hash. A mismatched or malformed member fails closed.

All writes honor the shared 100 GiB free-space floor.

## Current limitations

- Production corpus/training contracts and supervisors are implemented; generated corpus
  and CUDA checkpoint evidence are reported separately from this frozen foundation.
- The frozen foundation adapter invokes the complete deterministic pixel renderer. The
  production corpus uses an exact semantic-only teacher path, with cross-theme equivalence
  tests, and avoids RGB/hazard-frame allocation.
- Density budgets, spatial-statistic regularizers, animation phase, a compiled
  neural art renderer, blind held-out comparisons, and ForgeLab integration are
  later stages.
- CPU byte replay is covered. CUDA byte replay must be characterized per PyTorch,
  driver, and GPU build before any CUDA checkpoint is publishable.
