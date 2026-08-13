# Actual neural sprite motion presentation bank

`forge.multifield_style_neural_motion` converts accepted raw neural categorical
sprites into native 48 px motion/presentation atlases. It does not replace a
neural silhouette with a procedural specimen. Raw `part_owner`, `material`, and
`emission_level` fields remain the pixel authority; the neural rig bridge only
derives driver transforms, joints, and sockets; presentation only derives RGBA
layers.

## Production scope

The v1 bank audits all 80 immutable samples from the final stratified production
generation bank, but animates five identities: one bank-ordered identity from
each family that binds and passes the complete motion matrix. It must not be
described as an 80-identity animation bank.

The manifest contains the full binding census. The frozen source currently has
70 bindable and 10 rejected samples. Rejections are recorded individually and
grouped as three anchors on neural background, one plant topology failure, three
missing required owners, and three safety-margin failures. This census is a
source-quality audit, not a claim that all 70 bindable identities were animated.

Each selected identity has all 13 motions and all 8 facings: 104 clips and 944
frames. Across five families the bank contains 520 clips and 4,720 frames.

## Derived layers

Every frame has seven exact native layers in this order:

1. `base`: opaque three-tone perceptual material shading.
2. `outline`: separate categorical-alpha chromatic 1 px outline.
3. `emission_core`: categorical-alpha emissive body pixels.
4. `aura`: partial-alpha effect only; never collision/body authority.
5. `bloom_r1`: deterministic Chebyshev radius-1 ring.
6. `bloom_r2`: deterministic Chebyshev radius-2 outer ring.
7. `composite`: exact alpha composition of the preceding presentation layers.

There is no antialiasing or resizing in the bank. Contact sheets and video use
nearest-neighbor preview scaling only. One identity palette is derived from the
immutable static presentation parent and reused byte-for-byte across every
motion, facing, and frame. Motion emission pulses may promote existing effect
alpha/color, but cannot add support or change a categorical pixel.

## Provenance and validation

The top-level contract is
`shared/schema/multifield_style_neural_motion_bank.schema.json`; each identity
uses `shared/schema/multifield_style_neural_motion_identity.schema.json`.
`motion_style_neural_manifest.json` binds:

- generation manifest, corpus, split, and legal-tuple fingerprints;
- static presentation manifest and compiler source hash;
- neural rig bridge/motion-program source hash;
- this compiler source hash;
- every identity manifest, palette, binding, source-motion collection, frame
  index, layer atlas, contact sheet, poster, and MP4 artifact.

The loader rejects noncanonical JSON, unsafe/symlink paths, hash or byte-count
tamper, noncanonical binding/clip/frame hashes, matrix reordering, malformed or
oversized NPZ members, wrong shapes/dtypes, atlas/index disagreement, palette
drift, clipped effect rings, aura/body overlap, noncategorical body/outline alpha,
and incorrect composites. Exact replay recompiles all four bounded shards per
family, reassembles all final bytes, and deterministically re-encodes the ffmpeg
showcase.

The bridge's validating motion constructor rerenders every stored affine bound
frame. To keep native processes short and resume-safe, a family is compiled in
four canonical motion shards (216, 248, 280, and 200 frames), then assembled into
one 944-cell atlas. `_build_shards/` is derivation workspace, not runtime content;
runtime consumers use only hash-bound identity artifacts listed by the final
manifest.

## Commands

```powershell
$env:CUDA_VISIBLE_DEVICES=''
python -m forge.multifield_style_neural_motion compile `
  outputs/production_handoff_v2/final_best_stratified80_bank_attempt1/generation_manifest.json `
  outputs/multifield_style/final_best_stratified80_v3/style_manifest.json `
  outputs/multifield_style_neural_motion `
  --ffmpeg C:\path\to\ffmpeg.exe

python -m forge.multifield_style_neural_motion replay `
  outputs/multifield_style_neural_motion/motion_style_neural_manifest.json `
  --report outputs/multifield_style_neural_motion/verification_report.json
```

The compiler is CPU-only, enforces the 100 GiB free-space floor, refuses to
overwrite mismatching resume artifacts, and fails closed if any pinned source
changes. The Godot/runtime adapter is intentionally separate and must require
both the ready bank manifest and passed replay report.

## Known limitations

- V1 animates five identities, not all 70 bindable identities.
- Motion is graph-driver affine animation of neural categorical fields, not a
  learned temporal diffusion model.
- Collision remains external authority and is deliberately not authored here.
- The MP4 is a visual audit artifact, not the runtime animation format.
