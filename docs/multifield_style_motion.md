# Motion-coherent categorical presentation reference

This milestone is a deterministic **procedural reference bank**. It contains
zero neural-generation samples. Its inputs are the five authoritative
procedural morphology specimens and their canonical 13-motion, 8-facing
motion matrix. The separate neural-production motion path must begin with an
accepted raw neural sample, a strict `neural_rig_bridge` binding, and the
public motion-program adapter; procedural pixels must never be substituted
for neural pixels.

## Result

- 5 reference identities: humanoid, animalian, plantlike, anomaly, machine.
- 520 clips and 4,720 native 48x48 frames.
- 7 derived RGBA layers per frame: base, one-pixel chromatic outline,
  emission core, partial-alpha aura, Chebyshev radius-1 bloom, Chebyshev
  radius-2 bloom, and composite.
- One immutable palette/style identity per specimen across every motion and
  facing. Frame categorical, rig, socket, and event authority remains separate
  and source-bound.
- Family-isolated CPU workers; no checkpoint, trainer, GPU, Godot, or motion
  core mutation.
- Exact byte replay for every family atlas/index/manifest and the ffmpeg MP4.

The finalized bank is
`outputs/multifield_style_motion/motion_style_manifest.json` (SHA-256
`d3fbb4487b870c4efc7c980332a0f6e2d73ce94b9f94ca7fb4a142e1326ef7ad`).
Its verification report is
`outputs/multifield_style_motion/verification_report.json`.

## Authority split

`IdentityStyleFields.aligned_sha256` is deliberately the stable specimen style
key passed to the existing public presentation renderer. It is not presented
as the moving frame's categorical hash. Every frame independently records:

- source frame SHA-256;
- categorical part/material/emission SHA-256;
- joint SHA-256;
- socket SHA-256;
- combined authority SHA-256;
- seven presentation-layer SHA-256 values.

Rendering is required to leave the categorical arrays and anchor maps
unchanged. Base alpha exactly follows non-aura body occupancy. Aura remains an
effect, never body or collision authority. Outline and bloom supports must be
the exact Chebyshev rings requested by their names.

## Commands

```powershell
python -m forge.multifield_style_motion compile `
  game/generated/v2/asset_index.json `
  outputs/multifield_style_motion `
  --ffmpeg C:/path/to/ffmpeg.exe

python -m forge.multifield_style_motion replay `
  outputs/multifield_style_motion/motion_style_manifest.json `
  --report outputs/multifield_style_motion/verification_report.json
```

Compilation refuses a finalized destination. An interrupted destination may
resume only when `build_contract.json` is byte-exact. Any upstream source-hash
change fails closed instead of mixing compiler versions.

## Visual artifacts

- `motion_contact_sheet.png`: all five families across idle wiggle,
  locomotion, joy, attack, cast, and death, with six native layers plus a 2x
  nearest-neighbor composite.
- `motion_showcase_poster.png`: first frame of the animated grid.
- `motion_showcase.mp4`: deterministic ffmpeg/libx264 showcase at 12 FPS.

The contact sheet, poster, and decoded timestamps 9, 18, and 27 were visually
inspected. Pixel silhouettes are crisp, palette identity is stable, effect
rings remain outside body authority, and representative motion changes are
visible.

## Known boundary

This bank proves the presentation compiler across the complete canonical
motion vocabulary and all five morphology families. It does **not** prove that
the final 80-sample neural production bank can be animated. That requires the
separate public neural motion-program adapter and a new derived bank whose raw
NPZ/manifest, generation manifest, legal-tuple table, binding, frame, and
presentation hashes are all immutable and replayed exactly.
