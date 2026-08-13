# Morphology motion forge

The motion forge turns one deterministic 48×48 morphology specimen into a
graph-driven animation bank without flattening or relabelling its semantics.
Every frame keeps the same 12 binary layers, regenerates the categorical
training fields, and carries updated joints and gameplay sockets.

## Motion vocabulary

Looping clips:

- `idle_breathe`, `idle_wiggle`
- family-specific `locomote`
- `joy`, `anger`, `fear`, `confused`, `sleep`, `taunt`

One-shot actions:

- `attack`, `cast`, `hit`, `death`

All clips support `north`, `northeast`, `east`, `southeast`, `south`,
`southwest`, `west`, and `northwest`. The direction transform is part of the
same graph pass as the bone motion, and one clip-wide fit transform prevents
per-frame scaling jitter.

Locomotion is morphology-aware: humanoids use counter-swinging bipedal legs,
animalians use a diagonal gait and tail wave, plantlike creatures walk on
roots with branch counterbalance, anomalies hover with asynchronous tendrils,
and machines rock their treads while stabilizing the turret.

## Rig and semantic guarantees

`forge/morphology/motion.py` applies hierarchical affine transforms to body,
head, limbs, appendage, and weapon bones. Humanoid weapons inherit the right
arm; the other families inherit their head/turret. Detail and emission pixels
are split by their source anatomical owner and follow the same bone as that
owner, so the fields cannot drift apart.

Every generated clip is rejected unless:

- all 12 semantic masks remain binary and nonempty;
- the visible sprite clears the three-pixel logical margin;
- the structural union has exactly one eight-connected component;
- all child parts remain attached to an allowed parent;
- detail and emission remain inside the structural silhouette;
- joints and sockets land on their contracted semantic owner;
- part/material/emission triples belong to the versioned cross-field rules;
- looping endpoints are bit-exact;
- the motion has at least two distinct semantic frames;
- idle/sleep foot sockets remain planted;
- frame and clip SHA-256 values replay exactly.

Clip manifests also carry frame-indexed socket events (foot plants, emote
peaks, weapon strikes, spell releases, impacts, and the grounded death pose),
so a runtime does not need to infer action timing from pixels.

`blend_motion_poses()` linearly blends rig state before rasterization. It gives
the runtime deterministic transition and overlay parameters without ever
cross-fading categorical pixels.

The strict JSON contract is
`shared/schema/morphology_motion_manifest.schema.json`.

## Commands

Generate the production preview bank:

```powershell
python -m forge.morphology.motion_preview
```

Run a deterministic broad-seed/facing/motion fuzz pass:

```powershell
python -m forge.morphology.motion_fuzz --count 500
```

Replay the saved semantic archive and byte-compare every generated frame:

```powershell
python -m forge.morphology.motion_replay
```

Run only the motion tests:

```powershell
python -m pytest tests/test_morphology_motion.py
```

Both writers reserve at least 100 GiB of free disk space before writing.

## Generated artifacts

`outputs/morphology_motion/` contains:

- `morphology_motion_contact_sheet.png`: all 13 motions across all 5 families;
- `morphology_motion_showcase.gif`: 32-frame animated comparison grid;
- `morphology_motion_showcase_frames.png` and `.meta.json`: a stable vertical
  PNG animation strip for tools that should not decode GIF;
- `morphology_motion_semantics.npz`: packed semantic frames, tokens, joints,
  sockets, phases, clip offsets, and hashes;
- `morphology_motion_manifest.json`: source and per-clip provenance;
- `motion_fuzz_report.json`: the last deterministic fuzz result.
- `motion_replay_report.json`: exact source, manifest, field, anchor, and frame
  replay results.

The GIF is encoded by ffmpeg. Pillow's multi-frame GIF writer is intentionally
not used because it has caused native Windows crashes in prior preview jobs.
