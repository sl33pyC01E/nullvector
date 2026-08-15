# NULLVECTOR // Action-conditioned cellular motion corpus

## Purpose

The version 1 motion corpus turns the deterministic cellular animation
authority into a strict training and evaluation dataset for the Stage 2 neural
motion model. It is not a sprite sheet and it does not rasterize appearance.
It records how every anatomical cell moves relative to its own rest position
under an explicit action, aim, locomotion command, or external event.

Coverage is exhaustive over the current native curriculum:

- five morphology families;
- four structural chassis per family;
- thirteen motion states;
- 72 frames per clip at 30 Hz;
- 20 chassis, 260 clips, and 18,720 total frames;
- 4,201,704 cell-frame samples.

This gives an action-conditioned learner genuine paired anatomy and motion
instead of asking it to infer causal pose labels from rendered pixels.

## Publication

The Godot exporter is invoked before the playable world is constructed:

```powershell
C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe `
  --headless --path C:\Users\forre\Documents\neural-game\game -- `
  --creature-stage-motion-corpus=C:\Users\forre\Documents\neural-game\outputs\creature_stage_motion_corpus_v1_final_a
```

Publication is additive and atomic. The command refuses an existing final or
staging directory, writes the complete corpus into `<destination>.tmp`, closes
and hashes both artifacts, then renames the staging directory. Before writing,
it reserves 64 MiB and verifies that at least 100 GiB remains free.

The final directory has exactly two members:

```text
manifest.json
motion_frames.u16le
```

No Python, neural checkpoint, GPU, viewport, or browser is required to create
the corpus.

## Binary layout

`motion_frames.u16le` is a dense little-endian unsigned-16 stream in this
order:

```text
clip -> frame -> anatomical cell -> x delta, y delta
```

Each coordinate stores a local rest-relative pixel displacement:

```text
encoded = round(delta_pixels * 256) + 32768
delta_pixels = (encoded - 32768) / 256
```

Every cell therefore costs four bytes per frame. Clip records state their
absolute byte offset, length, cell count, and frame stride. Version 1 had zero
clipped coordinate values. It preserves subpixel motion at 1/256 pixel while
remaining compact, deterministic, and directly memory-mappable.

The manifest records, for every chassis:

- family, morphotype, seed, generation, and genes;
- ordered grid coordinates;
- tissue and organ identity;
- appendage and side ownership;
- initial quantized health;
- an exact ordered cell-identity hash.

Every clip records the motion state, chassis, frame count, byte range,
trajectory hash, movement vector, normalized aim vector, attack and utility
channels, and external event (`none`, `impact`, or `terminal`). The manifest
also binds the three native producer files by SHA-256.

## Validation and loading

Run the independent validator from the repository root:

```powershell
python -m forge.creature_stage_motion_corpus `
  outputs/creature_stage_motion_corpus_v1_final_a
```

The validator fails closed on:

- oversized, non-UTF-8, duplicate-key, non-finite, or schema-invalid JSON;
- symlinks, missing files, extra files, bad paths, or oversized artifacts;
- stale producer source hashes;
- incorrect family, chassis, morphotype, motion, control, or event ordering;
- duplicated anatomical grid cells or forged cell identities;
- byte gaps, overlaps, truncation, trailing data, or altered clip hashes;
- non-finite decoded motion, a collapsed clip, displacement beyond 14.01
  pixels, or excessive death-spread growth;
- any mismatch in the complete corpus identity chain.

`load_clip_deltas(path, clip_id)` first performs the complete validation, then
returns a read-only NumPy float32 array shaped `[72, cell_count, 2]`. The
caller cannot accidentally mutate the authoritative decoded tensor.

## Reproducibility evidence

Two independent final native exports were validated and compared byte for
byte:

| field | value |
| --- | --- |
| format | `nullvector-creature-stage-motion-corpus-v1` |
| manifest SHA-256 | `82ca3a80325106e7cab06d17d51937e6ff0cc924a8c07caa49bfb7d3e14769d4` |
| binary SHA-256 | `9e17688195791fa7a898569cd30fe1ccc20c272aa1b85cf3033aa3fc85437bca` |
| producer source SHA-256 | `13704d6052fd6ac187e6d6e2e05566f71443111116a68fae04054fbaa862f493` |
| corpus identity SHA-256 | `23c28c6cf09a540a8d761e5d32651b528444697880db495418c03d689e51f30d` |
| binary bytes | 16,806,816 |
| maximum decoded displacement | 8.975374 pixels |
| minimum meaningful clip displacement | 0.453125 pixels |
| quantization clipping | zero values |

The compared roots are
`outputs/creature_stage_motion_corpus_v1_final_a` and
`outputs/creature_stage_motion_corpus_v1_final_b`. Their manifests and binary
payloads are separately byte-identical. Older `v1_a` and `v1_b` directories
are preserved as historical evidence but are intentionally stale after the
disk-floor producer guard was added.

## Scope and next aligned corpus

Version 1 is authoritative for local rest-relative anatomy motion conditioned
on the current action vocabulary. It does not claim to represent:

- RGB, emission, shadow, fluid, or material rasterization;
- damage, severing, scar formation, healing, or organ-capacity trajectories;
- collisions, projectile contacts, resource transfer, or world movement;
- long-horizon controller memory, ecology, society, or map state.

Those omissions are explicit so a learner cannot be called a full organism or
world model merely because it reconstructs clean pose trajectories. The next
aligned dataset should combine this stable cell identity with intervention
traces: wounds, bonds, fluids, organs, contacts, controls, and rendered layers.
That dataset can supervise an ensemble dynamics model and later the recurrent
action-DiT/VAE student without weakening the existing deterministic replay
authority.
