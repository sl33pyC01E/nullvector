# Native Neural Asset Workshop

`game/NeuralWorkshop.tscn` is an additive Godot 4.3 tool scene for inspecting
the actual neural sprite bank and topology-v2 map bank. It is not a browser and
does not replace the game: `game/project.godot` continues to start
`res://Arena.tscn`.

## Runtime bundle

The scene consumes only
`res://generated/neural_workshop/v1/asset_index.json` and its recorded PNG/JSON
artifacts. Python, NumPy, checkpoints, corpora, raw semantic NPZ files, and
model code are build-time dependencies only.

The current bundle exposes:

- 80 real neural identities;
- humanoid, animalian, plantlike, anomaly, and machine families;
- 20 subtypes, eight roles, and two variants per family/role pairing;
- native 48px `base`, `outline`, `emission_core`, `aura`, `bloom_r1`,
  `bloom_r2`, and `composite` layers;
- six topology-v2 map themes;
- the nine rendered map-art layers plus exact visualizations of
  `protected_backbone`, `required_clearance`, `decoration_forbidden`,
  `walkability`, hazard semantics, zones, and navigation cost;
- nearest filtering and exact native 1x/4x sprite views.

Every runtime asset has a byte count and SHA-256 in the index. Versioned output
directories incorporate both upstream identity and workshop compiler hashes.
The sync validates the static model/checkpoint lineage, all per-layer hashes,
the balanced identity matrix, every topology-v2 invariant, exact semantic
hashes, atlas regions, the 100 GiB free-space floor, and the untouched Arena
baseline.

## Neural-motion adapter

The workshop contains a staged adapter for
`outputs/multifield_style_neural_motion/motion_style_neural_manifest.json`.
Motion is enabled only when the public bank has all of the following:

- format `nullvector-multifield-style-neural-motion-bank-v1`;
- `status=ready` and `neural_output=true`;
- a `verification_report.json` with exact replay `status=passed`;
- five identity manifests with 520 complete motion/facing clips and 4,720
  frames;
- seven exact 48px presentation atlases per identity;
- all authority, palette, binding, topology, event, loop, outline, and bloom
  gates passing.

The adapter validates the canonical bank and identity manifests against their
strict Draft 2020-12 schemas. It also requires the exact production binding
census: 80 immutable static samples, 70 bindable, and 10 rejected. The family
breakdown is humanoid 15/1, animalian 16/0, plantlike 13/3, anomaly 13/3, and
machine 13/3 (bindable/rejected). Rejection evidence must close exactly at
three background-anchor failures, one plant topology failure, three required
owner failures, and three safety-margin failures.

Version 1 animates one exact representative per family, not all 70 bindable
identities and not all 80 static identities:

- humanoid: `0000_f0_s00_r0_v00`, static cell 0;
- animalian: `0016_f1_s04_r0_v00`, static cell 16;
- plantlike: `0032_f2_s08_r0_v00`, static cell 32;
- anomaly: `0048_f3_s12_r0_v00`, static cell 48;
- machine: `0064_f4_s16_r0_v00`, static cell 64.

The motion panel always labels this family-representative mapping. Selecting a
different static member of the same family does not imply that identity was
animated. Looping clips retain the duplicated terminal frame as exact replay
evidence, while native playback cycles over `frame_count - 1` to avoid holding
the first pose twice.

The replay report is itself canonical and schema-validated. It must bind the
exact bank path, byte count, SHA-256, and compiler hash; contain five ordered
family proofs with four shards and 12 artifacts each; close at 5 identities,
520 clips, 4,720 frames, and 63 compared artifacts; and assert exact identity,
showcase, and gate replay. Broad `status=passed` claims are insufficient.

Missing, partial, stale, path-escaping, hash-mismatched, or failed-replay input
is rejected fail-closed. The UI clearly displays the staged/rejected state and
never substitutes a procedural animation while claiming neural output.

The runtime bundle includes each upstream identity manifest only as a verbatim
audit copy. Paths embedded inside those copied manifests remain relative to
the upstream compiler output and are deliberately not runtime-resolvable; the
runtime contract is the workshop index plus its hashed atlas records.

## Rebuild and validate

From the repository root:

```powershell
python -m forge.neural_workshop_sync `
  --repeat-check `
  --report outputs/neural_workshop_sync_report.json
```

The repeat check builds two fresh runtime trees and requires exact tree hashes.
For release/CI promotion, additionally require the authoritative motion bank:

```powershell
python -m forge.neural_workshop_sync `
  --require-neural-motion-ready `
  --repeat-check `
  --report outputs/neural_workshop_sync_ready_report.json
```

That command fails closed while the source is staged or rejected. Do not use
the non-required command as evidence that neural motion is ready.

Import and exhaustively smoke the native scene:

```powershell
C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe `
  --headless --editor `
  --path C:\Users\forre\Documents\neural-game\game --quit

C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe `
  --headless `
  --path C:\Users\forre\Documents\neural-game\game `
  res://NeuralWorkshop.tscn -- `
  --neural-workshop-smoke `
  --neural-workshop-report=C:/Users/forre/Documents/neural-game/outputs/neural_workshop_godot_report.json
```

The smoke hashes every runtime file through Godot, visits all 80 identities and
seven static layers, checks every atlas region, traverses every theme/layer/map
frame, verifies topology-v2 markers and Arena preservation, and exhaustively
checks neural motion when that bank is authoritative. A staged build currently
prints coverage equivalent to:

```text
NEURAL_WORKSHOP_SMOKE_OK identities=80 static_layers=7 static_regions=560 maps=6 map_regions=96 map_frames=138 motion=staged motion_atlases=0 motion_clips=0 motion_frames=0 motion_atlas_regions=0 hashes=27
```

For a ready bank, the smoke requires exactly 69 runtime inventory records,
loads all 35 motion atlases at 768x2832 RGBA pixels, and constructs and checks
all 4,720 native 48x48 atlas regions (in addition to validating all 520 clips).

## Controls

- `Q` / `E`: previous / next filtered identity
- `1`, `2`, `3`: cycle family, subtype, and role filters
- `0`: clear filters
- `Z` / `X`: previous / next presentation layer
- `V`: native 1x / 4x view
- `W` / `S`: previous / next motion
- `A` / `D`: previous / next facing
- Left / Right: motion frame scrub
- Space: pause / play
- `R` / `F`: previous / next map theme
- `T` / `G`: previous / next map layer
- Comma / Period: map-frame scrub

The scene may also be opened directly in the Godot editor. All controls have
matching native buttons.
