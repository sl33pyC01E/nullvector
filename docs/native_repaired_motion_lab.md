# Native all-80 repaired motion lab

`game/RepairedMotionLab.tscn` is an additive Godot 4.3 inspection scene for
the sealed all-80 repaired neural motion bank. It does not replace
`Arena.tscn`, which remains the project main scene.

The lab is the runtime promotion of two independently replayed authorities:

- repair bank `2a58a6435b50963d2b415376fb7c72cb0e2e9bec93acd0d196ed7f7ffe241c2a`;
- styled motion bank `99a8043ef9d226e52ec44ac79bf27128d8babf310374e05af8423c9d939b3cb6`.

Unlike the original Neural Workshop's five representative animations, this
lab animates all 80 neural identities. Every identity has 13 motions, eight
facings, 944 stored frames, and seven native presentation layers. The complete
runtime contract is therefore 8,320 clips, 75,520 stored frame regions, and
560 `768x2832` nearest-filtered atlases.

The compiler never edits the neural rest pixels. Repair is restricted to
logical driver assignment, anchor support, component links, bounded motion
attenuation, uniform clip fitting, and clip-local layer ordering. Every source
tuple, plan, binding, frame record, and presentation artifact was independently
replayed before the style bank was sealed.

## Native controls

- `Q` / `E`: previous / next identity
- `W` / `S`: previous / next motion
- `A` / `D`: previous / next facing
- `Z` / `X`: previous / next presentation layer
- Left / Right: scrub stored frames
- Space: pause / resume
- UI buttons also jump by family and select native 1x or pixel 6x display

Looping clips retain their duplicated terminal frame as exact-replay evidence,
but playback cycles over `frame_count - 1`, preventing a one-tick seam hitch.

## Rebuild

From the repository root:

```powershell
python -m forge.repaired_motion_lab_sync `
  --report outputs/repaired_motion_lab_sync_report.json
```

The projection validates all primary and replay shards, all identity
self-hashes, all clip matrices, all PNG hashes and dimensions, the source
compiler hash, and the 100 GiB free-space floor. Publication is atomic and
refuses an existing destination. The current native bundle is additive under
`game/generated/repaired_motion_lab/v1` and requires no Python runtime.

## Headless proof

```powershell
C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe `
  --headless --path C:\Users\forre\Documents\neural-game\game `
  res://RepairedMotionLab.tscn -- `
  --repaired-motion-lab-smoke `
  --repaired-motion-lab-report=C:/Users/forre/Documents/neural-game/outputs/repaired_motion_lab_godot_report.json
```

On Windows, use `Start-Process -Wait -WindowStyle Hidden` with the GUI-subsystem
Godot executable so the invoking shell waits for the authoritative exit code.
The smoke streams atlases one at a time, verifies every runtime byte/hash and
dimension, validates all 8,320 clip bounds, and constructs all 75,520 frame
regions without retaining the full atlas bank in memory.

## Motion quality contract

Addressable frames alone do not prove useful animation. The additive
`forge.repaired_motion_quality` audit decodes every base atlas and evaluates all
8,320 clips after removing integer centroid translation. It fails on blank or
near-static playback, occupancy flashes, motion-specific articulation collapse,
duplicate rendered identity sequences, family imbalance, source drift, or a
fully rehashed forged report.

The calibrated minimum translation-compensated articulation floors range from
3% for breathing to 55% for death. Locomotion requires at least 8.5%; attack
and hit require 32%; cast requires 19%. Every playback clip must contain at
least four distinct silhouette frames, and no frame may exceed 1.35 times its
clip's median occupancy. These are regression floors, not aesthetic maxima.

The current explicitly attested, visually inspected report is
`outputs/repaired_motion_quality_v2/motion_quality_report.json`:

- 80 unique rendered base atlases, 8,320 clips, and 75,520 stored frames;
- zero blank, static, occupancy-spike, or articulation-floor failures;
- minimum observed articulation `36,680 ppm` and maximum occupancy spike
  `1,307,918 ppm`;
- 104 shared alpha-sequence groups belonging to one intentional machine
  palette-variant pair, but zero duplicate full-RGBA sequences;
- clip-metric identity
  `fbf217dcc7988a353896bdac7dde95ba0c8084d16e6ffbf76b7d469fb3c5ea66`;
- report identity
  `1a8a1ec57834902d0ca719715dae8974bb52bf611161f5b0793e8c9da17a5993`;
- report file SHA256
  `7e567134ac9245448c6bf47b80dde84344de71ad7135dd848afd6a1b6d6a8eb9`;
- audit source identity
  `c5632c0a7ad443a7165c27359dbea750e9b0e72d08e8190815c3ed40ab517a11`;
- dynamics contact sheet
  `592b26ddfbe4ef0ab272e51d9d4abade1180d4b456591532320521efae55ac0e`.

Each contact-sheet cell shows the first frame beside the frame with maximum
translation-compensated articulation for one north-facing family
representative. Exact replay recomputes every metric and the PNG bytes from the
hash-bound native bank. Version 2 additionally fails publication unless the
builder supplies the explicit `--visually-inspected` attestation; validation
replays that attestation instead of silently inventing it.

```powershell
python -m forge.repaired_motion_quality validate `
  --output outputs/repaired_motion_quality_v2 `
  --runtime game/generated/repaired_motion_lab/v1
```
