# Power-recovery handoff — 2026-08-12

## Closure update — 2026-08-14

The old `neural_rig_repair_v2_recovery_20260812` tree below remains preserved
as stale, unsealed evidence and must not be resumed against the current source.
Its fail-closed source check correctly rejects the old `c702843c...` shards.
The later authoritative bank is already sealed at
`outputs/neural_rig_repair_v2_sharded_replay_20260813` against current repair
source `11d4faccc9d9a5d66e35cf2199dc8d415ceb250c2e0d0cb63a99ab827672a54e`.

A fresh CPU-only replay on 2026-08-14 independently recompiled all 80 plans,
rerendered 8,320 motion/facing clips and 75,520 semantic frames, and passed all
exact comparisons. The report is
`work/neural_rig_repair_replay_current_20260814.json`; its canonical replay
SHA-256 is `1444d2f5bbf99166e3968469db506b5f2be658981fed17c747f3898688d3e4f9`
and its file SHA-256 is
`9883c5c28e9c5e85313809327e992b2b5b6d553fc36d0855c0493715110b5b85`.
The focused repair suite passed 17 tests with 3 expected sealed-bank skips, the
14 non-exhaustive bridge adversarial tests passed, and the repair-style suite
passed 2/2. The existing all-80 style bank remains current and source-bound:
80 identities, 560 layer atlases, 8,320 clips, and 75,520 frames.

This is the frozen handoff after the unexpected Windows shutdown at 17:03.
Nothing listed as staged is a publishable bank. No neural-game Python, Godot,
FFmpeg, or CUDA training process was left running.

## Host boundary

- NTFS reported the C: volume healthy after boot.
- Free space at handoff: 207.203 GiB (required floor: 100 GiB).
- RTX 4090 at handoff: 880 MiB used / 23,259 MiB free; no project CUDA job.
- The host still has known intermittent native `0xC0000005` failures. Resume
  long CPU work only through the existing process-isolated supervisors.

## Revalidated immutable outputs

### Neural sprite motion

- Bank: `outputs/multifield_style_neural_motion`
- Scope: 5 actual neural representatives, 520 clips, 4,720 frames,
  63 compared artifacts, 23,698,097 compared bytes.
- Fresh post-boot exact replay:
  `outputs/power_recovery_neural_motion_replay_20260812.json`
- Replay SHA-256: `101b8a0723a525fa3641c8ebfb45fddece3d97e44c17698a30f3d47f5853f6bd`
- Result: byte-identical to the original verification report; all gates passed.

### Native Neural Workshop

- Ready sync report:
  `outputs/power_recovery_neural_workshop_sync_20260812.json`
- Godot smoke report:
  `outputs/power_recovery_neural_workshop_godot_20260812.json`
- Runtime coverage: 80 identities, 560 static regions, 6 maps, 96 map
  regions, 138 map frames, 35 motion atlases, 520 clips, 4,720 motion
  regions/frames, 69 hashed runtime assets.
- Godot 4.3 headless import and runtime smoke both exited 0.
- `project.godot`, `Arena.tscn`, `scripts/arena_game.gd`, and the Arena main
  scene were preserved exactly.

### Sprite quality and neural map topology

- Sprite-quality exact replay passed:
  report `466af395f0848da414d53fa3029a1e77b05396e203a4e1046fb6c8fd91262df9`,
  semantic report `4692bcd45247940a8407e479d688d2998f7a757bd0369c484c138be13dd50d5e`,
  heatmap `6eb886e1da25ec1ecf123fb8505c79aa9c95f71431b526909270a4568e1a587b`.
- Neural map-topology exact replay passed: 6 themes, 72 arrays, 3,496
  ledger entries, exact contact sheet, bounded checkpoint load, exact codec
  decode. Manifest SHA-256:
  `b3e4628fb67877b5231011ac7a24d6222283e2fd4f6c7397c4de3d19a847e576`.
- Focused sprite-quality + map-topology suite: 29 passed.
- Map-quality audit and showcase replayed exactly; focused suite: 18 passed.

## Decorator v2 index — recovered and published

- Published index:
  `outputs/map_decorator_production_v2/foreground_index_v2`
- Complete: 216/216 shards, 3,096 samples.
- Splits: 2,496 train / 576 validation / 24 test.
- Duplicate sample identities: 0; duplicate full-map identities: 0.
- Outage staging evidence remains untouched at
  `outputs/map_decorator_production_v2/.foreground_index_v2.tmp-eb0282329d7c46e2bc8d7a655207e4fd`.
- The recovery imported 104 byte-identical shards and built 112 missing shards
  with two CPU workers: 112 attempts, 0 retries, 0 native failures.
- Corpus SHA-256:
  `16ed5f3b1a661e2bfc2abe9e16c39e9b8caaecba81f50fed6658cc4f73cffab8`.
- Index semantic SHA-256:
  `7fe634cf293302738220c9b4b12db93fe6fa2cde5e045079c2daa81b52db8838`.
- Recovery report SHA-256:
  `77d100f7f48ec93ddc9f61d288ec180e886bcddaaf08083fa13f842477ce8849`.
- Focused recovery/v2 suite: 10 passed; independent index validation passed.

Safe read-only validation:

```powershell
C:\Users\forre\AppData\Local\Programs\Python\Python312\python.exe -m forge.map_decorator_production_v2.index validate --corpus C:\Users\forre\Documents\neural-game\outputs\map_decorator_corpus_v1 --index C:\Users\forre\Documents\neural-game\outputs\map_decorator_production_v2\foreground_index_v2
```

Do not rerun recovery: the target is sealed and intentionally fails closed.
The optional monolithic fresh corpus replay failed without identifying a shard;
it wrote nothing. Do not claim that optional gate as green until replaced with
shard-isolated diagnostics.

## Neural rig repair v2 — intentionally staged

- Recovery tree:
  `outputs/neural_rig_repair_v2_recovery_20260812`
- Repair source SHA-256:
  `c702843c80ab9e126efb3d1ac8ad249ff1d3d87a1035d0e69b022ae57734c267`.
- Passed shards: 9/16 (`00,01,02,04,05,06,07,08,10`).
- Missing shards: `03,09,11,12,13,14,15`.
- Shard 09 has a preserved bounded semantic failure artifact.
- No bank manifest, stress report, or verification report exists. Nothing was
  promoted. Focused tests: 14 passed, 3 expected sealed-bank skips.
- The corrected motion gate measures distance to the closed raster-cell
  footprint instead of only pixel centers; rest pixels and the frozen bridge
  contract remain unchanged.

Exact resume command:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:NEURAL_RIG_REPAIR_CPU_ONLY='1'
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
& 'C:\Users\forre\AppData\Local\Programs\Python\Python312\python.exe' -m forge.neural_rig_repair compile --destination 'C:\Users\forre\Documents\neural-game\outputs\neural_rig_repair_v2_recovery_20260812' --workers 2 --timeout-seconds 900 --max-attempts 3
```

The compiler will reuse exact source-bound shards and retry missing work. Do
not integrate this repair bank until all 16 shards and the independent 75,520-
frame replay are sealed.

## Semantic sprite FSQ codec — production closure 2026-08-14

- Current foundation source SHA-256:
  `aa0887845933ea3980efb2765a4fec73278f7cc8e9e12e97d11bd4f3bcd8ce4c`.
- `outputs/sprite_latent/smoke_v1` remains intentionally stale evidence.
- The additive `outputs/sprite_latent/smoke_v2` bank passes strict semantic and
  byte replay. It reconstructs all five families at native 48 px, reaches
  0.82944876 aligned legal-tuple accuracy, and uses 235 implicit codes. Its
  self-hash is
  `5e10adb63752508e788d797fce2ca4d684ac8e3f6d8131d837864ee5f52aa3c4`.
- The 17-test foundation suite passes, including nonzero differentiable usage
  gradient, malicious NPZ rejection, manifest/checkpoint reseal rejection,
  and traversal, absolute, drive-qualified, backslash, and explicit-dot path
  rejection. The reconstruction contact sheet was visually inspected; it is a
  coarse representation smoke and does not claim production quality.

The later segmented production trainer is complete and current at
`checkpoints/sprite_latent_production_v1_run3`. Its source SHA-256 is
`9ddd274701d553777e95176db8ccde721d0c9094a5a1e5ca1c638d27c713fa79`;
the immutable 12-segment, 24-epoch chain validates with manifest SHA-256
`7ca60d478090470186b10c7f39b7e11e880fd4c21d8d38b1135f1f555dda5b08`.
The best EMA checkpoint is epoch 24 and passes every global and worst-family
quality gate: 0.960376 aligned tuple accuracy, 0.831137 visible tuple accuracy,
0.973630 visible silhouette IoU, 0.810480 worst-family visible tuple accuracy,
0.958969 worst-family silhouette IoU, and 0.775156 latent-code utilization.

The trained latent space is also exercised by the current deterministic
three-generation fusion/mutation bank at
`outputs/neural_fusion_production_evolution_v1_run2`. Its source SHA-256 is
`eacad82ea06e2a0d9f78e94f9241186684fe5b08051e3e0afd609a286b043a22`
and evolution SHA-256 is
`d9a9935018472d36a7b7e9e2a59803b173b906798131c7e1f177d110e7eaac71`.
Validation proves 108 accepted candidates, 36 selected descendants, all five
families, all six fusion operators, all six mutation operators, closed lineage,
legal connected rigs, and valid motion. The independent repeat has the same
evolution hash, and the three-generation contact sheet was visually inspected.
