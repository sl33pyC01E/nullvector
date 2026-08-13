# Native Forge Lab

`game/ForgeLab.tscn` is an additive, native Godot 4.3 inspector for the v2
morphology-motion and semantic map-art banks. It does not replace the arena:
`game/project.godot` still starts `res://Arena.tscn`.

## Runtime contract

The scene reads `res://generated/v2/asset_index.json` and Godot-imported PNG
atlases only. Python, NumPy, checkpoints, corpora, and source semantic archives
are not needed at runtime. The compiled bank contains:

- five morphology families;
- 13 motions and eight facings, for 520 clips and 4,720 frames;
- six map themes, each exposing nine display layers;
- 96 map frames, including eight-frame hazard animation per map;
- role-conditioned morphology renderer `broad-morphology-grammar-v2-role-conditioned`;
- graph motion renderer `graph-layer-rig-v1`;
- nearest-neighbor filtering at the project, control, and import layers.

## Rebuild and verify

From the repository root:

```powershell
python -m forge.forge_lab_sync --repeat-check --report outputs/forge_lab_sync_report.json
```

The repeat check performs two complete source replays and proves that all 19
JSON/PNG runtime files are byte-identical. It validates source renderer hashes,
the complete family/motion/facing matrix, atlas dimensions and cell bounds,
every source and atlas SHA-256, map-art pack validation, and the 100 GiB disk
floor. Adjacent Godot `.import` sidecars are preserved as engine cache but are
excluded from the portable runtime bank inventory.

## Godot import and exhaustive scene smoke

The exact Godot 4.3 commands used for the locked audit are:

```powershell
C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe --headless --editor --path C:\Users\forre\Documents\neural-game\game --quit

C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe --headless --path C:\Users\forre\Documents\neural-game\game res://ForgeLab.tscn -- --forge-lab-smoke --forge-lab-report=C:/Users/forre/Documents/neural-game/outputs/forge_lab_godot_report.json
```

The runtime smoke visits every selector state, loads every family and map atlas,
checks its embedded SHA-256 through Godot, and checks every one of the 4,720
motion-frame regions and 96 map-frame regions against the imported texture
bounds. A passing run prints:

```text
FORGE_LAB_SMOKE_OK families=5 motions=13 facings=8 clips=520 motion_frames=4720 maps=6 layers=9 map_regions=54 map_frames=96 hashes=18
```

## Controls

- `Q` / `E`: previous / next family
- `W` / `S`: previous / next motion
- `A` / `D`: previous / next facing
- Left / Right: motion frame
- Space: pause or play
- `R` / `F`: previous / next map theme
- `T` / `G`: previous / next map layer
- Comma / Period: map frame

The scene may also be opened directly in the Godot editor. The arena main scene
and its controls remain unchanged.
