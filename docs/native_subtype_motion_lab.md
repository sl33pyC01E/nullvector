# Native subtype motion lab

`SubtypeMotionLab.tscn` is an additive Godot 4.3 viewer for the explicit
twenty-chassis morphology grammar. It is deliberately labeled as a procedural
reference rather than neural output. `Arena.tscn` remains the main scene.

The runtime bundle contains 20 identities, 400 clips, 3,620 stored frames, and
60 compact 768x576 atlases. Each subtype has all thirteen north-facing idles,
locomotion, emotes, and actions plus eight-way locomotion. Three switchable
views expose composite color, categorical semantic ownership, and emission.
Loop playback omits the duplicate terminal proof frame.

```powershell
python -m forge.morphology_subtype_runtime_sync

C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe --headless `
  --editor --path C:\Users\forre\Documents\neural-game\game --quit

C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe --headless `
  --path C:\Users\forre\Documents\neural-game\game `
  res://SubtypeMotionLab.tscn -- `
  --subtype-motion-smoke `
  --subtype-motion-report=C:/Users/forre/Documents/neural-game/outputs/subtype_motion_godot_report.json
```

Controls: `Q/E` changes subtype, `W/S` changes motion, `A/D` changes locomotor
facing, `Z/X` changes field view, and `Space` pauses playback. Atlas filtering
is nearest-neighbor and no Python runtime is required.
