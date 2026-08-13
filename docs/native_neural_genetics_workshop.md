# Native Neural Genetics Workshop

`game/NeuralGeneticsWorkshop.tscn` is an additive Godot 4.3 laboratory for the
fusion, mutation, learned-latent, and multi-generation evolution artifacts. It
does not replace the game: `game/project.godot` still starts `Arena.tscn`.

The runtime bank contains only PNG and JSON:

- 10 verified cross-family categorical hybrids;
- all five fusion operators and all six mutation operators;
- 70 selected fusion clips / 660 stored frames across seven presentation layers;
- 12 learned-FSQ latent blends across four blend modes and three alpha values;
- 48 latent clips / 420 stored frames across seven presentation layers;
- 24 motion-gated evolutionary survivors from two generations;
- exact per-file SHA-256, source-bank hashes, lineage hashes, and a deterministic bundle ID.

The learned latent bank is deliberately labeled
`learned-fsq-smoke-not-production`; its presence is not a production-quality
claim. Categorical fusion and evolution use the frozen legal semantic tuple
vocabulary and pass the existing connectivity, fresh-rig, and motion gates.

Rebuild and prove exact repeatability:

```powershell
python -m forge.neural_genetics_workshop_sync `
  --repeat-check `
  --report outputs/neural_genetics_workshop_sync_report.json
```

Godot must import the freshly copied PNGs before the runtime smoke:

```powershell
C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe `
  --headless --import `
  --path C:\Users\forre\Documents\neural-game\game

C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe `
  --headless `
  --path C:\Users\forre\Documents\neural-game\game `
  res://NeuralGeneticsWorkshop.tscn -- `
  --neural-genetics-smoke `
  --neural-genetics-report=C:/Users/forre/Documents/neural-game/outputs/neural_genetics_workshop_godot_report.json
```

The smoke hashes all 178 runtime artifacts, loads all 154 layered atlases,
constructs all 1,080 48px motion regions, and loads all 24 evolutionary images.

Controls:

- `1`, `2`, `3`: categorical fusion, learned latent, evolution;
- `Q`, `E`: previous/next specimen;
- `W`, `S`: previous/next clip, or switch evolution generation;
- `Z`, `X`: presentation layer;
- Left/Right: frame scrub;
- Space: play/pause.
