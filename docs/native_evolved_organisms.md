# Native evolved organisms

`game/EvolvedOrganismLab.tscn` connects the sealed production EMA-FSQ
evolution bank to the pixel-cell organism simulation. It is additive; the game
still starts `Arena.tscn`.

The source authority is the current `run2` evolution bank, not the stale
historical directory. All 36 selected descendants from three generations are
compiled without changing their semantic sprite fields:

- one physical cell per non-aura visible neural pixel;
- 14,457 physical cells and 48,569 breakable bonds;
- 580 named organs, including 93 eyes;
- blood, hemolymph, sap, phase ichor, or coolant networks;
- all five morphology families in every generation;
- all six learned latent fusion modes and all six mutation modes;
- exact parents, latent seed/alpha, lineage hash, operator pair, and fitness;
- metabolism, feeding, damage, fracture, bleeding, regeneration, and
  reproduction in native Godot without Python.

Runtime reproduction inherits and mutates metabolic genome traits. It does not
run the neural codec or regenerate a new categorical body in the shipped game;
that limitation is explicit as `runtime_offspring_redecode=false`. Forge-time
multi-parent shape evolution remains the authoritative structural reproduction
path.

Build, validate, replay, and project:

```powershell
python -m forge.evolved_cellular_organism build
python -m forge.evolved_cellular_organism validate `
  outputs/evolved_cellular_organism_v1/evolved_cellular_organism_manifest.json
python -m forge.evolved_cellular_organism replay `
  outputs/evolved_cellular_organism_v1/evolved_cellular_organism_manifest.json
python -m forge.evolved_cellular_organism_sync `
  --report outputs/evolved_cellular_organism_sync_report.json
```

Native smoke:

```powershell
C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe --headless `
  --path C:\Users\forre\Documents\neural-game\game `
  res://EvolvedOrganismLab.tscn -- `
  --cellular-organism-smoke `
  --cellular-organism-report=C:/Users/forre/Documents/neural-game/outputs/evolved_cellular_organism_godot_report.json
```

The smoke loads all 36 runtime JSON anatomies and exercises damage, fracture,
fluid leakage, feeding, and reproduction. Controls are the same as the base
cellular lab: `Q/E` species, `V` view, `F` feed, `R` reproduce, `Space` blast,
left-drag tear, right-click food, `B` bonds, and `P` pause. The native lab is
top-down: detached tissue remains planar and leaked fluid forms surface puddles.
