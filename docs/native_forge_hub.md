# Native Forge Hub

`game/ForgeHub.tscn` is an additive Godot 4.3 launcher and status console for
the native forge. It does not replace the game: `game/project.godot` still
starts `res://Arena.tscn`, and the protected Arena scene and script remain
byte-identical.

The hub exposes fourteen native destinations:

- Arena and the compact Forge Lab;
- the 80-identity Neural Workshop and neural genetics lab;
- all-80 repaired motion, subtype motion, and cellular motion;
- base, symmetric, evolved, and bred cellular organisms;
- cellular ecology and ontogeny;
- neural decorated topology-v2 maps.

Every card is backed by the current source scene and, except for Arena, a
runtime JSON catalog. Startup loads every scene as a `PackedScene`, checks the
exact catalog format and selected census metric, rejects non-ready catalogs,
rejects catalog errors and Python runtime dependencies, verifies bundle-ID
syntax, and computes source and manifest SHA-256 values. An invalid card is
locked instead of launching stale assets.

## Launch

```powershell
C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe `
  --path C:\Users\forre\Documents\neural-game\game `
  res://ForgeHub.tscn
```

Use the category controls to isolate neural, motion, organism, ecology, or map
labs. `OPEN` switches the current Godot instance to the lab. `DETACH` starts a
second native instance and leaves the hub available. Escape closes the hub.

## Headless audit

```powershell
C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe `
  --headless `
  --path C:\Users\forre\Documents\neural-game\game `
  res://ForgeHub.tscn -- `
  --forge-hub-smoke `
  --forge-hub-report=C:/Users/forre/Documents/neural-game/outputs/forge_hub_smoke_report.json
```

The smoke exits nonzero if a scene cannot load, a catalog contract drifts, a
runtime requires Python, or the project main scene ceases to be Arena. A green
report covers fourteen scenes and thirteen runtime catalogs; it is an inventory
and launchability proof, not a claim that every lab's full behavioral test suite
was replayed inside the hub process.
