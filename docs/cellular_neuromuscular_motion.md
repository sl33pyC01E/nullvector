# Cellular neuromuscular motion

`forge.cellular_motion` v2 gives the 45 symmetry-refined structural offspring a
deterministic motor program. It does not swap sprite frames. Each body keeps
its live cells, organs, fluids, health and breakable spring bonds while bounded
organ target forces bend it into a pose.

The bank contains:

- 45 exact anatomy identities and 748 partitioned organ mappings;
- five family-specific actuation programs;
- 13 motions: breathe, wiggle, locomotion, joy, anger, fear, confusion, sleep,
  taunt, attack, cast, hit and death;
- eight top-down facings;
- 520 family/motion/facing clips and 4,720 driver frames;
- 14 continuous driver channels for body, head, paired appendages, locomotors,
  auxiliary organs, weapons, senses, emission, propulsion and pain;
- exact duplicate loop endpoints and named gameplay events such as foot plants,
  strike, release, impact and expiration;
- one deterministic attachment-root hinge for every organ, derived from the
  immutable cross-organ bond graph.

Family gains preserve morphology character. Animalians receive stronger leg
locomotion; plantlike bodies emphasize auxiliary tendrils and grow slowly;
machines move with a rigid chassis and stronger propulsion; anomalies have the
largest irregular body/auxiliary motion; humanoids balance arms and legs.

The native `CellularMotionLab.tscn` loads the symmetry-refined physical bodies
and the motion catalog independently. Every simulation step computes a target
for each living cell from its rest position and semantic organ channel, then
adds a bounded impulse before the ordinary spring, fracture, fluid, metabolism
and collision step. Appendages now swing from their body-side attachment cells
instead of spinning around their own centroids. A severed organ is no longer
actuated by the main neural component. Partial neural damage progressively
reduces motor strength and introduces bounded tremor; destroying the neural
organ stops intentional motion.
Motion costs cellular energy. Damage can still kill cells, tear appendages,
break springs and spill internal fluid while a motion is active.

Build and verify:

```powershell
python -m forge.cellular_motion build
python -m forge.cellular_motion validate `
  outputs/cellular_motion_v2/cellular_motion_manifest.json
python -m forge.cellular_motion replay `
  outputs/cellular_motion_v2/cellular_motion_manifest.json
python -m forge.cellular_motion_sync `
  --report outputs/cellular_motion_sync_report.json
```

Native smoke:

```powershell
C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe --headless `
  --path C:\Users\forre\Documents\neural-game\game `
  res://CellularMotionLab.tscn -- `
  --cellular-motion-smoke `
  --cellular-motion-report=C:/Users/forre/Documents/neural-game/outputs/cellular_motion_godot_report.json
```

Controls are inherited from the organism lab. `W/S` selects motion and the
arrow keys select facing. `Q/E` changes species, `V` changes anatomy view,
`F` feeds, `R` reproduces, `Space` blasts, left-drag tears cells, right-click
places food, `B` toggles bonds, `G` toggles gravity and `P` pauses.

The compiler contact sheet is
`outputs/cellular_motion_v2/cellular_motion_contact_sheet.png`. It shows one
family representative at the principal event pose for every motion. The native
lab is the authority for continuous in-between spring deformation.
