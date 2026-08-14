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

The native `CellularMotionLab.tscn` loads the symmetry-refined physical bodies,
the motion catalog, and the connected physiology catalog independently. Every simulation step computes a target
for each living cell from its rest position and semantic organ channel, then
adds a bounded impulse before the ordinary spring, fracture, fluid, metabolism
and collision step. Appendages now swing from their body-side attachment cells
instead of spinning around their own centroids. A severed organ is no longer
actuated by the main neural component. Partial neural damage progressively
reduces motor strength and introduces bounded tremor; destroying the neural
organ stops intentional motion.
Motion costs cellular energy. Damage can still kill cells, tear appendages,
break springs and spill internal fluid while a motion is active.

The motion lab keeps its manual whole-scene selector for close inspection, but
derived ecology scenes opt into per-organism motion state. Each organism owns
its clip, facing, epoch, action lock, event cursor and behavior label. This
restores the authored alternating appendage strokes in locomotion, the separate
attack/cast/hit poses, and the low-amplitude breathing/wiggle loops without
replacing the live cell body with a sprite. Non-loop actions are allowed to
finish instead of being reset by the next ecology tick.

Motor authority is physiological. The eight connected cell systems are heart
and circulation, respiratory exchange, digestive tract, brain/neural network,
sensory organs, locomotor tissue, reproductive tissue, and immune/repair
tissue. Destroying a core or severing its conduits reduces that exact system's
capacity. Heart failure starves every dependent system; respiratory failure
depletes oxygen and then injures neural cells; gut failure blocks nutrient to
energy conversion; brain damage removes sensing and locomotion; locomotor
damage scales appendage force; immune damage suppresses clotting and healing;
reproductive damage prevents offspring. The native smoke independently removes
each system's physical core and requires all eight corresponding capacities to
fall to zero.

Cycle to `SYSTEM NETWORK` with `V`, then use `C` to inspect each system in
place. Core cells are bright, effectors are pale-tipped, conduits use the
system color, severed or unreachable members turn red, and system-member bonds
are drawn over the ordinary body springs. The live overlay reports the selected
network's remaining capacity, so a tear can be followed from broken conduit to
lost function without leaving the simulation.

The additive trauma overlay now gives those injuries persistent wound state:
open bonds clot, recent tears attract weakly toward their matching endpoint,
successful reconnection leaves scar tissue, and expired fragments become
biomass or family-specific polyps. See `docs/cellular_trauma_and_repair.md`.

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
