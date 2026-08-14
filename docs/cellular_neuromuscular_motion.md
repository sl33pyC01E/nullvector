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

The sealed connected-organ capacity remains the structural authority, while a
separate reserve-aware functional layer determines what the organism can do at
this instant. A torn-out brain removes consciousness and deliberate motion
immediately. A destroyed lung leaves a brief oxygen reserve, then progressively
weakens perception and appendage actuation before incapacitation. A failed
heart drives circulatory shock, a destroyed digestive core prevents feeding
energy from supporting reproduction and repair, and low energy suppresses
motion even when the relevant tissue is still physically connected. The same
functional capacities now gate clotting, healing, locomotion, digestion, and
reproduction instead of serving only as status-panel diagnostics.

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
places food, `B` toggles bonds and `P` pauses. The lab is permanently top-down;
uniform screen gravity is rejected and external fluid diffuses over surface XY.

The native lab now applies each authored driver over a continuous attachment
root-to-tip coordinate. Breathing and idle squash remain coherent at the
chassis, while locomotion, attacks, casting, recoil, and emotes bend distal
appendage cells progressively instead of rotating each organ as a rigid card.
Local motor-tract reachability gates those forces, so anatomical cuts are
visible in the motion itself.

Native physiology now carries the Python authority's graded widest-path
signal rather than reducing every living conduit to a binary connection. A
partially crushed nerve or motor cell weakens its downstream appendage before
it dies; severance still reduces the signal to zero. System-network colors
show this continuum from healthy system color through weak red, and clotting
uses local circulation and immune delivery rather than intact whole-body
counters.

That circulation signal is perfusion-aware: leaking a limb's blood, sap,
ichor, or coolant now weakens its authored action amplitude even while its
cells and motor tract remain physically connected. Refilled intact vessels can
recover the motion; continued hemorrhage or a cut route drives it toward zero.

The compiler contact sheet is
`outputs/cellular_motion_v2/cellular_motion_contact_sheet.png`. It shows one
family representative at the principal event pose for every motion. The native
lab is the authority for continuous in-between spring deformation.

The native smoke now closes the gap between authored curves and actual cell
motion. It instantiates every one of the 45 physical identities and drives five
observed cases per body: chassis breathing, appendage wiggle, left locomotor,
right locomotor, and attack appendage/weapon motion. All 225 cases must produce
finite nonzero velocity in the intended living cell channels through the real
attachment-root, neural, perfusion, energy, and spring-force path. This catches
the failure mode where a clean driver graph exists but a particular anatomy's
appendages remain visually frozen.

Semantic collapse is audited separately so a clip cannot pass merely because
its frame count and numeric bounds are valid:

```powershell
python -m forge.cellular_motion_quality build
python -m forge.cellular_motion_quality validate `
  outputs/cellular_motion_quality_v1/motion_quality_report.json
```

The audit covers all 65 family/motion programs and records active-driver count,
body/appendage/locomotor/expression excursion, temporal energy, event-pose
displacement, exact loop closure, paired gait anti-phase, and distinct
family-specific trajectory hashes. A flattened action or rigid locomotion
curve therefore fails closed before it reaches the native runtime.
