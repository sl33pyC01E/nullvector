# NULLVECTOR // Creature Stage Reboot

## Product target

The playable product is a native 2.5D creature-stage simulation. It is not a
gallery, arena survival clone, or lab dashboard. The player inhabits one
cellular organism inside a persistent, extremely large ecology and can hunt,
graze, scavenge, mate, mutate, heal, construct, migrate, and discover unusual
biomes and organisms.

The camera is top-down and movement occurs on the ground plane, but organisms
remain vertically aligned: locomotor tissues occupy the lower silhouette,
sensory tissues occupy the upper silhouette, organs sit inside a chassis, and
detached matter settles only to the organism's ground/shadow plane.

## Neural authority

Deterministic native code is the bounded runtime substrate: tensor inference,
spatial indexing, persistence, viewport, input, validation, and safety clamps.
Game-producing authority belongs to model outputs:

1. A coordinate-conditioned morphology network emits occupancy, tissue,
   organ, bond, symmetry, and appendage fields from a heritable genome.
2. A recurrent controller consumes directional sensory rays, internal organ
   state, memory, and local social/resource signals. It emits movement,
   locomotor phase, attention, feeding, attack, mating, construction, and
   anomaly/utility channels.
3. A continuous neural field emits biome, elevation, substrate, nutrient,
   moisture, temperature, radiation, and resource potentials for world
   coordinates.
4. A neural cellular automaton advances wounds, fluid diffusion, scars,
   infection, repair, growth, and developmental programs.
5. A neural presentation decoder eventually emits native cell appearance and
   material response. Until that model is integrated, rendering is a strict
   visualization of neural morphology fields rather than a separate sprite.

Every network has a versioned weight hash and every generated organism records
its genome, parent lineage, decoder hash, controller hash, and world seed.

## Three-stage research program

### Stage 1 — causal procedural scaffold

Build the complete playable shape with deterministic and procedural systems:
controls, cellular anatomy, organs, damage, fluids, locomotion, senses,
metabolism, ecology, reproduction, traits, societies, cities, world streaming,
quests, and presentation. This is not disposable mock code. It is the causal
oracle, curriculum generator, intervention harness, quality baseline, and
fallback implementation for every later model.

The scaffold must be fun and aesthetically coherent on its own. A model is not
allowed to hide a weak game design problem.

### Stage 2 — validated neural ensemble

Replace bounded authorities one at a time while preserving identical public
contracts and counterfactual tests:

- coordinate morphology decoder;
- recurrent creature controller;
- physiology/trauma/repair cellular automaton;
- continuous world and ecology fields;
- settlement/culture dynamics model;
- action-conditioned motion model;
- continuous VAE presentation decoder.

Every model is evaluated on exact replay, long rollout stability, family and
trait separability, action adherence, organ intervention causality, topology,
temporal coherence, diversity, resource conservation, and actual play quality.
An ensemble checkpoint is promoted only when the entire combined simulation is
playable and its failure modes are bounded.

### Stage 3 — monolithic action-DiT reverse distillation

The deployment target is one recurrent action-conditioned model:

```
previous latent world state + controls + memory/context
    -> next latent world state + visible frame
```

A diffusion transformer advances structured latent tokens and a continuous VAE
decodes the viewport. Training retains auxiliary prediction heads for cell ID,
organs, bonds, depth/ground plane, fluids, events, resources, entity memory, and
coarse off-screen world state. These heads make causality observable and stop a
beautiful renderer from silently forgetting severed organs, inventories,
lineages, or distant cities.

Reverse distillation uses scaffold and ensemble rollouts, player traces,
adversarial interventions, rare failures, and long-horizon world histories. The
student must reproduce capabilities and causal state transitions, not merely
imitate screenshots. The final runtime may expose one model endpoint while
retaining validation heads in research builds.

### Causal trajectory contract

The native scaffold can emit fixed-rate teacher rollouts with
`--creature-stage-trace=<path>`. Version 1 records 240 transitions at 30 Hz.
Each transition contains the exact action, before/after player position and
velocity, full organ-capacity snapshot, genome scalars, alive/total organ cell
counts, five-channel local neural world field, chunk and biome, nearest
resource, inventory, objective state, active ecology count, projectile state,
construction count, and society discovery count.

`python -m forge.creature_stage_trace <trace.json>` validates the strict schema,
bounded size, duplicate-free JSON, exact producer transition hash, continuous
state chain, normalized controls, finite values, organ conservation, movement,
physiology change, and action coverage. Two independent native rollouts must be
byte-identical before a scaffold change can become a training authority. The
trace format is deliberately model-neutral so controller, dynamics, VAE, and
monolithic students can share the same replay evidence.

This structure is the technology claim: an AI game can be authored with taste,
trained against explicit causal systems, inspected, reproduced, and improved
rather than emitted as a one-shot opaque artifact.

## Scale model

The world is addressed in signed 64-bit chunk coordinates and has no authored
edge. Simulation uses three tiers:

- **Exact:** visible chunks contain individual organisms, body cells, fluids,
  projectiles, structures, and active neural state.
- **Cohort:** nearby dormant chunks contain species cohorts, resource pools,
  nests, structures, diseases, and migration flux.
- **Field:** remote regions retain only neural-field coordinates plus sparse
  history deltas. They are reconstructed deterministically when approached.

Cost follows the player's active frontier rather than total explored area.
Far ecology is advanced in bounded analytical epochs and materializes into
individuals only when it becomes relevant.

## Five non-convergent morphology priors

- **Humanoid:** upright paired legs, manipulators, upper sensory crown, central
  protected organs, grasp/build bias. This is one chassis, not the universal
  template.
- **Animalian:** quadruped, crawler, swimmer, or radial locomotor base; mouth
  and sensory crown are offset from the mass; tails and paired limb chains are
  first-class.
- **Plantlike:** rooted basal plate with vertical stem, fronds, bulbs, runners,
  and tessellating daughter nodes. No human limb scaffold.
- **Anomaly:** one or more cores with orbiting islands, phase bonds, nonlocal
  sensory fields, and transformations unavailable to ordinary matter.
- **Machine:** armored chassis, lower tracks/legs/thrusters, sensor mast,
  manipulators, tool hardpoints, ranged weapons, storage, and heat/coolant
  networks.

Family is a prior over topology and metabolism, not a palette label. Genome
variation may cross boundaries, but validation measures family separability,
appendage count, organ placement, chassis aspect, symmetry, topology, and
locomotor/sensory vertical ordering.

The first native scaffold exposes four named chassis programs per family:

- humanoid: balanced, longarm, sixlimb, crowned;
- animalian: quadruped, crawler, longtail, horned;
- plantlike: treeform, rosette, runner, twin-stem;
- anomaly: triad, cross, pentad, halo;
- machine: tracked, walker, hover, crab.

These are structural curricula, not final content categories. Continuous genes
still vary proportion, tissue, repair, metabolism, fertility, bonds, and soft
asymmetry inside each program. The native morphology audit covers 160 seeded
specimens and fails on missing morphotypes, low family-local signature
diversity, disconnected cells, missing vital organs, invalid cell counts,
weak bilateral/radial structure, or sensory tissue below locomotive tissue.
The current bank has 14–28 distinct metric signatures per family and replays
byte-exactly.

## Layered cellular motion authority

The scaffold motion vocabulary is deliberately shared with the neural sprite
forge: `idle_breathe`, `idle_wiggle`, `locomote`, `joy`, `anger`, `fear`,
`confused`, `sleep`, `taunt`, `attack`, `cast`, `hit`, and `death`.

These are not whole-sprite scale or flip effects. Each state produces bounded
targets for anatomical subsets:

- breathing pulses internal respiratory, structural, phase, or armored cells;
- locomotion alternates appendage chains with root-to-tip leverage;
- emotes change appendage posture, sensory attention, neural pulse, or stance;
- attack and cast respect aim while leaving the upright chassis orientation at
  zero rotation;
- hit reaction is a time-normalized impulse that cannot be restarted by a
  lower-priority idle or emote;
- death preserves the death-relative cell layout and settles the whole corpse
  toward its 2.5D shadow plane with bounded damped dispersion. It does not use
  unbounded screen-down gravity and does not pancake cells onto one line.

Motion priority is explicit: death > hit > action > emote > locomotion > idle.
Holding an action therefore cannot reset its clock every simulation tick, and
incidental NPC emotes cannot interrupt trauma or attacks.

The native motion audit runs 20 chassis x 13 motions x 72 frames at 30 Hz. A
twin body receives the same commands and must reproduce every cell position to
within 1e-7. The gate also requires finite coordinates, unchanged organ counts,
zero chassis rotation, meaningful local displacement, a 14-pixel hard motion
bound, non-explosive corpse spread, complete ordered matrix coverage, and 260
unique sampled pose signatures. `forge.creature_stage_motion_audit` validates
the strict schema and recomputes the cross-runtime clip identity hash.

The companion `--creature-stage-motion-sheet=` mode uses only CPU `Image`
operations, so it remains deterministic under Windows headless Godot. Its rows
are the 20 chassis in family/morphotype order; columns follow the 13-state
vocabulary above. The current sheet replays byte-exactly.

### Action-conditioned cell trajectory corpus

`--creature-stage-motion-corpus=<directory>` publishes the complete motion
matrix as an atomic, model-neutral teacher corpus. Every clip contains 72
fixed-rate frames of local per-cell position deltas, plus the exact chassis
anatomy, genes, motion ID, movement and aim vectors, action channels, and
external impact/terminal event. The ordering is deliberately dense and
regular: chassis, motion, frame, cell, then x/y.

The exporter refuses to overwrite an existing destination, writes through a
staging directory, and refuses publication if its reserve would cross the
100 GiB free-disk floor. `forge.creature_stage_motion_corpus` independently
checks the strict schema, exact two-file closure, byte ranges, source hashes,
cell and clip identities, ordered coverage, controls/events, topology metadata,
motion magnitude, death spread, and every binary slice. The loader exposes an
immutable float32 `[72, cells, 2]` tensor for a requested clip.

This corpus is the first bounded curriculum for the Stage 2
action-conditioned motion model. It does not yet encode rendered RGB, fluid
fields, injury evolution, contact forces, or whole-world temporal state; those
remain separate causal authorities until a later aligned corpus joins them.
The exact format and current evidence are recorded in
`docs/creature_stage_motion_corpus.md`.

### Cellular intervention trajectory corpus

The next aligned authority is exported with
`--creature-stage-intervention-corpus=<directory>`. It covers all 20 chassis
under nine paired conditions: intact control, partial wound, wound followed by
healing, lower-body cut, and complete ablation of neural, circulatory,
respiratory, digestive, or sensory cells. Each 180-frame clip contains an
identical 15-frame pre-intervention baseline before the event occurs.

The fixed-width binary preserves per-cell local position, health, and alive
state; eight normalized physiology fields; death; fluid count; and 160 exact
ground-plane fluid slots containing position, velocity, radius, and remaining
life. Capacity fields are not trusted labels: the independent validator
recomputes them from the static organ membership and frame-level alive flags.
It also verifies partial wounds do not kill, healing raises health, cuts remove
cells, every ablation zeroes its target capacity, neural ablation becomes
terminal, fluids expand, unused slots are canonical zero, and every
non-control clip is byte-identical to its control before the intervention.

The exact contract and reproducibility evidence are in
`docs/creature_stage_intervention_corpus.md`. Version 1 deliberately stops
short of bond fracture graphs, detached-polyp identity, scars, infection, or
cell regrowth; those need explicit state before a neural dynamics model can be
credited with learning them.

### Stage 2 cellular motion transformer

`forge.creature_stage_neural_motion` is the first model designed directly
against the native 20-chassis authority. Each anatomical cell becomes a token
containing body coordinates, tissue, organ, side, appendage, centrality, and
eight genome values. Its recurrent input is the preceding cell displacement
and velocity. Family, morphotype, motion state, phase, move/aim vectors,
attack/utility channels, and impact/terminal events use a separate condition
path.

The production model has 27,409,156 parameters: ten 384-wide transformer
blocks with eight attention heads. Global attention supplies organism-wide
coordination; every block also aggregates the exact local eight-neighbor cell
graph. This makes distant limbs coordinate without reducing motion to an
unconstrained whole-sprite deformation. Padded cells remain exactly zero.

Morphotypes 0–1 from every family train, morphotype 2 is validation, and
morphotype 3 is sealed test. The sampler is family-balanced. Production uses
deterministic BF16 segments, immutable optimizer/RNG/EMA checkpoints, a 16 GiB
free-VRAM gate, and the 100 GiB disk floor. A four-step CPU proof validates the
full tensor and optimization path but is not a quality claim. CUDA production
was deliberately not started while another sprite-training process occupied
the GPU. Exact evidence and commands are in
`docs/creature_stage_neural_motion.md`.

## First playable loop

1. Spawn as one of five neural organisms with distinct metabolism and tools.
2. Read the world through visible sensory cones and internal organ meters.
3. Acquire matter by grazing, hunting, scavenging, mining, photosynthesis,
   transmutation, or trade/tool use.
4. Survive organ-specific damage. Broken locomotor cells slow or immobilize;
   neural loss corrupts control; respiratory/circulatory loss causes delayed
   collapse; gut loss prevents digestion; severed fluids diffuse as puddles.
5. Reach nutrient/energy thresholds, find a compatible organism or use a
   family-specific reproductive strategy, and create a mutated descendant.
6. Spend assimilated matter on neural mutations, new organs, stronger bonds,
   appendage changes, senses, or constructed habitat.
7. Migrate toward rare biome signals, nests, ruins, megafauna, storms, and
   world-scale anomaly events.

## Traits, societies, and cities

Genomes expose named, inspectable traits instead of only opaque numeric
latents. Traits may be anatomical (paired graspers, redundant heart, plated
roots), metabolic (lithovore, photosymbiotic, blood filter), neural (echo
memory, pack resonance, dream compass), or cultural (tool tradition, caste
signal, oath recognition). Each named trait is still backed by model inputs and
measurable body/controller effects.

Societies are persistent actors in the same hierarchy as organisms:

- exact nearby citizens retain bodies, organs, memories, inventories, tasks,
  kinship, allegiance, and neural state;
- settlement cohorts retain population strata, genomes, professions,
  production, stores, disease, defenses, and political drives;
- regional polities retain treaties, migrations, wars, religions, trade paths,
  myths, and construction projects.

Cities grow from needs and available matter. Founders choose a civic latent
program; builders place paths, membranes, walls, workshops, farms, nests,
reactors, shrines, reservoirs, and defensive organs in connected districts.
Buildings are cellular conglomerates rather than isolated lethal pixels and
collide through broad continuous hulls. Same-society traffic flows through
door/path fields; hostile bodies collide with defenses.

The intended texture is **Spore creature stage plus Caves of Qud**: playful
embodied evolution alongside strange generated cultures, histories, artifacts,
factions, mutations, ruins, and systemic quests. A creature may found a nest,
join a polity, become sacred livestock, trade organs, steal a machine chassis,
spread as a plant colony, or replace a city's ruling neural cluster.

## Aesthetic direction

Dark mineral substrates, soft volumetric shadows, restrained bioluminescence,
fine native cell pixels, fluid halos, and readable organ color should replace
the flat cyan grid/lab-dashboard look. Creatures use compact silhouettes with
clear negative space. Motion comes from appendage chains, weight transfer,
cell spring lag, recoil, and organ pulse—not global sprite scaling or flipping.

The HUD is minimal: body integrity, neural coherence, circulation, respiration,
digestion/energy, hydration/coolant, current sense target, and context actions.
Research provenance lives in a pause/debug screen, not over the playfield.

## First vertical-slice acceptance

- Five silhouettes are recognizable in monochrome and do not share one chassis.
- Every body has connected cells, at least one controller, circulatory source,
  metabolic organ, sensory organ, and family-appropriate locomotor system.
- Body orientation never rotates or flips; only appendages and aim channels do.
- At least four resource acquisition modes and five distinct controller
  behavior profiles are observable in one seeded run.
- Cell damage changes organ capacities; severing produces persistent parts and
  ground-plane fluid diffusion; death does not explode the body.
- Exact simulation remains bounded with at least 100 active organisms while a
  much larger cohort world persists outside the viewport.
- A seeded headless smoke run exactly replays hashes, population totals, and
  player start state.
