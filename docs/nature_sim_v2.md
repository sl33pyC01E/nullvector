# Nullvector nature simulation v2

This milestone turns the cellular creature substrate into a persistent ecology. It is deliberately a deterministic authority scaffold: every rule is measurable, replayable, and suitable for producing supervision for later neural replacements.

## Simulation contract

An organism owns four coupled layers:

1. A developmental genome describing family mixture, soft traits, organs, bilateral components, articulated appendages, skeleton, and muscles.
2. A living cellular body in which injury, severing, fluids, organ capacity, healing, scarring, incapacity, death, polyps, and biomass are causal.
3. An ecological genome describing diet, metabolism, perception, sociality, reproduction, mutation, and colony behavior.
4. A world state: ground position, velocity, depth, age, life stage, reserves, intent, ancestry, mate/gestation state, and colony membership.

The environment is not an abstract food counter. It is a set of spatial material fields: water, light, minerals, charge, phase flux, oxygen, heat, toxin, living flora, and dead biomass. Consumption removes local material. Plants and anomalies can create particular fields, machines mine charge/minerals, animals eat flora or prey, and humanoids can exploit every family through a broad resource vocabulary.

## Reproduction and mutation

Reproduction is delayed, costly, local, and capacity-limited. Offspring inherit an actual developmental program. Continuous traits blend; matching organs and paired appendages cross over together; mutations can change organ scale, regulatory traits, bilateral appendage length, and—rarely—graft a compatible locomotor pair. Bilateral structures remain reciprocal. Every child is developed and validated before entering the world.

The result is not a recolored clone: offspring cell fields, organs, skeleton, muscles, appendage geometry, metabolism, diet, behavior, and life history can all differ from their parents.

## Life, colonies, and death

Organisms pass through embryo, juvenile, mature, senescent, dead, and decomposed states. They acquire resources, metabolize, move, compete, cooperate, mate, gestate, repair injuries, scar, die, and gradually return material to the environment. Death does not explode a body; cohesion decays and biomass diffuses over time.

Social organisms form spatial colonies. Plant offspring build tessellating clonal patches; animal packs share threat information; machines form service networks; anomalies form phase constellations; humanoids form bands. Colony identity affects movement and mating, and large colonies may fission rather than becoming one global blob.

## Scale

The final world uses four conservative levels of detail:

- Active: cellular bodies, organs, skeleton, fluids, contacts, and projectiles.
- Local: articulated bodies with organ and material ledgers.
- Regional: lineage cohorts with age, health, resource, mutation, and colony distributions.
- Historical: biome fluxes, migrations, extinction/speciation, settlement, and conflict events.

Promotion and demotion conserve population, biomass, energy, lineage ancestry, mutation statistics, and colony membership. Detail changes representation, never history.

## Neural replacement path

The scaffold is intended to be replaced in layers:

- the trained anatomical VAE renders cells and organ fields;
- a recurrent contact-and-muscle controller replaces hand-authored gait timing;
- the causal cellular NCA replaces local physiology updates;
- a behavior transformer replaces goal arbitration;
- developmental diffusion replaces crossover/mutation proposals while the scaffold remains the validator;
- regional/world models replace distant cohort evolution;
- the validated ensemble becomes the teacher for a monolithic action-conditioned DiT/VAE student.

Menus, viewport, and HUD may remain conventional. The simulated world and rendered future are the learned system.

## Acceptance gates

A nature build is not accepted merely because it runs. Long-horizon tests require exact replay, finite state, no spawn-time organ cascade, bounded population, nonzero births and deaths, resource conservation, multiple surviving lineages, morphology and behavior diversity, delayed reproduction, predation, healing, decomposition, colony formation/fission, and tier-transition conservation.

## Selection and organism perception

Selection is ecological rather than a scripted genome rewrite. Heritable anatomy and ecology vary in grounded locomotor efficiency, metabolic cost, perception range, fertility, parental investment, repair, diet, aggression, and other traits. Organisms must survive long enough and acquire enough matter to reproduce. Mate behavior weighs expected offspring viability and local carrying capacity, while the resulting child still has to develop, feed, avoid injury, and reproduce in the same world. The online evolution ledger reports which phenotype clades actually leave descendants; it does not assign traits directly.

Controlled organism view is the union of perfect self-awareness, a broad omnidirectional hearing/proximity bubble, and the organism's longer aimed cone or radial sense. Structures occlude distant sensing. Unattended terrain stays readable but subdued. Previously observed structures and material cells persist as last-known map memory and update only when sensed again. Unseen organisms, loose matter, projectiles, and effects remain fully simulated but never persist in memory: they reappear only when a live sensor reacquires them. Separate 32x32 current-visibility and map-memory fields are recorded beside frames, controls, cellular anatomy, and world state so the final action/world model cannot treat perception as a cosmetic overlay.
