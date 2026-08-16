# Neural manipulation arena

`forge.creature_stage_manipulation_v1` is the first live closed loop around the accepted neural grasper. It is deliberately separate from the trained model package: the arena binds the exact v3 runtime SHA-256, then integrates physical appendage reach, attachment, target mass, cohesion, body recoil, ground bracing, release, throw momentum, feeder collision, digestive connectivity, nutrition reserve, and fullness.

The neural controller remains authoritative for which appendage acts and whether it reaches, grips, braces, releases, or throws. Deterministic code is currently the physics and safety scaffold: it prevents teleportation, rejects impossible inputs, conserves unbraced pair momentum, accounts for braced impulse through the ground, and prevents nutrition without physical feeder contact and a live feeder-to-digestion path.

The closed-loop test runs the five base families independently. Each begins with a food clump away from the body, reaches and attaches to it, carries it to its own anatomical feeder cells, establishes contact, and absorbs more than 0.15 mass. A damaged-feeder case touches the same region but absorbs exactly zero. Throwing requires an existing attachment and produces target velocity plus body recoil.

The contact field is 1.50 cell radii plus the clump radius. This is intentionally tolerant enough for stable autonomous ecosystems while remaining local to the physical feeder; general body overlap cannot feed an organism. Reserve defaults to 1.5 units with 90 seconds of fullness and a four-unit capacity. These are balancing controls, not shortcuts around the mouth collision.

This arena is a verified integration substrate, not yet the full `NatureWorld` ecology. The next integration step is to replace abstract animal/humanoid ground-field uptake with persistent material clumps and let the neural behavior layer choose acquisition, delivery, eating, carrying, and throwing over time. Plants retain root-substrate feeding; anomalies and machines use their aperture and port respectively.
