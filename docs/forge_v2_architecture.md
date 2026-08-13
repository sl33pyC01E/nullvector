# NULLVECTOR Forge v2 Architecture

Status: active design contract. The validated 32px vehicle forge remains the
production baseline while this system is built beside it.

## Non-negotiable guarantees

- Every artifact is deterministic from a versioned manifest and can be replayed
  independently.
- Neural output is stored before and after bounded cleanup. Invalid raw samples
  are rejected, never silently converted into valid-looking assets.
- Semantic anatomy, rig ownership, emission, and rendered color are separate
  data. A sprite sheet is a compiled artifact, not the source of truth.
- Logical body connectivity is distinct from rendered connectivity. Orbiting
  anomaly parts may be visually disconnected while remaining members of one
  valid rig graph.
- Map objectives are connected before decoration. Neural or stochastic terrain
  may decorate and perturb space but may not destroy the protected mission
  backbone.
- Every generation/training entry point enforces the 100 GiB free-disk floor.

## Creature representation

Morphology and combat role are orthogonal:

```text
morphology = humanoid | animalian | plantlike | anomaly | machine
combat_role = assault | charger | ranged | siege | support | ambient | ...
```

This permits combinations such as an animalian ranged unit, plantlike charger,
or humanoid siege unit without forcing one label to represent anatomy and
behavior simultaneously.

### Body-plan graph

Each generated identity starts as a small logical graph. A node contains:

```text
id, parent_id, semantic_role, part_slot
joint_type: fixed | hinge | slider | orbital | spline
rest_position, angle_limits, length_limits
phase_group, symmetry_group, material_role, z_policy
render_component_policy, optional socket roles
```

Canonical semantic roles are `root`, `axial`, `sensor`, `locomotor`,
`manipulator`, `emitter`, `ornament`, and `field`. Render ownership uses bounded
part slots so a categorical network can predict it while manifests retain
human-readable graph roles.

### Aligned neural fields

The first v2 model targets 48x48 pixels and predicts aligned discrete fields:

- `part_owner`: transparent plus bounded rig-node slots;
- `material`: transparent plus palette/material roles;
- `emission_level`: 0 through 3;
- optional quantized joint/contact heatmaps.

The conditional U-Net receives morphology, subtype, combat role, the complete
continuous genome, graph-node embeddings, rasterized bones/joints, palette, and
faction. Categorical absorbing diffusion remains the base generator because it
preserves hard pixel and label boundaries. The model class already supports
versioned image sizes, token counts, archetype counts, and gene dimensions.

Production training uses the `scaffold_only` guide policy. Exact target-derived
silhouette, body, and core masks are removed before the model sees a batch;
only bones, joint/socket heatmaps, normalized coordinates, and root distance
remain. Deterministic guide dropout and jitter prevent the network from copying
a procedural answer or ignoring subtype/genome conditioning. A `full_debug`
policy may exist for reconstruction diagnostics, but checkpoints trained with
it are not generation checkpoints.

## Morphology grammars

| Family | Required logical structure | Locomotion/idle basis |
| --- | --- | --- |
| Humanoid | pelvis, axial chain, head, two legs, optional arms/equipment | alternating stance, counter-swing, breath/weight shift |
| Animalian | pelvis/chest/head chain, 2-6 limbs, optional tail/wings | walk/trot/gallop phase tables, tail and ear secondary motion |
| Plantlike | root/base, trunk branches, blooms, vines/tendrils | anchored sway, traveling branch waves, root stepping or creeping |
| Anomaly | logical root, orbitals/fields/pseudopods | orbital constraints, phase drift, pulse and controlled separation |
| Machine | chassis, modules, hinges/sliders, locomotor modules | rigid transforms, wheel/track cycles, recoil, hover bob |

## Animation compiler

The identity model generates rest anatomy once. Animation does not independently
diffuse each frame.

1. Pose the graph using a family-specific procedural motion program.
2. Transform owned masks with nearest-neighbor sampling.
3. Resolve per-frame depth through node z policies.
4. Transform and export sockets from the same node transforms.
5. Optionally apply a compact categorical pose refiner conditioned on identity,
   action, phase, and facing.
6. Bake directional atlases, semantic ownership atlases, emission atlases, and
   timing/event manifests.

Required motion vocabulary:

- `idle_breathe`, `idle_wiggle`, `idle_alert`, and morphology-specific ambient
  loops;
- `locomote_slow`, `locomote_fast`, `dash`, and stance/contact metadata;
- `attack_primary`, `attack_secondary`, `cast`, `hit`, and `death`;
- `emote_happy`, `emote_angry`, `emote_fear`, `emote_confused`, `emote_sleep`,
  and `emote_taunt`.

Important animation invariants include binary alpha, safe margins, correct
socket ownership in every frame, bounded pivot jitter, stance-foot stability,
loop closure, facing agreement, and postprocess change below five percent.

The optional learned refiner predicts bounded residual token changes over a
stable procedural pose; it does not regenerate independent RGB frames. This
follows the persistent-layer representation used by deformable-sprite work and
avoids the identity drift of unconstrained per-frame diffusion. Directional
z-order is explicit because articulated sprite research identifies changing
layer order and strong occlusion as first-class failure cases.

## Map forge

```text
mission graph
-> spatial zones
-> protected spanning-tree corridors
-> semantic tile fields
-> topology-specific perturbation
-> hazards/props/spawns
-> deterministic validation and repair
-> visual skin and Godot layers
```

Semantic truth is stored independently from art:

```text
walkable, protected, zone_id, terrain, hazard,
cover, height, nav_cost, spawn_class
```

Mandatory zones form a spanning tree. Connected interiors and every tree
corridor are written into `protected_walkable`; later cellular noise, props,
hazards, WFC, or neural decoration cannot clear those cells. Optional branches
and loops are added afterward.

Validation covers deterministic hashes, player-clearance-eroded reachability,
objective and portal reachability, protected-cell preservation, safe paths that
exclude hazards, spawn safety, zone integrity, combat openness, boss-room
diameter, tile adjacency completeness, and bounded repairs. Repair order is:

1. A* carve unreachable required anchors.
2. Widen insufficient clearance.
3. Relocate hazards, props, and spawns.
4. Regrow only the failed zone.
5. Resample macro layout only after bounded local repairs fail.

Mission, layout, terrain, decoration, and repair use independent deterministic
RNG streams so a local repair cannot reshuffle unrelated content.

A neural cellular automaton is a future *decoration* stage, never a topology
authority. It may grow wall texture, vegetation, ruins, and anomaly patterns on
the semantic lattice while immutable protected cells and semantic validators
gate every result. Quality-diversity NCA research shows useful level-space
coverage, but its solvability objective does not replace the forge's explicit
agent-radius and hazard-free reachability proofs.

## Research basis

- Discrete absorbing diffusion: [D3PM](https://papers.neurips.cc/paper/2021/file/958c530554f78bcd8e97125b70e6973d-Paper.pdf)
- Rig topology and joint placement: [RigNet](https://people.cs.umass.edu/~zhanxu/papers/RigNet.pdf)
- Sprite part extraction and occlusion limits: [APES](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_APES_Articulated_Part_Extraction_From_Sprite_Sheets_CVPR_2022_paper.pdf)
- Persistent layered deformation: [Deformable Sprites](https://openaccess.thecvf.com/content/CVPR2022/papers/Ye_Deformable_Sprites_for_Unsupervised_Video_Decomposition_CVPR_2022_paper.pdf)
- Mission/space graph separation: [Dormans, Adventures in Level Design](https://pcgworkshop.com/archive/dormans2010adventures.pdf)
- Cellular cave construction: [Johnson et al.](https://pcgworkshop.com/archive/johnson2010cellular.pdf)
- Local texture synthesis constraints: [WaveFunctionCollapse](https://github.com/mxgmn/WaveFunctionCollapse)
- Diverse neural cellular automata for levels: [Earle et al.](https://arxiv.org/abs/2109.05489)
- Runtime map layers: [Godot 4.3 TileMapLayer](https://docs.godotengine.org/en/4.3/classes/class_tilemaplayer.html)
- Agent-radius caveat: [Godot navigation meshes](https://docs.godotengine.org/en/4.3/tutorials/navigation/navigation_using_navigationmeshes.html)
- Cutout/debug rigging: [Godot cutout animation](https://docs.godotengine.org/en/stable/tutorials/animation/cutout_animation.html)

## Build order

1. Deterministic multi-family grammar and graph schema.
2. Graph-driven animation compiler and property tests.
3. Deterministic semantic map forge and large-seed fuzzing.
4. 48px graph-conditioned categorical rest-identity model.
5. Godot sprite/map inspection laboratory.
6. Optional categorical temporal pose refiner.
7. Neural tile/decoration generator, constrained by the valid semantic map.
