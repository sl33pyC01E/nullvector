# Output gallery

Selected outputs from the current NULLVECTOR pipeline. These are generated
results, not concept art. Some show accepted runtime systems; others are
clearly labeled research prototypes.

## Physical creatures

### Grounded locomotion and feeding

![Grounded locomotion and feeding](showcase/articulated_grounded_feeding_v12.gif)

The neural target field, muscle controller, and contact controller approach a
material clump. An articulated arm then carries it into live feeder cells. Feet
remain planted and the held object stays constrained to the hand.

### Five acquisition strategies

![Five-family feeding](showcase/articulated_five_family_feeding_v11.gif)

Humanoids kneel and grasp. Animals bite from the ground. Plants siphon through
roots. Anomalies use a phase field. Machines collect with a suspension-mounted
tool.

### Throwing and material response

![Ballistic throw](showcase/articulated_ballistic_throw_v11.gif)

Thrown matter has horizontal momentum, elevation, a ground-plane shadow,
gravity, recoil, and landing behavior.

![Impact modes](showcase/articulated_impact_modes_v11.gif)

Different materials bounce, roll, or thud.

### Damage and severing

![Severed grasper](showcase/articulated_severed_grasper_v11.gif)

The complete arm detaches at the shoulder, drops its payload, twitches briefly,
and settles. The intact shoulder returns to rest.

![Damaged grasper](showcase/articulated_damaged_grasper_v11.gif)

Severe neural and muscle damage leaves residual motion but prevents useful
grasping.

![Feeder ablation](showcase/articulated_feeder_ablation_v11.gif)

Destroying feeder cells prevents absorption even when material reaches the
body.

## Neural motion and rendering

### Cellular locomotion

![Neural cellular locomotion](showcase/neural_cellular_locomotion.gif)

A learned cellular motion loop across the five organism families. Each point
is a body cell.

### Grounded controller

![Neural grounded controller](showcase/neural_grounded_controller.gif)

The accepted grounded controller predicts causal muscle and contact feedback
inside physical constraints.

### Anatomical VAE

![Anatomical VAE motion](showcase/anatomical_vae_motion.gif)

A continuous anatomical graph VAE reconstructing held-out creature motion.

![Anatomical VAE comparison](showcase/anatomical_vae_motion.png)

Target and reconstruction frames across families and motion phases.

## Cells, inheritance, and trauma

| Breeding and symmetry | Cellular trauma | Neural cellular dynamics |
|---|---|---|
| ![Breeding and symmetry](showcase/cellular_breeding_symmetry.png) | ![Cellular trauma](showcase/cellular_trauma.png) | ![Cellular NCA](showcase/cellular_nca_rollout.png) |

These cover inherited chassis variation, organs and fluids, damage, healing,
scarring, debris, and learned local cell-state updates.

## World systems

| Living world | Map themes | Neural topology |
|---|---|---|
| ![Living world](showcase/living_world.png) | ![Map themes](showcase/map_themes.png) | ![Neural map topology](showcase/neural_map_topology.png) |

The playable scaffold includes ecology, resources, organisms, construction,
and settlements. Maps begin as validated semantic topology before art is
rendered. Neural topology remains under evaluation and does not bypass the
safety repair stage.

## World-model research

| World VAE | Sparse Action-DiT |
|---|---|
| ![World VAE](showcase/world_vae_reconstruction.png) | ![Sparse Action-DiT](showcase/sparse_action_dit.png) |

The world VAE reconstructs held-out frames. The sparse Action-DiT prototype
learned localized action edits but did not pass the stricter latent acceptance
gate, so it is evidence rather than runtime authority.

## Storage

This gallery stays small enough to browse on GitHub. Full corpora, checkpoints,
replay banks, failure cases, and evaluation reports remain under the local
`outputs/` tree and are not included in normal Git history.
