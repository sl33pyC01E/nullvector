# Android foundation parity

The Android target is a port of the desktop foundation, not a separate minigame. A build is not playable until the same underlying state drives presentation, physics, physiology, ecology, and interaction.

## Acceptance gates

- Five selectable organism families with distinct anatomy, traits, appendages, organs, and neural VAE cell appearance.
- Live neural muscle/contact inference executed through grounded skeleton constraints. No frame-loop locomotion.
- Analog movement over the full ground plane while sprites remain vertically aligned in 2.5D.
- Physical cells can be damaged, healed, severed, leaked, scarred, killed, and converted to material.
- Grasp, feeder contact, held objects, all-direction ballistic throw, strike, cut, scrape, beam, and projectile interactions.
- Independent projectile ground position and elevation; shadows stay on the ground path while objects rise, fall, bounce, roll, or thud.
- Neural organism intent and steering, local resources, feeding, predation, reproduction, mutation, selection, colonies, and stable ecology.
- Perception cone, awareness radius, map memory, persistent but hidden unseen organisms, overlay toggles, and clean student view.
- World regions, maps, settlements, construction, crafting, trade, quests, grafting, evolution, and persistent saves.
- Mobile neural ensemble provenance, failure-safe fallbacks, per-model timing, memory telemetry, and measured 24–30 FPS on the Galaxy S25 Ultra target.

## Current replacement slice

Implemented but not release-ready:

- Five selectable developmental rigs with hundreds of physical cells each.
- Continuous-cell-VAE per-cell RGB, density, footprint, and subpixel style authority.
- Batched 3.5M-parameter live grounded muscle/contact controller.
- Skeleton constraints, planted appendage anchors, vertical lock, and analog world-plane travel.
- 125k-parameter ecology intent/steering policy for autonomous organisms.
- Physical grasp/feed/strike/throw control surface and cell-local damage substrate.

Still incomplete: full organ physiology binding for every selected body, severing and leakage, feeder/grasper constraints, reproduction/evolution, construction/society/adventure systems, map persistence, clean-view perception, performance profiling on real hardware, and full parity capture.

The prior `android-v0.4.0-preview` release is an internal runtime prototype and is not a gameplay milestone.
