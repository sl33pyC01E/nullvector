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

## Current coupled slice

Implemented and running in the Android FP32 preview:

- Five selectable developmental rigs with hundreds of physical cells each.
- Continuous-cell-VAE per-cell RGB, density, footprint, and subpixel style authority.
- Batched 3.5M-parameter live grounded muscle/contact controller.
- Event-driven 8.01M-parameter grasper policy exported from the accepted desktop checkpoint. It selects an intact appendage, predicts reach/force/release, and drives the same fixed-length articulated chain used by the body rig.
- Skeleton constraints, planted appendage anchors, vertical lock, and analog world-plane travel.
- 125k-parameter ecology intent/steering policy for autonomous organisms.
- Five-family recurrent neural cell physiology with explicit circulation, respiration, digestion, neural, sensory, locomotion, reproduction, and repair fields.
- Cell damage feeds the live NCA; organ failure feeds back into consciousness and locomotion. Healthy neural tissue has a scaffolded homeostatic floor to prevent out-of-distribution long-rollout collapse.
- Physical grasp/feed/strike/scrape/cut/throw controls, all-direction ballistic elevation, cell-bond fracture, skeleton severing, persistent fragments, and diffuse ground-plane fluids. Grasping closes only after hand/material contact; feeding requires contact with live digestive cells and absorbs gradually; throws release from the hand with independent height and a ground-plane shadow.
- Whole-body terrain collision and hostile inter-family collision; same-family bodies can overlap.
- Living sensory-organ health now controls an aimed vision cone and a large near-awareness/hearing radius. Current sight and persistent explored-map memory are fed into the recurrent neural action model; unseen organisms remain simulated but are not rendered.
- Remembered terrain is dimmed instead of blacked out. Sight geometry, labels, physiology bars, diagnostics, and the full HUD are independently presentational toggles; HUD-off leaves the world and creatures visible without disabling simulation mechanisms.
- The desktop macro-resource, colony-role, society/construction, timeline, and five-way counterfactual teacher checkpoints are exported to ONNX and scheduled against the same mobile cellular world. Macro, colony, and society outputs are causal; timeline and counterfactual outputs remain observers.
- The selected organism is re-rasterized from its current posed, damaged cell features at 3.75 Hz. VAE color influences presentation, while physical cell opacity, attachment, and survival stay scaffold-authoritative.
- The compact Android flavor runs the same causal ensemble with quantized action and high-level teacher weights. Five high-level graphs compress from 351.2 MiB to 93.6 MiB, retain at least 99.5% decision agreement on deterministic held-out probes, and complete a live Android world cycle with roughly 432 MiB RSS.

Still incomplete: full city/adventure content, save persistence, broader population scaling, neural replacement of the remaining conservation/constraint scaffold, calibrated mobile quantization, physical-S25 profiling, and a complete parity capture. Society-driven organized structures are present, but this is not yet the intended city game.

The current emulator capture is mechanics evidence, not a gameplay presentation target. No Android release should be promoted until a visible parity capture covers the full desktop foundation and the Galaxy S25 Ultra meets the frame-time and memory gates above.

The prior `android-v0.4.0-preview` release is an internal runtime prototype and is not a gameplay milestone.
