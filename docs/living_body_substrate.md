# Living-body substrate

This package is the deterministic causal authority beneath the creature-stage
demo and future learned dynamics models. It deliberately replaces global hit
points with a connected cellular body:

- every cell has health, tissue, component/organ authority, fluid, scar state,
  and adjacency;
- organs only contribute while their cells remain alive and connected to the
  primary body;
- appendage damage reduces locomotion through actual owned cells;
- circulation, respiration, digestion, sensing, and neural capacity are derived
  independently, so local injury has a legible local consequence;
- opened tissues leak into top-down diffusing puddles rather than falling as if
  the game were side-on;
- slow energy-limited healing leaves scars;
- severed plant, anomaly, and machine components can consolidate into polyps;
  severed humanoid and animal components consolidate into biomass;
- death preserves cellular matter. Nothing explodes into a different material.

The scaffold is deterministic so it can generate paired intervention data and
reject physically or biologically impossible learned transitions. It is not the
intended final runtime authority. A specialist learned dynamics model can
replace each transition once it matches the causal gates; the long-term student
remains the action-conditioned monolithic world model.
