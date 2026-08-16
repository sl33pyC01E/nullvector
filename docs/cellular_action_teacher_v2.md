# Cellular action teacher v2

The v1 frame teacher records pixels, controls, and 64 world-level ecology features. That is not sufficient to predict organ damage or cell-material outcomes: the selected creature's fluid, scars, neural connectivity, graft topology, and tissue health may be hidden at render resolution.

V2 adds privileged deterministic scaffold context for neural training:

- a 128-value actor vector covering family, stage, intent, developmental and ecological genes, diet, seven physiological systems, energy/reproduction state, motion, damage/connectivity, resource consumption, neural actuator state, component counts, per-tissue health, fluid, and scars;
- an 8x32x32 float16 cellular field containing occupancy, health, fluid, scar, core connectivity, neural tissue, vital-organ tissue, and locomotor cells.

The world image remains the prediction target. These fields are teacher inputs and auxiliary truths, not a replacement renderer. A later recurrent visual student can learn to infer the hidden state from prior frames, after the privileged ensemble proves that the state is predictable and useful.
