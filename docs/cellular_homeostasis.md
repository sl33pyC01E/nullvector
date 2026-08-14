# Cellular homeostasis and organ failure kinetics

`forge.cellular_homeostasis` is an additive reserve-aware functional layer over
the sealed pixel-cell anatomy and connected physiology bank. It does not alter
one source cell, organ assignment, bond, fluid channel, or system route.

The underlying anatomy still decides whether a heart, respiratory exchange
surface, gut, brain, sensory organ, locomotor tract, reproductive organ, or
immune seed physically survives and remains connected. The homeostasis layer
then separates that structural fact from the organism's remaining oxygen,
energy, nutrients, hydration, shock, consciousness, and local perfusion.

This distinction produces meaningful failure kinetics:

- heart destruction removes perfusion and causes rapidly fatal shock;
- respiratory destruction leaves a brief oxygen reserve, followed by hypoxia,
  neural injury, locomotor collapse, incapacitation, and death;
- digestive destruction prevents food conversion and reproduction even while
  circulation and thought initially remain intact;
- major brain destruction immediately removes consciousness, sensing, and
  deliberate locomotion while a living heart can continue briefly;
- immune, locomotor, sensory, and reproductive lesions remove their own
  capacities without pretending that unrelated organs disappeared.

The immutable report audits all eight core lesions across all 45 organisms
(360 structural cases) and four timed lesion scenarios for one representative
of every morphology family (20 dynamic cases). Subtype-specific visuals and
behavior remain governed by the neural anatomy and motion banks.

```powershell
python -m forge.cellular_homeostasis build
python -m forge.cellular_homeostasis validate `
  outputs/cellular_homeostasis_v2/homeostasis_report.json
```

The layer is deterministic reference logic and requires no neural checkpoint or
CUDA. Native and WebUI runtimes can implement the compact equations without a
Python dependency.
