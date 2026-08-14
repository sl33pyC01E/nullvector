# Neural cellular organism dynamics

This stage replaces the local biochemical update rule—not anatomy authority—with a learned neural cellular automaton. Each native 48×48 sprite pixel is a physical cell. The model receives 85 immutable anatomy channels, eight directed live-bond planes, and twelve evolving state fields, then predicts the next bounded cell state.

The 85 anatomy channels encode chassis occupancy, fourteen tissues, eight cell flags, five families, eight organ systems with core/conduit/effector roles and weights, six healing classes, clot/scar/regrowth tendencies, physical attributes, graph degree, and directional bond conductance. The evolving state is health, internal fluid, nutrient, energy, oxygen, clot, scar, open wound, neural activity, top-down surface fluid, biomass, and viability.

The default model uses 256 hidden channels and ten full spatial residual blocks—about ten million parameters. This is intentionally much larger than the original conservative prototype. Two-step rollout supervision penalizes both state error and update-velocity error, preventing the trivial identity rule from winning simply because physiology changes are locally small.

Training cases are deterministically derived from all 45 symmetry-bred organisms (25,668 physical cells and 85,357 explicit bonds). Randomized radial injuries, elongated cuts, bond severing, hypoxia, starvation, feeding, pre-existing scars, and fluid loss teach the model different failure and recovery regimes. The reference dynamics couple circulation, respiration, digestion, neural function, immune capacity, clotting, healing, scarring, death, biomass conversion, and radial surface-fluid diffusion. No screen-down gravity is used.

This is still a teacher-imitation milestone. The graph and chassis remain authoritative, and v1 is not yet evidence that an ecosystem-scale learned world model can replace every rule. The correct next progression is measured rollout integration, observed trajectory collection, then distillation into a recurrent world model with learned bond formation/fracture and cell birth.

Commands:

```powershell
python -m forge.cellular_nca train --output outputs/cellular_nca/nca_v1 --steps 2048 --segment-steps 256 --batch-size 12
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m forge.cellular_nca evaluate --output outputs/cellular_nca/nca_v1 --device cuda
python -m forge.cellular_nca validate --output outputs/cellular_nca/nca_v1
```

Training is split into immutable fresh-process segments with exact optimizer, EMA, and CUDA generator resume. A supervisor records Windows native failures and retries a segment at most three times. Every launch enforces the 100 GiB free-space floor.
