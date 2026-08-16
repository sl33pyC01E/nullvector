# Neural macro ecology and settlement dynamics

`forge.nature_macro_nn` is the first learned transition bridge between the organism simulation and biome/city scale. It observes two consecutive 32x32 macro states derived from a 64x64 authority region, plus climate, ecological, demographic, settlement, and biome context. It predicts the next spatial and global state as sparse gated edits over persistence.

The spatial vocabulary includes ten resource fields, five family populations, energy and body integrity, colony influence, nine building purposes, roads, structures, and material mass/damage/temperature. The global vocabulary retains all eight biomes independently alongside seasons, climate, events, population, ecological history, colony/settlement counts, wealth, food, power, building activity, knowledge, and cohesion.

The deterministic `NatureWorld` and `SocietyLayer` systems remain teacher authorities for this stage. A runtime checkpoint is accepted only if it beats persistence on overall spatial state, changed spatial cells, global state, and an eight-step autoregressive rollout. This is an ensemble-stage neural replacement, not the final monolithic action-conditioned world model.

Typical commands:

```powershell
python -m forge.nature_macro_nn build-corpus outputs/nature_macro_nn/corpus_v1 --worlds 32 --steps 48
python -m forge.nature_macro_nn validate-corpus outputs/nature_macro_nn/corpus_v1
python -m forge.nature_macro_nn train outputs/nature_macro_nn/corpus_v1 outputs/nature_macro_nn/production_v1 --steps 2400 --batch-size 24 --device cuda
```
