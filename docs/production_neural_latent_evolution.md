# Production Neural Latent Evolution

`forge.neural_fusion_production_evolution` breeds the sealed production
EMA-FSQ hybrids forward instead of returning to the older categorical-only
genetics engine.

The v1 experiment has three deterministic generations. It plans 30, 36, and
42 births, applies bounded retry without relaxing legality or rig gates, and
selects 12 elites per generation. Every generation must retain:

- all five morphology families, with two or three elites from each;
- all six learned-latent fusion operators;
- all six mutation operators;
- connected legal categorical fields and bilateral parent contribution;
- a fresh graph rig and valid idle, locomotion, and attack clips;
- recursive parentage whose parents belong to an earlier generation.
- at least 6% categorical-field distance between every selected pair.

Fitness is deliberately multi-objective: parent balance, parent and archive
novelty, boundary readability, emission balance, controlled asymmetry,
occupancy, latent-code diversity and novelty, and motion strength. Connectivity
repair is penalized. No single score is treated as a substitute for the hard
gates.

Compile in a CPU-only process:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
python -m forge.neural_fusion_production_evolution compile
```

Validate the sealed manifest and every artifact:

```powershell
python -m forge.neural_fusion_production_evolution validate `
  outputs/neural_fusion_production_evolution_v1/production_evolution_manifest.json
```

The compiler refuses an existing destination and checks the 100 GiB free-space
floor before birth. Outputs include full semantic fields, binding manifests,
seven-layer motion atlases, a candidate ledger, a recursive lineage graph, and
a three-generation contact sheet. Runtime consumers need only PNG and JSON;
Python and the neural checkpoint remain forge-time dependencies.
