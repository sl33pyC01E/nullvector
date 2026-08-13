# Reference-normalized conditioning audit

The v3 conditioning diagnostic answers a narrower and more defensible question
than raw nearest-centroid accuracy: does a neural sprite preserve the condition
signal that the same diagnostic can recover from its exact held-out source?

This matters because the v2 centroid diagnostic is deliberately lossy. On the
authoritative held-out fields it recovers only 36.17% of subtype labels and
30.27% of complete family/subtype/role triples. Those values are measurement
ceilings, not generator acceptance targets.

`forge.multifield_conditioning` verifies the immutable bank and benchmark
schemas, strict checkpoint/corpus/source provenance, every recorded artifact
size and SHA-256, every raw NPZ key/shape/dtype, source conditions, genes,
sanitized guide, held-out targets, field hashes, and recomputed v2 validation.
It then reports, per axis and jointly:

- generated and exact-source match rates on the identical 80 conditions;
- generated/reference prediction agreement;
- the paired both-correct, generated-only, source-only, and both-wrong table;
- retention when the reference is classifiable and correction when it is not;
- an exact two-sided McNemar test for a generated regression;
- the complete 2,560-reference diagnostic ceiling;
- independently recorded full-mask counterfactual NLL preference for
  morphology/subtype, role, and genes.

Run it without allocating the GPU:

```powershell
python -m forge.multifield_conditioning `
  --bank outputs/production_handoff_v2/final_best_stratified80_bank_attempt1/generation_manifest.json `
  --benchmark outputs/production_handoff_v2/final_best_stratified80_benchmark_attempt2.json `
  --cuda-interventions `
  --output outputs/multifield_conditioning/final_best_stratified80_audit_v3.json
```

The output is immutable and schema validated. It never changes a v2 checkpoint,
bank, benchmark, training source, or selection score. A training change is
recommended only when the paired test finds a statistically significant neural
regression; a low score from a classifier that is equally low on authoritative
references is not sufficient evidence for auxiliary losses or guidance.

`--cuda-interventions` is a bounded eight-sample smoke probe. It replays the
baseline exactly with freshly reset per-sample generators, then changes one
axis at a time while holding the guide, all other axes, temperature, and random
seeds fixed. The family probe changes to the corresponding family-local subtype;
the subtype probe stays within the family; role changes alone; genes are
inverted. Output Hamming and silhouette changes prove causal sensitivity only,
not the validity of an intentionally counterfactual combination.
