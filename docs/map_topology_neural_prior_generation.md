# Seeded neural map-prior generation

`forge.map_topology_neural_prior_generation` is the first stage that asks the
frozen map prior to synthesize maps without target latent tokens. It consumes
held-out test conditions, begins with every valid latent cell masked, samples a
seeded top-k proposal, decodes through the frozen EMA VQ codec, and then passes
the immutable raw categorical proposal through the deterministic topology
compiler.

This separation is deliberate:

- `raw/` records exactly what the neural prior and codec proposed;
- `compiled/` records every deterministic repair and exact compiler replay;
- `latent_sample.npz` preserves tokens, uncertainty, and the valid mask;
- `case_manifest.json` binds sampling seed, reveal trace, conditioning,
  frozen checkpoints, source map identity, metrics, and artifacts;
- `preview.png` compares the held-out condition source, raw proposal, compiled
  candidate, and edit overlay without hiding repairs.

The source procedural map supplies only theme, dimensions, point heatmaps, and
global openness/hazard conditions. Its latent target tokens are never loaded by
the generation pipeline. The latent training corpus remains provenance only.

## Sampling contract

The sampler is CPU-only, batch-size one, and independent of global RNG state.
Each iteration computes a stable vocabulary sort, draws Gumbel noise from a
dedicated seeded CPU generator, chooses within the recorded top-k, and reveals
the most confident fraction of the remaining cells. Every valid cell starts as
mask token 512; every invalid padded cell remains zero. The final token tensor,
uncertainty tensor, and every intermediate reveal state are SHA-256 bound.

Default production settings are eight reveal steps, temperature `0.8`, top-k
`16`, two variants for each of the 24 held-out theme/size conditions, at most
two isolated CPU workers, and at most three attempts per worker. The supervisor
preserves stdout/stderr and classifies Windows access violations. Publication
requires at least 100 GiB free and is atomic and resumable.

## Quality boundary

Compiler validity is a safety result, not a model-quality result. Quality is
reported separately:

- unique latent codes and distinct raw/compiled identities;
- raw required-point and radius-one connectivity;
- requested versus generated openness and hazard density;
- cell repair fraction and preservation fraction;
- repair excess over the same held-out procedural source compiled under the
  identical point/config contract.

The bank always keeps `production_promotion_allowed=false`. A deterministic
compiler can rescue a weak proposal, but it cannot make the neural prior good.
The manifest therefore records collapse, condition adherence, and calibrated
repair gates independently and truthfully.

## Frozen held-out bank result

The first balanced bank is frozen at
`outputs/map_topology_neural_prior_generation/seeded_test_v1`.

- 48 cases: six themes x four sizes x two seeded variants;
- 48 unique raw topologies, 48 unique compiled topologies, and 48 unique token
  sequences;
- every case began fully masked and every target latent token remained absent;
- every raw artifact, compiler result, preview, and fresh-process neural replay
  verified exactly;
- mean unique-token count: `12.25`;
- raw required-point connectivity: `9/48` (`18.75%`);
- raw radius-one required-point connectivity: `8/48` (`16.67%`);
- coarse openness/hazard condition adherence: `26/48` (`54.17%`);
- mean compiler repair: `6.9338%`, versus `2.6761%` for the held-out
  procedural references;
- maximum compiler repair: `24.0234%`;
- case-level quality acceptance: `24/48`;
- production promotion: **rejected**.

The immutable bank manifest is
`b83caa0bd0cf81b549fc6edb8a165537dce51d2c15ec7e7029545e7b6bf4b557`.
The contact sheet is
`70b15878b004327ac8d5833fb987a78202fe8c71efaf63beca5e112d8dfcf538`,
and the replay report is
`2b3270d8e0da2934c2795716c9166688d0b94d89f0619f91ab42594d3f87df28`.

| Theme | Accepted | Condition | Raw radius-one | Mean repair | Reference repair |
| --- | ---: | ---: | ---: | ---: | ---: |
| arena | 8/8 | 8/8 | 0/8 | 2.993% | 1.646% |
| rooms | 0/8 | 2/8 | 0/8 | 9.585% | 2.927% |
| caves | 8/8 | 8/8 | 2/8 | 0.543% | 1.240% |
| archipelago | 0/8 | 0/8 | 0/8 | 14.198% | 2.933% |
| garden | 8/8 | 8/8 | 6/8 | 0.152% | 2.169% |
| anomaly | 0/8 | 0/8 | 0/8 | 14.133% | 5.141% |

The current condition gate measures only global openness and hazard-density
error. Passing it does not prove theme grammar. Visual inspection of the
contact sheet and 256-square representatives found:

- arena proposals preserve broad occupied-field mass but simplify deliberate
  arena routing;
- caves and gardens need little repair, although their large-scale compositions
  remain bland compared with the condition sources;
- rooms lose coherent chamber/corridor hierarchy;
- archipelagos collapse toward mostly void, and the compiler must draw a large
  mission backbone through sparse islands;
- anomalies lose the source spiral/ring language and require conspicuous
  route repairs;
- edit overlays are legible and honest, with no clipping or hidden repairs.

This is positive evidence that the masked prior can freely sample diverse,
seed-replayable latent fields. It is also direct evidence that the current
500-step calibration does not model long-range theme structure or mission
connectivity well enough for runtime use.

## Next prior revision

The next neural-prior slice should preserve this bank as a fixed baseline and
train an additive v2 rather than changing the compiler or weakening gates. Its
predeclared comparison should include:

1. full-mask and high-mask training quotas that match free-generation use;
2. multi-scale spatial conditioning instead of relying chiefly on global
   openness/hazard scalars and point heatmaps;
3. topology-aware auxiliary signals for required-point reachability,
   radius-one corridors, and walkable component structure;
4. balanced per-theme acceptance, with explicit archipelago, rooms, and anomaly
   gates rather than an aggregate score;
5. macro-grammar diagnostics that distinguish room hierarchy, island
   distribution, and anomaly rings even when global density is correct;
6. a longer segmented training schedule with immutable milestones and the same
   seeded 48-case replay bank for matched comparison.

Compiler repair must remain a visible downstream safety layer. It must not be
used as a training-time substitute for learning topology or as evidence that a
weak raw proposal is production quality.

## Commands

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'

python -m forge.map_topology_neural_prior_generation generate `
  --destination outputs/map_topology_neural_prior_generation/seeded_test_v1 `
  --corpus outputs/map_decorator_corpus_v1 `
  --variants 2 --sampling-steps 8 --temperature 0.8 --top-k 16 `
  --workers 2 --max-attempts 3

python -m forge.map_topology_neural_prior_generation validate `
  --destination outputs/map_topology_neural_prior_generation/seeded_test_v1 `
  --corpus outputs/map_decorator_corpus_v1
```

`validate --exact-cases` reruns all neural samples in the calling process. The
production generator already performs fresh-process exact replay for every case
before it publishes the bank; the monolithic option is intended only for a
bounded follow-up on a stable host.
