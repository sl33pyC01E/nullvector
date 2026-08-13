# Neural Map Topology Model

Status: researched design contract with an implemented isolated CPU foundation.
It is not a production checkpoint or a replacement for the authoritative
topology-v2 generator.

## Goal

Learn a controllable prior over the structural fields of valid top-down maps
without making neural output the safety authority. The model should learn the
global shape language of the six current themes and support inpainting,
blending, and latent exploration. A deterministic compiler must still prove or
repair mission reachability, agent-radius clearance, point safety, legal terrain
and hazard tuples, and exact topology-mask provenance before a generated map can
be published.

The design deliberately combines three ideas:

- a compact discrete topology codec inspired by VQ-style autoencoders;
- parallel masked-token refinement, following MaskGIT and absorbing-state
  discrete diffusion rather than Gaussian RGB diffusion;
- generation-as-repair, so hard validity is established by a small,
  independently testable compiler rather than inferred from model confidence.

This matches the existing categorical sprite diffusion philosophy: neural
models propose structured discrete content; deterministic code owns validity.

## Research basis

- [PCGML survey](https://arxiv.org/abs/1702.00539): map representation,
  data scarcity, repair, critique, and controllable generation are first-class
  PCGML concerns.
- [Multi-domain VAE level blending](https://arxiv.org/abs/2006.09807): a latent
  model is useful for recombining structural motifs across domains.
- [MaskGIT](https://arxiv.org/abs/2202.04200): bidirectional masked-token
  refinement supports parallel sampling, inpainting, and editing.
- [D3PM](https://arxiv.org/abs/2107.03006): absorbing-state categorical
  diffusion is a principled fit for finite semantic vocabularies.
- [Simplified masked diffusion](https://arxiv.org/abs/2406.04329): masked
  diffusion can be trained as weighted categorical cross-entropy and supports
  state-dependent masking schedules.
- [Path of Destruction](https://arxiv.org/abs/2202.10184): iterative learned
  repair is effective for playable tile-level generation with small datasets.
- [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937):
  the original VQ-VAE establishes straight-through discrete latents and
  codebook learning; the CPU foundation uses the EMA form of this quantizer.

## Authority boundary

The learned system has three explicitly different artifact classes.

1. `raw_neural_topology`: immutable sampled categorical predictions and model
   provenance. These may be invalid and are never loaded by the game.
2. `compiled_topology`: a deterministic projection/repair result with a complete
   edit ledger. It is still not a map pack until every topology-v2 invariant and
   exact replay gate passes.
3. `map_pack_v2`: the existing authoritative runtime contract. No loader should
   ever silently reinterpret a raw neural artifact as this type.

Every compiled artifact must bind the raw artifact hash, checkpoint and EMA
hashes, corpus/split hashes, codec and diffusion source hashes, compiler source
hash, repair ledger hash, compiled-array hash, and complete validator report.

## Training corpus

The existing frozen decorator corpus is an appropriate first source because it
already contains 3,096 full topology-v2 maps across:

- six themes;
- eight size/aspect profiles;
- four objective-count buckets;
- strict train/validation/test identity separation;
- all semantic arrays, mission points, and exact captured topology masks.

The topology loader must be a new read-only, bounded reader. It must validate
the corpus root manifest, shard sidecar, ZIP member table, array shapes/dtypes,
per-member hashes, full-map identity, split, and topology-v2 replay. It must not
import decorator targets as structural supervision.

Recommended first-stage fields:

| field | representation | authority after compilation |
| --- | --- | --- |
| terrain | categorical, 9 classes | neural proposal, compiler may edit |
| hazard | categorical, 5 classes | neural proposal, compiler may clear |
| elevation | categorical signed bins | neural proposal, compiler clamps |
| walkability | derived from terrain | deterministic only |
| nav cost | derived from terrain/hazard/elevation | deterministic only |
| zones | deterministic connected partition | deterministic only |
| mission points | conditioning channels | immutable user/generator input |
| protected backbone | conditioning/diagnostic target | freshly captured by compiler |
| required clearance | conditioning/diagnostic target | freshly captured by compiler |
| forbidden decoration | derived union | deterministic only |

Walkability, navigation cost, zones, and authoritative masks must never be
independently sampled because independent heads could disagree about the same
cell.

## Stage A: discrete topology codec

Use a fully convolutional VQ autoencoder with an exact integer patch scale of
four. Inputs are categorical embeddings plus continuous coordinate and mission
conditioning channels; no bilinear resize is permitted.

Suggested production candidate:

- encoder/decoder width 96, residual depth 2 per scale;
- latent grid `ceil(H/4) x ceil(W/4)` with exact right/bottom padding metadata;
- 512 codebook entries, 96-dimensional codes;
- EMA codebook updates, dead-code replacement from a deterministic reservoir;
- categorical cross-entropy heads for terrain, hazard, and elevation;
- auxiliary boundary, walkable-union, distance-transform, and mission-route
  losses, used only during training;
- theme, objective bucket, dimensions/aspect, hazard budget, openness, and
  route-complexity conditioning;
- no alpha/RGB loss and no continuous reconstruction of categorical IDs.

Codec selection must report per-theme and per-size confusion matrices, rare
terrain/hazard recall, boundary F1, walkable IoU, connected-component errors,
agent-radius reachability errors, codebook perplexity/utilization, and exact
decode determinism. Empty/background accuracy is not a useful selection score.

The codec is a representation model, not a generator. Its first acceptance gate
is faithful reconstruction of held-out maps and all 24 size sentinels.

## Stage B: masked latent diffusion

Train a bidirectional transformer or spatial U-Net over the discrete latent
code grid with one absorbing mask token. The first implementation should favor
a compact axial/spatial transformer because map sizes vary and long-range
mission relationships matter.

Training:

- randomly mask 5-100% of latent codes with a cosine schedule;
- include structured masks: rectangles, corridors, one side of a map, and
  sparse edit masks;
- condition on theme, dimensions/aspect, objective count, start/exit/objective
  heatmaps, openness, hazard budget, and optional source-style blend weights;
- apply condition dropout for classifier-free guidance;
- weight rare code targets and mission-near latent cells without allowing those
  weights to dominate the global topology;
- keep codebook and codec frozen for the first diffusion run.

Sampling:

- begin fully masked for generation or partially masked for inpainting;
- reveal the highest-confidence stable subset each step with row-major tie
  breaking;
- use a dedicated recorded generator seed;
- optionally reject/resample locally when the decoded proposal exceeds
  predeclared occupancy or rare-token bounds;
- preserve all immutable user-painted latent cells exactly.

The raw sampler must expose uncertainty maps and per-step token states. It must
not call the compiler internally; proposal quality and repair cost must remain
separately measurable.

## Stage C: deterministic topology compiler

Compilation is a pure seeded transform from raw semantic arrays plus immutable
mission points to topology-v2-compatible arrays. It publishes a complete,
ordered edit ledger. No edit category may be implicit.

Required phases:

1. Reject malformed arrays, vocabularies, noncanonical dimensions, unsafe
   points, and invalid provenance.
2. Establish a sealed outer boundary and normalize terrain/hazard/elevation
   tuples.
3. Select the primary walkable component using mission-point support and stable
   row-major tie breaking.
4. Connect start, exit, and objectives with a minimum-edit route planner whose
   cost favors existing neural walkability. Capture every route-carve write in
   `protected_backbone`; do not reconstruct it later.
5. Enforce the square radius-one agent contract using exact erosion and
   minimum-edit widening. Record each widened cell separately.
6. Clear and capture required safe disks around mission points and spawns.
7. Remove hazards from backbone/clearance and enforce theme-local hazard
   legality and density bounds.
8. Derive walkability, navigation cost, connected zones, and
   `decoration_forbidden = backbone | clearance | hazard`.
9. Run full topology-v2 validation and deterministic seed replay. Publication
   fails if any invariant or replay comparison fails.

The compiler must expose at least these quality costs:

- terrain cells changed, added-walkable cells, removed-walkable cells;
- route cells carved and radius-one widening cells;
- hazard cells cleared/retyped;
- per-field Hamming distance from raw proposal;
- distance from every mission point to the nearest untouched neural cell;
- pre/post map-quality metrics, including agent-scale mission articulations;
- `neural_preservation_fraction` and `repair_fraction`.

A technically valid map with excessive repair is not a good neural sample.
Production acceptance should initially require a calibrated repair fraction,
then compare that distribution by theme and size rather than hiding it behind a
single all-valid rate.

## Quality and diversity gates

Safety gates are exact and binary:

- all topology-v2 invariants pass;
- exact replay passes;
- no mission point moves;
- no invalid terrain/hazard/elevation tuple remains;
- topology masks are captured at actual write sites;
- no raw artifact is imported as a runtime pack.

Quality gates are statistical and must be reported separately:

- raw proposal validity before repair;
- repair fraction and edit category distribution;
- held-out reconstruction and conditional adherence;
- theme classifier and size/aspect adherence, normalized by reference ceiling;
- occupancy, hazard density, elevation range, route length/detour, clearance,
  chokepoint, zone, and spatial-frequency distributions against held-out maps;
- latent code utilization and nearest-training-map distance;
- exact duplicate rate and semantic-array hash uniqueness;
- pairwise diversity at fixed conditions and conditional responsiveness at
  fixed seed/noise;
- visual contact sheets at native tile scale using the authoritative map-art
  compiler, with raw/compiled/edit-overlay columns.

No score based primarily on empty cells is acceptable. No safety metric should
be presented as evidence of learned visual quality.

## Failure containment

The host has demonstrated nondeterministic native process corruption. Corpus
audits, codec training segments, diffusion training segments, sampling batches,
and replay validation must therefore run in fresh bounded processes with:

- unique staging directories and atomic publication;
- exact source/corpus/checkpoint hashes at process start and before publish;
- no more than three attempts per work unit;
- Windows exit-code, stdout/stderr, PID, elapsed-time, and retry telemetry;
- immutable checkpoints at short segments, including optimizer, EMA, RNG, and
  sampler state;
- a 100 GiB disk-floor check before every large write;
- no destructive checkpoint rotation.

## Implementation order

1. Bounded corpus reader and topology tensor contract.
2. Deterministic compiler/repair ledger with exhaustive property tests.
3. Small CPU codec smoke and strict checkpoint/replay contract.
4. CUDA codec calibration only after the current decorator trainer releases the
   GPU; train to a frozen reconstruction milestone.
5. Masked latent diffusion CPU smoke, then segmented CUDA training.
6. Raw-versus-compiled evaluation bank and visual audit.
7. Only after those gates, additive Native Workshop integration. The current
   procedural topology-v2 generator remains the production fallback.

## Implemented CPU foundation

The isolated `forge.map_topology_neural` package now implements stages 1-3 of
the order above without editing the authoritative generator, decorator trainer,
game, or frozen corpus. It is deliberately a foundation rather than a claim of
production generation quality.

The corpus reader pins both the semantic corpus identity and the complete root
manifest file, then verifies validation, sidecar, artifact, ZIP/NPY, split,
sample, and direct point identities. It hashes the full stored shard but inflates
only `terrain`, `hazard`, `elevation`, seed, theme, and point arrays. The
53-channel decorator feature tensor is never materialized.

The versioned tensor contract has exactly three categorical sampled fields.
Walkability, navigation cost, zones, and topology masks are absent from the
sampled tensor. Four immutable point heatmaps and fourteen bounded global/theme
conditions accompany exact right/bottom padding to a scale of four. Cropping
removes only the recorded padding; no resize path exists.

The deterministic compiler uses stable minimum-edit routing, square radius-one
widening, captured write-site masks, safe point/spawn disks, theme-local hazard
normalization and density caps, and authoritative `assert_valid`. Its ordered
ledger includes every mutation to a sampled field or captured mask. Replay
reconstructs arrays, ledger, report, and validation from the raw artifact and
current source dependency hashes, so a locally rehashed forged ledger is still
rejected. Pre/post diagnostics include agent-scale mission articulation counts;
those quality metrics remain separate from binary safety gates.

The compact fully convolutional codec has categorical reconstruction heads,
theme/global/point conditioning, a straight-through EMA VQ codebook, and exact
scale-four geometry. Its bounded smoke is limited to one or two CPU steps. The
checkpoint contains model, model EMA, optimizer, dedicated training-generator,
and CPU RNG state plus source, tensor-contract, and frozen-corpus hashes. Loading
uses `weights_only=True`, CPU mapping, pre-load file hashes, strict sidecar
schema, and recursive tensor/container byte bounds. It is explicitly
`representation_only_not_generative`.

The adversarial test surface covers illegal categorical IDs, disconnected
mission points, one-cell corridors, route hazards, open boundaries, forged and
rehashed ledgers, ZIP traversal and duplicate members, member header size,
dtype/shape drift, root-manifest drift, checkpoint provenance, exact pad/crop at
32 and 256 cell limits, and a bounded rectangular fuzz over all six themes.

### Frozen smoke artifact

The visually inspected CPU foundation bank is
`outputs/map_topology_neural_smoke_v1`. It contains 42 files totaling 1,130,172
bytes. Publication recorded 205.866 GiB free against the 100 GiB floor.

- source SHA-256:
  `fa08eb4c8ed837e5d46e75d7e2ed95aa963675db8adf1714fae8bc6d244b0ca8`
- compiler source / tensor contract SHA-256:
  `0ac826b17c8379c7b12933bd4a3e4d0aa4209689d7d3a726ce8a4867e6f43dad` /
  `f0fb16a48fedb94e6cb90abd8f66f997da874bc1938f016b36954f7da5fd6f45`
- frozen corpus SHA-256:
  `16ed5f3b1a661e2bfc2abe9e16c39e9b8caaecba81f50fed6658cc4f73cffab8`
- frozen corpus manifest file SHA-256:
  `fd5ee2e88725262f23ef1943e34aad7f19c1b0886100f43298f93226de2ccbaf`
- smoke manifest file SHA-256:
  `b3e4628fb67877b5231011ac7a24d6222283e2fd4f6c7397c4de3d19a847e576`
- smoke identity SHA-256:
  `252108dfa163f7b609ed6f332acc6f883c06e0883afb1e9b6a1cb70b3a12f3d3`
- contact-sheet SHA-256 (27,669 bytes):
  `89bc769e43c30c20859a0c2636e0a9b03b570f56cf8d2c26ec8af0f7dcc6abc3`
- checkpoint file SHA-256:
  `f840781a447b307eb35ef53cada54a36780a4ba1317ddd1444084d282e0fe71f`
- model / EMA state SHA-256:
  `019eb279abeb812ae02cb2781af7f574afaac187b61e87227983f0671d5d574c` /
  `6aabf0f4fd73a5faf45a25b075a2e55f509551ffa0843b12198042101f42a91f`
- exact codec decode SHA-256:
  `f88f7fc4a2d41c5733f495a600bf1b62c2f3b84aabbf37ef8dde37bdc3699443`

Exact replay compared 72 raw/compiled arrays and 3,496 ordered ledger entries
across six themes, reproduced the contact sheet byte for byte, safely loaded the
two-step CPU checkpoint, and reproduced the codec decode identity. The focused
suite contains 23 tests; the combined topology-foundation, authoritative map,
and map-quality run passed 63 tests.
