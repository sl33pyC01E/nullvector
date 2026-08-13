# Topology-Locked Neural Map Decoration

Status: implementation contract for the decorative neural pass. The semantic
map generator and deterministic pixel renderer remain authoritative until a
trained decorator passes every gate in this document.

## Scope

The neural pass may improve visual rhythm, thematic clustering, prop/decal
placement, micro-patterns, and glow. It must never decide whether a cell can be
traversed or whether the mission can be completed.

The following source fields are immutable:

- terrain, walkability, hazard, elevation, zone, and navigation cost;
- start, exit, objective, and spawn locations;
- the protected mission-backbone mask and all configured clearance masks;
- terrain autotile and elevation-edge masks, which are exact derived fields.

The neural pass may predict only bounded decorative fields:

- an 8-way terrain micro-variant;
- a theme-local decal class, including an explicit empty class;
- a theme-local non-colliding prop class, including an explicit empty class;
- a 4-way emissive accent level;
- an optional 8-way animation phase for decorative, non-hazard effects.

Collision is not a model output. Occlusion is looked up from the validated
catalog entry and may only affect drawing. Hazard animation remains derived
from the immutable hazard field.

## Safety boundary

```text
versioned semantic map
        |
        +-- immutable topology hash -----------------------------+
        |                                                        |
        v                                                        |
deterministic feature encoder                                    |
        |                                                        |
        v                                                        |
categorical neural decorator                                     |
        |                                                        |
        v                                                        |
legal-class masks + protected/clearance masks                    |
        |                                                        |
        v                                                        |
catalog lookup -> art fields -> renderer                         |
        |                                                        |
        v                                                        |
semantic/hash/topology/artifact validators <---------------------+
        |
        +-- publish atomically, or reject the complete candidate
```

The validator reloads the source map from its manifest instead of trusting the
inference process. It recomputes source hashes, all map invariants, derived
autotile/elevation masks, reserved-cell clearances, and catalog legality. A
failed neural candidate is retained as a diagnostic artifact but cannot be
published as a playable art pack. The deterministic renderer is the explicit
fallback.

## Authoritative map-contract extension (implemented)

Map schema `2.0.0` / generator `2.0.0` persist these additional strict
`uint8[H,W]` arrays:

- `protected_backbone`: every cell modified or cleared by the authoritative
  five-cell-wide mission carve;
- `required_clearance`: the union of start, exit, objective, spawn, and any
  future interaction clearances;
- `decoration_forbidden`: the exact hard mask consumed by renderers and neural
  inference. Initially this is the union of the preceding two masks and hazard
  cells; a future contract version may add authored exclusions without changing
  either source mask.

These arrays participate in the canonical semantic hash. The backbone mask is
marked by the same indexed operation that performs each radius-two route carve;
the clearance mask is marked inside the exact safe-region loops; the forbidden
mask is then formed from the final authoritative union. The manifest carries
their meanings, cell counts, individual/combined hashes, and capture-policy
identity. Hash records name `sha256-canonical-named-arrays-v1`, which binds the
array name, dtype, shape, and C-order bytes. Loading rejects legacy packs instead
of inventing masks, and pack validation requires exact deterministic
regeneration from the recorded seed/theme/config.

## Feature tensor

All inputs are derived deterministically and use a versioned channel table:

| Group | Channels | Encoding |
| --- | ---: | --- |
| terrain | 9 | one-hot semantic IDs |
| walkability | 1 | binary immutable mask |
| hazard | 5 | one-hot, including none |
| elevation | 1 | signed normalized scalar |
| elevation edges | 4 | exact cardinal drop flags |
| terrain adjacency | 4 | exact cardinal match flags |
| zone | 1 | normalized local zone ID |
| zone boundary | 4 | cardinal boundary flags |
| navigation cost | 1 | clamped/log-normalized scalar |
| protected/clearance | 3 | backbone, clearance, forbidden |
| required points | 4 | start, exit, objective, spawn masks |
| distance fields | 4 | normalized distances to the same point classes |
| coordinates | 4 | x, y, radial distance, boundary distance |
| seeded noise | 8 | coordinate-hashed multi-scale noise |

The seed-noise channels are essential. Procedural target variants and sparse
decorations intentionally contain entropy that cannot be recovered from map
semantics alone. Explicit, versioned noise makes that entropy learnable and
replayable instead of encouraging the model to average it away.

Theme, map size, aspect ratio, public seed embedding, renderer version, and
catalog version are global conditions. No feature is read from final RGB
pixels, preventing a circular target leak.

## Model

The first production candidate is a compact categorical refinement network,
not an image diffusion model:

- depthwise residual U-Net backbone with 48-64 base channels;
- theme embedding injected through FiLM at each scale;
- four discrete heads for variant, decal, prop, and emission;
- one optional animation-phase head enabled only after static quality passes;
- masked categorical corruption during training;
- 8-16 parallel refinement steps at inference;
- EMA weights and deterministic categorical sampling.

This keeps outputs aligned to the map grid and supports maps from 32 through
256 cells without resampling categorical labels. A neural cellular-automaton
refiner is the second candidate for local organic clustering, but it must use
the same output heads, masks, provenance, and validation contract.

## Hard output masks

Logits are masked before every sampling step, not repaired only at the end:

1. `variant` is always legal in `[0, 7]`.
2. `decal` is restricted by theme and terrain catalog membership.
3. `prop` is restricted to non-colliding catalog entries and their allowed
   terrain. The empty class is forced wherever decoration is forbidden.
4. `emission` is forced to zero where the selected terrain/catalog combination
   has no emissive role.
5. At most one object class may occupy a cell.
6. Optional density budgets cap each catalog class per connected zone and per
   16x16 window. Budgets are deterministic functions of theme and map size.

Post-sampling validation rejects, rather than fixes, a class outside its legal
mask. Rejection exposes model regressions that silent clamping would hide.

## Training corpus

The teacher corpus is synthesized from the validated procedural map and map-art
forges. A sample stores compact semantic and decorative arrays rather than RGB
renders:

```text
map features:  versioned packed arrays
conditions:    seed, theme, sizes, source/catalog hashes
targets:       variant, decal, prop, emission, optional phase
split key:     canonical seed/theme/config identity
```

Train/validation/test splits are assigned by a hash of the complete map
identity. Multiple crops or augmentations of one map must remain in the same
split. Rectangular crops may be used for training, but validation also runs on
complete 32, 72, 128, and 256-cell maps.

The first corpus target is 32,768 maps balanced across six themes, map sizes,
aspect-ratio buckets, objective counts, and hazard-density buckets. Compact
categorical storage is expected to remain well below the 100 GiB free-disk
floor. Corpus building checks free space before each atomic shard publication.

Augmentation is limited to transforms that preserve the semantic contract:

- cardinal flips and 90-degree rotations, with adjacency/edge channels remapped;
- seeded-noise regeneration from a transformed public seed;
- deterministic guide dropout for non-authoritative distance/noise channels;
- no arbitrary resize, blur, RGB jitter, or topology mutation.

## Loss and evaluation

Training uses class-balanced cross entropy on corrupted cells, plus bounded
regularizers for spatial behavior:

- local two-point statistics per theme and terrain;
- per-zone and per-window density errors;
- object nearest-neighbor distance distribution;
- emission/decal/prop compatibility;
- refinement consistency between adjacent noise levels.

The selection score cannot be dominated by the empty class. It includes macro
IoU and foreground F1 for every head, rare-class recall, catalog legality,
density-distribution distance, and deterministic replay. A checkpoint is
publishable only when all of these hard metrics are exact:

- zero immutable semantic changes;
- zero protected/clearance decorations;
- zero illegal catalog/terrain tuples;
- zero colliding neural props;
- zero source/catalog/checkpoint provenance mismatches;
- byte-identical replay for the same seed and checkpoint;
- every source map invariant still passes;
- every generated artifact hash verifies after disk reload.

Quality metrics are reported separately from safety metrics. Better visual
scores can never compensate for a safety failure.

## Inference manifest

Each candidate records:

```text
format_version
source_map_id and canonical semantic SHA-256
protected/clearance mask SHA-256 values
feature-contract and catalog versions/hashes
checkpoint file SHA-256, EMA tensor hash, and training-corpus SHA-256
guide policy, sampler, step count, temperature, and public generation seed
raw neural field artifact hashes
validated compiled art-field and rendered artifact hashes
fallback status and complete rejection reasons
timings, peak memory, and validator version/source hash
```

Raw outputs are never overwritten by compiled outputs. This permits exact
replay, model comparison, and diagnosis of a validator rejection.

## Delivery stages

1. **Complete:** persist and validate the exact backbone/clearance/forbidden
   masks without changing the underlying terrain/hazard generation path.
2. Extract a versioned feature encoder and legal-class-mask implementation.
3. Build a small balanced teacher corpus and prove byte-identical rebuilds.
4. Train a CPU/CUDA smoke model and exercise rejection/fallback paths.
5. Train the full categorical decorator and compare it blind against the
   deterministic teacher on held-out seeds.
6. Integrate only the validated compiled art fields into ForgeLab. Gameplay
   continues consuming the original semantic map arrays.
7. Add an organic NCA candidate only after the categorical baseline is stable.

At every stage, the old deterministic art renderer remains a complete and
versioned path. Neural decoration is an optional compiler pass, never a runtime
dependency or a source of gameplay truth.
