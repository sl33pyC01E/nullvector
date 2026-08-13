# Deterministic categorical sprite presentation compiler

## Scope and authority

`forge.multifield_style` is a derived presentation stage. The neural model's
aligned categorical fields remain authoritative. The compiler does not modify
or emit replacement part/material/emission fields, rig anchors, animation
drivers, collision masks, or gameplay semantics. Loaded fields are marked
read-only and their aligned hash is checked again after rendering.

The compiler is deliberately CPU-only and does not import torch. Output stays
at the model's native 48×48 resolution. There is no antialiasing or resizing in
sprite artifacts. Contact sheets scale those already-compiled pixels with
nearest-neighbor sampling for inspection only.

## Immutable neural input

The primary input is a canonical `generation_manifest.json` with format
`nullvector-multifield-generation-bank-v1`, status `ready`, and accepted,
hard-valid samples. Before compiling a sample, the loader verifies:

- the published generation-bank JSON schema and strict finite JSON;
- safe project-contained paths and regular non-symlink files;
- artifact byte count and SHA-256;
- a bounded NPZ/ZIP container with an exact member set and no duplicates;
- exact uint8 `[48,48]` part, material, and emission arrays;
- the 17/10/4 categorical domains and exact background tuple;
- embedded raw/processed hashes and the independent aligned-field hash.

The derived bank binds to the exact parent manifest, evaluator source,
checkpoint, EMA weights, trainer source, legal-tuple table, every compiled
field artifact, the style compiler source, and all map-art sources used by
contrast evaluation.

## Presentation layers

Every sample produces these independent native RGBA assets:

| Layer | Contract |
|---|---|
| `base.png` | Opaque non-aura body only; deterministic integer three-tone shading plus a bounded RGB-only asymmetric accent. |
| `outline.png` | Exact one-pixel Chebyshev ring around the body, with sparse chromatic pixels. |
| `emission_core.png` | Opaque color only where non-aura body has emission level 1–3. |
| `aura.png` | Partial-alpha effect support only; categorical aura owner 16 never becomes body. |
| `bloom_r1.png` | Exact Chebyshev radius-1 ring around categorical emission support. |
| `bloom_r2.png` | Exact Chebyshev radius-2 ring around categorical emission support. |
| `composite.png` | Back-to-front composition of bloom, aura, outline, base, and core. |
| `palette.json` | Perceptual palette and family/role vocabulary. |
| `metrics.json` | Thresholds, measurements, every gate result, and aggregate pass state. |

Contact sheets show every layer, transparent composites, and composites over
deterministically selected native crops from all six existing map-art themes.

## Perceptual palette

The compiler implements bounded OKLCH-to-sRGB conversion directly. Requested
colors outside sRGB reduce chroma using a fixed 24-iteration binary search, so
the result does not depend on an OS color profile or a heavy color dependency.

All ten materials receive independent shadow/mid/highlight ramps. Family hue,
chroma, and pixel motifs distinguish humanoid, animalian, plantlike, anomaly,
and machine presentation. Role hue and motifs control accents and emission.
The source field hash and sample seed introduce only bounded deterministic
variation.

## Objective gates

Compilation fails closed if any sample misses a gate:

- categorical field hash unchanged;
- native 48×48 uint8 RGBA layers;
- exact base, outline, emission-core, and radius-1/2 bloom supports;
- unclipped two-pixel effect rings;
- bounded partial-alpha aura/bloom that never becomes body;
- one 8-connected non-aura body component;
- asymmetric accents bounded to 2.5% of body pixels with unchanged alpha;
- bounded base and composite palette sizes;
- clipped-white fraction at or below 1.5%;
- ten distinct material mid-tones with minimum OKLab separation 0.045;
- strictly monotonic emission lightness with minimum step 0.055;
- minimum contrast against 18 source-hashed crops (three for each of arena,
  rooms, caves, archipelago, garden, and anomaly).

The contrast threshold is a perceptual presentation smoke gate, not a claim of
accessibility conformance or visibility under every possible dynamic effect.

## Replay and tamper behavior

Replay validates the derived schema, current compiler hash, parent generation
bank, categorical artifacts, map-art catalog, and every derived artifact's
size/SHA-256. It then regenerates all layers, JSON records, and contact sheets
and compares the exact encoded bytes. Any mismatch returns `passed: false`.
Output directories are immutable: compilation refuses a non-empty destination.

## Five-family reference (not neural)

The calibrated epoch-20 neural smoke bank contains humanoids only. The separate
`procedural-reference` command validates the existing authoritative morphology
prototype manifest and archive, checks each specimen's semantic and training
array hashes, selects one specimen per family, and runs the same presentation
gates. Its manifest uses the distinct
`nullvector-multifield-style-procedural-reference-v1` format and hard-codes:

```json
{
  "source_kind": "authoritative-procedural-reference",
  "neural_output": false,
  "authority": {
    "procedural_reference_only": true,
    "neural_output_claimed": false
  }
}
```

This reference demonstrates style vocabulary and must not be used as evidence
that the current neural checkpoint generates all five families.

## Commands

```text
python -m forge.multifield_style compile \
  outputs/multifield_generation/milestone_epoch020_cpu5_calibrated_final/generation_manifest.json \
  outputs/multifield_style/neural_epoch020_smoke_v1

python -m forge.multifield_style replay \
  outputs/multifield_style/neural_epoch020_smoke_v1/style_manifest.json \
  --report outputs/multifield_style/neural_epoch020_smoke_v1.replay.json

python -m forge.multifield_style procedural-reference \
  outputs/morphology_prototype/morphology_prototype_manifest.json \
  outputs/multifield_style/procedural_five_family_reference_v1

python -m forge.multifield_style replay-procedural-reference \
  outputs/multifield_style/procedural_five_family_reference_v1/procedural_reference_manifest.json \
  --report outputs/multifield_style/procedural_five_family_reference_v1.replay.json
```

All write paths enforce the 100 GiB free-space floor before output.
