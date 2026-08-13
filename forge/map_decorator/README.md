# Topology-locked map decorator foundation

`forge.map_decorator` is the deterministic safety boundary in front of a future
categorical neural decorator. It has no file-writing path and does not import or
run a model. The current semantic map remains authoritative.

## Required inputs

Every call requires the three authoritative map-format-v2 `uint8[H,W]` masks:

- `protected_backbone`
- `required_clearance`
- `decoration_forbidden`

The map generator captures these arrays at the exact route/safe-region mutation
sites and persists them in `semantics.npz`; the package deliberately does not
reconstruct a backbone from map topology.
`decoration_forbidden` must contain the other masks and every hazard cell, and
`required_clearance` must contain start, exit, objective, and spawn coordinates.
Additional forbidden cells are allowed. Wrong dtype, shape, domain, or subset
relationships are rejected.

`encode_features(...)` returns a read-only `float32[C,H,W]` tensor with 53
versioned channels, its canonical SHA-256, the full channel manifest and hash,
each source-mask hash, and deterministic global conditions. The eight noise
channels use random-access SplitMix64 coordinate hashes at scales 1, 2, 4, and
8; they do not consume mutable RNG state. `validate_encoded_features(...)`
recomputes the complete tensor and rejects byte drift.

## Class contracts

The model-facing heads have fixed shapes across all six themes:

- variant: 8 real classes (`0` is a variant, not empty)
- decal: 3 slots (`0` is empty; theme-local classes occupy remaining slots)
- prop: 3 slots (`0` is empty; only non-colliding theme-local classes exist)
- emission: 4 levels (`0` is off/empty)

On protected, clearance, forbidden, hazard, or required-point cells, decal,
prop, and emission masks expose only class `0`. Variant remains legal in
`[0,7]`, matching the renderer contract because it changes only a terrain
micro-pattern and has no empty class.

Catalog class IDs are compact model-head IDs. Each class retains the original
one-based map-art `catalog_index` for later compilation. Missing theme-local
slots are always illegal. Colliding props never enter the neural catalog.

Emission capability never inspects rendered pixels. Terrain capability exactly
mirrors authoritative renderer rules: exposed north/east/west edges of
walkable terrain, crystal detail, and odd growth variants can emit. Catalog objects are capable only when
their authoritative `PropSpec.color_role` is explicitly `primary` or
`secondary`; unknown roles force emission level `0`. Before sampling, a mask
may expose potential object emission. Final field validation rebuilds emission
legality conditioned on the actually selected decal and prop.

At most one of decal or prop may be non-empty in a cell. Validation rejects an
illegal choice and never clamps, repairs, or silently replaces it.

## In-memory use

```python
from forge.map_decorator import build_foundation

result = build_foundation(
    map_data,
    protected_backbone=protected,
    required_clearance=clearance,
    decoration_forbidden=forbidden,
    public_seed=decorator_seed,
)
assert result.report["passed"]
```

`FoundationCase` and `fuzz_foundation(...)` provide deterministic replay fuzzing
for persisted exact masks without writing artifacts. Map-format-v1 packs are
ineligible and explicitly rejected because they lack this provenance.
