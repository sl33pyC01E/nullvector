# NULLVECTOR offline map forge

This package generates semantic top-down maps without network access or external assets. It is deterministic across processes: the public seed is mixed into independent PCG64 streams for layout, hazard decoration, objective selection, spawn selection, and elevation. Changing a decorative pass therefore cannot silently move the mission topology.

## Themes

- `arena`: elliptical combat bowl, cardinal gates, and seeded pillars.
- `rooms`: room graph, minimum-spanning corridors, and optional loops.
- `caves`: cellular cavern fields with repaired mission routes.
- `archipelago`: water-separated islands joined by explicit bridges.
- `garden`: orthogonal hedge cells, growth, pools, and gates.
- `anomaly`: polar fractures, chasms, crystals, and interference arcs.

Every pack contains `semantics.npz`, `manifest.json`, and a nearest-neighbor neon `preview.png`. The compressed semantic arrays are:

- `terrain: uint8[H,W]`
- `walkability: uint8[H,W]`
- `hazard: uint8[H,W]`
- `elevation: int8[H,W]`
- `zone: int16[H,W]`
- `nav_cost: float32[H,W]`
- `protected_backbone: uint8[H,W]`
- `required_clearance: uint8[H,W]`
- `decoration_forbidden: uint8[H,W]`

Coordinates are `[x, y]`; arrays are indexed `[y, x]`. The manifest records all ID dictionaries, dtypes, shapes, generator version/configuration, topology repairs, point sockets, canonical semantic SHA-256, individual artifact hashes, and every invariant result.

Map format `2.0.0` and generator `2.0.0` persist the three topology masks as
authoritative semantic arrays. Each manifest records their meanings, individual
canonical SHA-256 values, aggregate SHA-256, cell counts, capture policy, and the
`sha256-canonical-named-arrays-v1` hash-algorithm identity (name, dtype, shape,
and C-order bytes are all bound).
Version-1 packs are rejected explicitly: the reader never guesses or fabricates
missing route provenance.

The CLI default is the schema-major-specific `outputs/maps_v2` root. Existing
v1 directories remain untouched and are never silently migrated or overwritten;
pass an explicit different output root if both generations must coexist.

## Authoritative topology and invariants

The theme pass establishes terrain first. Hazard decoration is applied next. Required mission points are then connected with deterministic A* repairs, and a protected five-cell-wide backbone is carved from the start to the exit and every objective. Later passes may clear hazards from protected areas but never add blocked terrain or hazards to them.

`protected_backbone` is marked in the same indexed write that carves each
Manhattan-radius-two route footprint. `required_clearance` is marked in the
same loops that clear the start, exit, objective, and spawn safe regions.
`decoration_forbidden` is then the exact union of those masks and the final
hazard mask. Pack validation regenerates the map from its seed/theme/config and
requires byte-identical arrays and point sockets; no post-hoc path reconstruction
is part of loading or validation.

Validation fails the whole map if any of these contracts fail:

1. Every semantic array has the exact declared shape and dtype.
2. Terrain/hazard IDs are legal, and walkability exactly matches terrain.
3. The outer boundary is sealed.
4. Start, exit, objectives, and spawns are unique, in bounds, and walkable.
5. Flood fill from start reaches the exit and every objective.
6. The mission backbone remains connected after one-cell square erosion, representing an agent radius of one tile.
7. Start/exit graph distance meets the configured minimum.
8. Spawns meet graph-distance clearances from start/objectives and Manhattan clearance from hazards.
9. Hazards occur only on walkable cells.
10. Elevation, zones, and navigation costs cover their exact legal domains.
11. Topology masks are binary, have exact subset/union relationships, and carry every required point.
12. JSON Schema, NPZ/PNG hashes, canonical/member hashes, and PNG dimensions/mode all agree.
13. Deterministic replay reproduces every semantic array and point byte-for-byte.

## CLI

From the project root:

```powershell
python -m forge.maps generate --themes all --count 4 --output outputs/maps_v2
python -m forge.maps validate outputs/maps_v2
python -m forge.maps fuzz --count 500 --report outputs/maps_v2/fuzz_report.json
```

Generation checks `forge.safety.require_disk_floor` before every batch and every atomic file write. A complete pack is staged in a unique directory and published by one rename. Existing valid packs are validated and reused by default; they are never overwritten or deleted. The fixed disk floor is 100 GiB free after planned writes.

Byte uniqueness alone is not accepted as evidence of map variety. The
additive structural audit in `forge.map_diversity` measures graph branching,
cycles, corridor widths, coarse occupancy, hazards, elevation, zones, and
theme separability across a fixed 48-map seed matrix. See
`docs/map_structural_diversity.md`.
