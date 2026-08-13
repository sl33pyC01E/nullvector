# NULLVECTOR map-art forge

`forge.map_art` turns a validated semantic map pack into deterministic,
pixel-native production assets. It never changes the source map topology.

Each art pack contains:

- 8 px terrain autotiles for every terrain ID and all 16 N/E/S/W masks;
- color plus additive-emission atlases for terrain, hazards, and objects;
- a full-map RGB base layer and static emissive layer;
- eight RGBA hazard frames and eight animated-emission frames packed into a
  4x2 PNG grid, with an explicit JSON frame-layout sidecar;
- deterministic decals and props with collision footprints, occlusion class,
  z-class, atlas coordinates, and world anchors;
- cell-resolution autotile, elevation-edge, variant, collision, occlusion,
  prop, and decal arrays in `art_semantics.npz`;
- a strict manifest binding every artifact to the source map's canonical
  semantic SHA-256 and to the renderer source SHA-256.

No filtered scaling or GIF encoder is used. All animation is represented as
single PNG sprite sheets plus metadata.

```powershell
python -m forge.map_art showcase --output outputs/map_art
python -m forge.map_art render outputs/maps --output outputs/map_art/packs
python -m forge.map_art validate outputs/map_art/packs
python -m forge.map_art fuzz --count 120 --report outputs/map_art/fuzz_report.json
```

Existing valid art packs are append-safe with `--skip-existing`. A pack made
from different renderer source is left untouched and reported instead of being
silently overwritten.

