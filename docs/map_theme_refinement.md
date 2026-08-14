# Deterministic map theme refinement

The topology-v2 map generator already produces hard-valid and highly diverse
maps, but its structural audit exposed one specific weakness: archipelago
recall was 62.5% while the other five themes scored 100%. High-land seeds could
read as gardens or anomaly fields after the required mission routes were
repaired.

`forge.map_theme_refinement` adds a bounded deterministic generation policy
without changing or reconstructing topology-v2 arrays. For an archipelago
request it evaluates at most 32 independently mixed candidate seeds and accepts
the first hard-valid map with:

- walkable land between 42% and 62%;
- at least 34% water;
- at least 5% explicit sand shoreline.

The selected map is still produced entirely by the authoritative map generator,
including exact generation-time backbone, clearance, forbidden-decoration,
hazard, objective, spawn, zone, elevation, and navigation arrays. Both the
requested seed and selected seed are retained; the selection never loops
without a bound or hides its attempt count.

The audit uses the same 48-map schedule and 83 structural features as the
original diversity authority. The refined matrix requires 48/48 semantic
uniqueness, 100% leave-one-out six-theme accuracy, and 100% recall for every
theme. A separate 128-request fuzz test proves deterministic bounded selection
and hard map validity.

```powershell
python -m forge.map_theme_refinement build
python -m forge.map_theme_refinement validate `
  outputs/map_theme_refinement_v1/map_theme_refinement.json
```

The contact sheet renders all eight refined archipelagos at pixel resolution,
including mission endpoints and objectives. Validation regenerates every map
and requires exact manifest/image byte closure.
