# Structural map diversity audit

`forge.map_diversity` checks that deterministic maps differ in navigable
structure rather than receiving a uniqueness pass from decorative pixels. The
default matrix generates eight fixed-size maps for every one of the six themes
(48 maps total) using an independently mixed seed schedule.

Each map is hard-validated before scoring. Its 83-dimensional structural
fingerprint contains normalized walkability, hazard, elevation and zone
statistics; protected-route and clearance density; cardinal degree,
endpoint, junction and cycle density; nine successive square-erosion corridor
width measurements; and an 8x8 occupancy field. A separately quantized coarse
signature prevents tiny floating or decorative differences from masquerading
as new topology.

The audit also performs deterministic leave-one-out nearest-theme-centroid
classification. This is not a gameplay classifier: it is evidence that arena,
rooms, caves, archipelago, garden and anomaly retain recognizable structural
languages across seeds. Gates require every semantic map to be unique, at
least 90% coarse-topology uniqueness, at least 85% theme accuracy, at least
50% recall for every theme, a positive mean true-theme margin, and nonzero
within-theme dispersion.

Build and exactly replay the additive report:

```powershell
python -m forge.map_diversity build
python -m forge.map_diversity validate `
  outputs/map_diversity_v1/map_diversity_report.json
```

The report binds the complete feature vocabulary, quantization contract,
configuration, seed schedule, all semantic and coarse hashes, classifications,
gates, and audit source. Validation regenerates all 48 maps and requires the
canonical report to match byte-for-byte.
