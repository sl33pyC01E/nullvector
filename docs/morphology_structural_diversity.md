# Neural morphology structural diversity

`forge.morphology_diversity` audits the immutable 80-specimen production neural
bank using only scale-normalized solid silhouettes. Palette, material, emission,
sample IDs, and family labels are excluded from the feature vector.

Each specimen is represented by ten geometric measurements and a deterministic
12 by 12 nearest-neighbor occupancy grid. The report records coarse chassis
uniqueness and leave-one-out nearest-centroid classification for family,
subtype, and role. Family classification and coarse chassis diversity are hard
gates. Subtype and role classification are diagnostics because four examples
per subtype are not enough to justify a production acceptance boundary.

Bilateral symmetry, paired arm/leg mass balance, and appendage fraction are also
reported per family. Symmetry is intentionally a soft biological prior rather
than a hard requirement, preserving anomalies, wounds, role asymmetry, and
useful generated variation.

Build and exact-replay the authority:

```powershell
python -m forge.morphology_diversity build
python -m forge.morphology_diversity validate outputs/morphology_diversity_v1/morphology_diversity_report.json
```
