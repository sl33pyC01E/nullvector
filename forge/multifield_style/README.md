# `forge.multifield_style`

CPU-only deterministic presentation compiler for accepted 48×48 categorical
sprite fields. It produces pixel-native RGBA/effect layers without importing
the torch training stack and without changing part, material, emission, rig,
or collision authority.

```text
python -m forge.multifield_style compile GENERATION_MANIFEST DESTINATION
python -m forge.multifield_style replay DESTINATION/style_manifest.json
python -m forge.multifield_style procedural-reference MORPHOLOGY_MANIFEST DESTINATION
python -m forge.multifield_style replay-procedural-reference DESTINATION/procedural_reference_manifest.json
python -m forge.multifield_style source-hash
```

The neural compiler accepts only an immutable, schema-valid, `ready`
`nullvector-multifield-generation-bank-v1`. Every accepted sample's compiled
NPZ container, exact key set, bounded ZIP members, dimensions, dtypes,
categorical domains, embedded hashes, manifest artifact hash, and aligned-field
hash are verified before rendering.

The separate procedural-reference command proves all five family vocabularies
using the authoritative morphology prototype fields. Its schema and manifest
state `neural_output: false`; these assets must never be presented as model
generations.

Published contracts:

- `shared/schema/multifield_style_bank.schema.json`
- `shared/schema/multifield_style_procedural_reference.schema.json`

See `docs/multifield_style_compiler.md` for the full layer, palette, gate, and
replay contracts.
