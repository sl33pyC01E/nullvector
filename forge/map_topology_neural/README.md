# Neural Map Topology Foundation

This package is an isolated, CPU-first foundation for neural map structure. It
does not replace `forge.maps`, publish `map_pack_v2`, run CUDA, or integrate the
game. Its authority boundary is deliberately narrow:

- the bounded corpus reader streams only `terrain`, `hazard`, `elevation`, seed,
  theme, and point members from the frozen decorator corpus;
- the tensor contract samples only the three categorical fields while points and
  global conditions remain immutable inputs;
- the VQ codec is a representation smoke, not a generative model;
- the deterministic compiler owns legality, reachability, square radius-one
  clearance, safe disks, masks, derived navigation fields, validation, and replay.

## Strict corpus read

```powershell
python -m forge.map_topology_neural read-corpus-sample `
  --corpus outputs/map_decorator_corpus_v1 `
  --shard main-t00-p00-o00 --index 0
```

The reader pins corpus SHA-256
`16ed5f3b1a661e2bfc2abe9e16c39e9b8caaecba81f50fed6658cc4f73cffab8`,
validates the root identity and recorded validation, sidecar and artifact hashes,
the complete ZIP member census, stored compression, safe member paths, NPY
headers, dtype/shape/byte descriptors, split identity, and direct point arrays.
It hashes the full shard but never inflates the 53-channel decorator tensor.

## CPU smoke and replay

```powershell
python -m forge.map_topology_neural build-smoke `
  --corpus outputs/map_decorator_corpus_v1 `
  --output outputs/map_topology_neural_smoke_v1

python -m forge.map_topology_neural replay-smoke `
  --output outputs/map_topology_neural_smoke_v1
```

The smoke is bounded to six 32x32 cases and two CPU optimization steps with a
tiny configurable codec. It produces immutable raw proposals, compiled
artifacts, ordered edit ledgers, a deterministic contact sheet, a bounded
`weights_only=True` checkpoint with model EMA and RNG state, and exact replay.
All large-write paths enforce the 100 GiB free-space floor.

## Compiler phases

The compiler validates immutable points, seals boundaries, normalizes legal
categorical tuples, selects the primary raw component for diagnostics, connects
mission/spawn sockets with stable minimum-edit routing, widens every route to a
square radius-one footprint, captures backbone and clearance masks at the write
sites, clears safe disks and hazards, caps theme-local hazard density, derives
walkability/zones/nav cost/forbidden decoration, and calls authoritative
`forge.maps.validate.assert_valid`.

Every mutation to a sampled field or captured mask is a sequence-numbered ledger
entry. Derived fields are identified as derivations rather than misrepresented as
neural samples. Loading a compiled artifact reruns the compiler and rejects a
tampered ledger even if an attacker recomputes the local JSON hashes.
