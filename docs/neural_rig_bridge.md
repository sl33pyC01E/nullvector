# Neural multi-field to rig binding bridge

`forge.neural_rig_bridge` turns one accepted 48x48 neural multi-field sample
into a deterministic logical rig. It is a CPU-only bridge between immutable
rest-identity generation and graph motion. It never invokes a checkpoint and
never uses a procedural sprite as a replacement for neural pixels.

## Authority boundary

The authoritative visual source is always the aligned raw tuple at each pixel:

```text
(part_owner, material, emission_level)
```

The bridge copies the three input arrays and makes them read-only. Sixteen
disjoint owner masks reconstruct those arrays bit-for-bit. A separate
`driver_index` assigns each foreground tuple to one of the existing compiler's
eight affine drivers:

```text
body, head, left_arm, right_arm,
left_leg, right_leg, appendage, weapon
```

That separation matters. Role conditioning can replace every `left_leg` owner
token with detail, emission, ornament, or core tokens while the pixels still
occupy the left-leg branch of the conditioning skeleton. Treating owner IDs as
bones would erase valid anatomy. The bridge instead uses named derived points
and skeleton segments to bind those existing tuples to motion drivers. It does
not change their owner, material, emission, position, or count in the rest
frame.

The manifest asserts both:

```json
{
  "pixel_authority": "raw_neural_aligned_fields",
  "procedural_pixel_substitution": false
}
```

## Inputs

`bind_neural_fields()` consumes:

- uint8 `part_owner`, `material`, and `emission_level` arrays;
- the float32 eight-channel conditioning guide;
- morphology family, family-local subtype, and combat role;
- a train-observed legal tuple table, or the compiler's versioned semantic
  tuple vocabulary when a table is not supplied;
- optional 24-value genes and upstream SHA-256 provenance;
- named derived anatomy containing only joints and sockets.

Named anatomy can come from two safe paths:

1. `derive_conditioned_anatomy()` replays the versioned corpus condition and
   returns only its named points. Its temporary procedural layers and tokens
   are discarded and cannot enter the binding API.
2. If every required owner region remains visible, the binder can derive
   points deterministically from owner geometry and the retained joint/socket
   guide channels. Samples whose role has overwritten a required region must
   provide conditioned anatomy rather than accepting a guess.

`bind_raw_sample_archive()` reads the evaluator's authoritative outer raw
sample v1 manifest with its embedded generation-validation v2 report. The
manifest is mandatory: the bridge accepts only samples whose evaluator report
is both `hard_valid` and `accepted`, has no errors, and has every hard gate set
to true. The bridge rebuilds conditioned point anatomy from `corpus_seed` and
retains checkpoint/corpus/source/legal-table hashes as upstream provenance.

The raw boundary is deliberately narrow. Before NumPy loads a byte, the bridge
requires a ZIP/NPZ archive no larger than 4 MiB compressed or uncompressed,
exactly fifteen named NPY members, no duplicate/encrypted/directory members,
exact shapes, native canonical dtypes, C array order, and a bounded format
scalar. The manifest is strict UTF-8 JSON no larger than 2 MiB, rejects duplicate
keys/non-finite numbers, must satisfy the current raw-sample schema, and must use
an exact condition record. Its POSIX artifact path must remain inside the run
directory and resolve to the archive passed to the API; byte count and SHA-256
must match. The legal tuple table is compared to the evaluator's raw-byte
fingerprint rather than merely recorded.

Example:

```python
from pathlib import Path
from forge.neural_rig_bridge import (
    bind_raw_sample_archive,
    motion_adapter_contract,
    render_bound_pose,
)

binding = bind_raw_sample_archive(
    Path("bank/raw/m00_s00_r00_v00.npz"),
    raw_manifest_path=Path("bank/raw/m00_s00_r00_v00.json"),
    legal_tuples=checkpoint_legal_tuples,
)

rest = render_bound_pose(binding)  # bit-exact aligned fields
adapter = motion_adapter_contract(binding, facing="southeast")
```

For evaluator artifacts, `legal_tuples` must be the exact canonical table from
the generating checkpoint/corpus. Omitting it deliberately falls back to the
compiler vocabulary, which will be rejected when its fingerprint differs from
the evaluator manifest.

## Fail-closed validation

Binding rejects a sample before any rig metadata is published when any of the
following is true:

- field shape, dtype, vocabulary, or guide range is invalid;
- a background pixel carries material or emission;
- a tuple is outside the supplied legal tuple table;
- any foreground enters the three-pixel safety margin;
- required body, head, or core ownership is absent;
- a non-anomaly physical rig is not one eight-connected component;
- an anomaly has more than three components or a dominant fraction below 0.85;
- a named point is outside neural foreground or unsupported by the scaffold;
- any motion driver cannot acquire its minimum existing-neural-pixel support;
- a pivot has no nearby pixel supporting its named driver;
- the logical driver graph, owner partition, provenance hash, schema, or exact
  rest reconstruction is inconsistent.

External `DerivedAnatomy` values must declare one of the two versioned bridge
sources and carry its reproducible SHA-256. Conditioned points are replayed from
the exact unsigned 32-bit corpus seed and condition, while guide/owner geometry
points are regenerated from the supplied neural fields. Relabelled points with
a copied source hash are therefore rejected. Legal tuple tables are uniquely
sorted, bounded by the finite semantic vocabulary, and matched to any upstream
evaluator fingerprint.

Aura pixels are non-physical for topology purposes. An anomaly may retain up to
two small disconnected visual components; each receives a logical orbital edge
to the body root. The rig graph is therefore connected even when the rendered
anomaly deliberately is not. Other families use a single-component policy.

Some valid grammars co-locate root, focus, and appendage pivots. A categorical
pixel cannot belong to all three drivers simultaneously, so every anchor stores
both its exact derived `point` and a bounded nearby `support_point` that really
belongs to the named driver. The pivot is used for transforms; the support point
proves ownership without inventing an overlapping pixel.

## Tuple-preserving motion adapter

`render_bound_pose()` accepts one source-to-destination affine 3x3 matrix per
driver. It inverse-samples with nearest-neighbor lookup, resolves overlaps with
a complete deterministic z-order, and copies the source tuple as one unit.
It cannot independently relabel a head, material, or glow channel. Every posed
tuple must be present in the rest source, and the identity transform must
reconstruct all three source arrays exactly.

`validate_bound_frame()` does not trust a frame's recorded hashes. It
recomputes categorical domains, margins, tuple membership, driver support,
aligned-field and driver hashes, parses the stored affine matrices and z-order,
then rerenders the binding and requires every array plus the canonical manifest
to match exactly. Neural motion clip validation applies this authority check to
every frame before accepting clip-level hashes.

Every public adapter entry point first runs complete binding validation. That
validation recomputes the driver partition, anchor support, owner layers, graph,
topology, hashes, schema, and entire canonical manifest rather than trusting
self-consistent supplied metadata. Affines must be numeric, finite, canonical
2D matrices with bounded translation, scale, anisotropy, and condition number;
singular, projective, Boolean, and numerically pathological matrices fail closed.

`facing_transforms()` and `render_facing_frame()` provide eight-way root-facing
with one fitted global transform. `motion_adapter_contract()` publishes the
driver order, pivots, hierarchy, joints, sockets, facing, matrix convention,
sampling rule, and binding hash needed by a motion producer.

The shared motion compiler owns family motion programs and action curves. Its
public `motion_pose()` and `motion_driver_matrices()` functions accept named
graph points rather than procedural pixels. `compile_neural_motion_clip()` uses
that program to animate a binding through all thirteen motions and eight
facings. It applies one clip-wide safety fit, transforms tuple layers as atomic
units, derives joints and sockets only from pixels owned by their driver, and
records exact frame and clip hashes. `replay_neural_motion_clip()` recompiles
the clip and compares every categorical field, driver map, anchor, frame hash,
and canonical manifest. No change to neural rest pixels or the binding format
is required.

```python
from forge.neural_rig_bridge import (
    assert_exact_neural_motion_replay,
    compile_neural_motion_clip,
    replay_neural_motion_clip,
)

clip = compile_neural_motion_clip(binding, "idle_wiggle", facing="southwest")
report = replay_neural_motion_clip(clip)
assert_exact_neural_motion_replay(report)
```

## Determinism and replay

Every binding records hashes for:

- authoritative aligned fields;
- conditioning guide and optional genes;
- legal tuple table;
- point-only anatomy;
- driver partition;
- upstream raw archive/manifest/checkpoint/corpus/source identities;
- the complete bridge source set;
- the canonical binding manifest.

`replay_binding()` repeats binding from source arrays and requires exact binding
hash, manifest, owner masks, driver partition, anchors, and rest reconstruction.
It reports `exact` only when every comparison succeeds. A single field mutation
either fails validation or produces a mismatch; there is no repair mode.
`assert_exact_replay()` also verifies the exact check vocabulary, every Boolean,
identical expected/actual binding hashes, empty error list, and the canonical
report SHA-256. A caller cannot forge success with only `{format, status}`.

The strict manifest schema is
`shared/schema/neural_rig_binding.schema.json`.

## Verification

```powershell
python -m pytest tests/test_neural_rig_bridge.py
```

The dedicated suite covers all five families, all eight combat roles, and all
eight facings; bit-deterministic rebinding; exact rest reconstruction; transformed
tuple preservation; anomaly orbital islands and dominance/count boundaries;
topology, margin, tuple, and owner rejection; scaffold-only point derivation;
raw v1 plus embedded-validation-v2 provenance; traversal, member substitution,
duplicate ZIP entries, endianness/shape/scalar admission; manifest reconstruction;
bounded affines; forged replay reports; schema validation; and exact replay.
Fixtures use current procedural fields only as synthetic inputs. Tests do not
read production checkpoints or use a GPU.
