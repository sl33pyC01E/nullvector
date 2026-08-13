# Neural rig repair v2

`forge.neural_rig_repair` is a CPU-only, fail-closed logical binder for the 80
authoritative neural specimens. It exists to answer a narrow question: can the
raw neural rest fields be animated as rigs without editing the identity pixels?

The answer for the current source bank is yes: all 80 identities can be bound.
This package is intentionally not connected to the game, Neural Workshop,
sprite atlas compiler, or the active multifield neural-motion output.

## Identity invariant

The authoritative rest identity is the aligned tuple of three `uint8 [48,48]`
arrays:

- `part_owner`
- `material`
- `emission`

Repair v2 never substitutes, relabels, translates, crops, inserts, or removes a
rest pixel. Every repaired binding reconstructs those three arrays exactly and
must reproduce the source aligned-fields SHA-256. The original raw manifest and
NPZ are verified by path, byte count, SHA-256, exact ZIP member registry, shape,
dtype, finite-value bounds, and source condition. Raw and compiled categorical
fields must already be identical.

Only logical metadata may change:

- anchor and support selection on existing physical pixels;
- driver assignment and graph support;
- logical links between detached physical components;
- aura treatment as a nonphysical effect layer;
- clip-local motion attenuation, uniform fitting, and layer order.

Posed frames may resample existing tuples under affine motion. They may not
introduce a tuple that was absent at rest. Rest identity and posed rasterization
are deliberately separate contracts.

## Frozen-v1 diagnosis

Repair planning reruns the read-only bridge whose source hash is
`46372e031c91d0202d0e55a8422385978c5157f76d83ed20adef9ed3e7250305`.
The census must be exactly 70 bindable and 10 rejected:

| Category | Count | Logical repair |
|---|---:|---|
| Anchor on background | 3 | Select the nearest existing physical driver support. Maximum observed displacement is one pixel. |
| Plant topology | 1 | Preserve both components and add one metadata-only fixed logical link. No bridge pixel is inserted. |
| Required owner absent | 3 | Use guide-conditioned anatomical anchors and existing physical support; do not invent the missing owner label. |
| Safety margin | 3 | Treat the offending aura pixels as nonphysical effect metadata and fit every posed clip inside the safe domain. |

The exact ten identities and bridge reasons are persisted in every plan and in
`rest_audit.json`. No class in this 80-sample bank is fundamentally
unrepairable. This is not a claim that an arbitrary future sample is repairable:
the loader and planner reject samples that cannot satisfy bounded support,
anchor, component, or motion constraints.

## Logical projection

The projection is deterministic and source-derived.

1. The conditioned anatomy supplies named joints and sockets but no pixels.
2. Each of the eight drivers receives at least 12 existing physical pixels.
3. Direct semantic owners are assigned to their natural drivers; flexible
   detail and ornament pixels use a deterministic nearest-segment Voronoi rule.
4. All 15 anchors use an existing non-aura pixel owned by their driver. Each
   support is grown, where physical coverage permits, into a deterministic
   four-pixel connected cluster within a two-pixel radius. This prevents a
   lone logical pixel from disappearing under diagonal inverse-nearest
   rasterization without changing any semantic tuple.
5. Aura tuples are attached to the nearest physical driver for motion but are
   never accepted as anchor support or a physical topology component.
6. Detached physical components remain byte-for-byte intact and receive a
   logical link to the dominant component.

Plans are canonical JSON, strict-schema validated, self-hashed, and bound to the
generation manifest, style manifest, raw manifest, raw archive, aligned field
hash, static palette hash, legal-tuple fingerprint, and frozen bridge hash.

## Motion envelope and stress

Each repaired binding is exercised through the exact 13-motion by 8-facing
matrix: 104 clips and 944 frames per identity, or 8,320 clips and 75,520 frames
for the bank.

The clip envelope is bounded and deterministic:

- candidate motion strengths are `1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25,
  0.125, 0.0`;
- attenuation is relative to the first pose, so it cannot change the requested
  facing baseline;
- one positive uniform fit is applied to the complete clip;
- logical z-order candidates are finite and source-derived, with the frozen
  default tried first;
- every frame must remain inside the three-pixel safety margin, retain all
  eight drivers, preserve source-only tuples, and keep every transformed anchor
  within three pixels of the closed unit-cell footprint of local driver
  support (the exact raster domain, rather than an overstated distance between
  integer pixel centres);
- looping clips must have byte-identical first and last frame records.

The stress run and the independent exact replay are each split into 16
process-isolated modulo shards, normally with a maximum of two workers. The
two canonical stress reports must be byte-identical before a bank is sealed.
CUDA is disabled with `CUDA_VISIBLE_DEVICES=-1`; BLAS
thread counts are pinned to one. Each shard has a 900-second timeout and up to
three attempts. Supervisor telemetry records signed and unsigned Windows exit
codes and explicitly classifies access violations (`0xC0000005`), stack buffer
overruns, stack overflows, hardware exceptions, signals, and timeouts. A valid
shard is reusable only when its generation, style, bridge, and repair-source
hashes are exact.

The compiler checks the 100 GiB free-space floor before planning and during
every shard. The repair output is metadata and audit JSON; it does not compile
atlases or use checkpoints.

## Output contract

The default additive destination is `outputs/neural_rig_repair_v2`:

```text
repair_bank_manifest.json
verification_report.json
rest_audit.json
plans/
  <sample_id>.repair.json        (80)
motion_stress/
  stress_report.json
  stress_telemetry.json
  shards/
    stress_shard_00.json ... stress_shard_15.json
motion_replay/
  stress_report.json
  stress_telemetry.json
  shards/
    stress_shard_00.json ... stress_shard_15.json
```

Before `repair_bank_manifest.json` exists, deterministic staging plans may be
refreshed. Once that manifest exists, the destination is sealed; recompilation
must use a new additive destination. No compiler path recursively deletes or
cleans existing data.

Schemas:

- `shared/schema/neural_rig_repair_plan.schema.json`
- `shared/schema/neural_rig_repair_bank.schema.json`
- `shared/schema/neural_rig_repair_replay.schema.json`

All JSON loaders reject duplicate keys, non-finite constants, symlinks, unsafe
or noncanonical paths, oversized files, unexpected properties, count drift,
self-hash drift, and artifact byte/SHA mismatch.

## Commands

Prepare and audit only the 80 rest bindings:

```powershell
python -m forge.neural_rig_repair prepare
```

Run the full process-sharded stress and independent exact replay:

```powershell
python -m forge.neural_rig_repair compile
```

Replay a sealed bank, independently rerendering all 75,520 frames:

```powershell
python -m forge.neural_rig_repair replay `
  outputs/neural_rig_repair_v2/repair_bank_manifest.json `
  --report outputs/neural_rig_repair_v2/verification_report.json
```

`--metadata-only` verifies the complete signed artifact graph without
rerendering. It deliberately reports `inspected`, sets `neural_output` false,
and cannot satisfy the final all-gates proof. A publishable verification report
requires exact-motion mode.

## Current limits

- Repair v2 is an isolated foundation, not a runtime integration.
- It does not train or infer a VAE/diffusion model and does not mutate a model
  checkpoint.
- It does not choose presentation palettes or generate sprite atlases.
- Layer-order repair is logical; a later presentation compiler should visually
  review occlusion even when categorical motion gates are exact.
- The future-bank rejection policy is intentional. A sample with too little
  physical support, an anchor requiring more than 12 pixels of rest displacement,
  an unbounded motion fit, or no bounded local layer ordering is rejected rather
  than silently changing neural identity.
