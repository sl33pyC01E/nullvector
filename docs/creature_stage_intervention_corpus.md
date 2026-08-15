# NULLVECTOR // Cellular intervention trajectory corpus

## Purpose

The motion corpus teaches intact action-conditioned pose. The intervention
corpus teaches causal consequences: which cells were damaged, which organ
capacity was lost, whether the organism died, how its remaining body moved,
whether healing restored viable tissue, and how leaked fluid spread across the
2.5D ground plane.

The paired matrix covers all 20 native chassis with nine conditions:

1. intact control;
2. partial central wound;
3. the same wound followed by healing;
4. a lower appendage-plane cut;
5. neural ablation;
6. circulatory ablation;
7. respiratory ablation;
8. digestive ablation;
9. sensory ablation.

Every clip begins with the same 15 unperturbed frames, then receives its event
at frame 15. Healing occurs at frame 75. Clips run for 180 frames at 30 Hz.
The complete corpus therefore contains 180 clips, 32,400 frames, and 7,304,580
cell-frame samples.

## Native publication

```powershell
C:\Users\forre\Desktop\Godot_v4.3-stable_win64.exe `
  --headless --path C:\Users\forre\Documents\neural-game\game -- `
  --creature-stage-intervention-corpus=C:\Users\forre\Documents\neural-game\outputs\creature_stage_intervention_corpus_v1_final_a
```

The exporter runs before the playable scene is constructed. It is CPU-only,
refuses an existing destination or staging directory, publishes by atomic
rename, reserves 128 MiB for publication, and aborts if that reserve would
cross the 100 GiB free-disk floor.

The published directory has exactly two members:

```text
manifest.json
intervention_frames.u16le
```

No browser, viewport, Python process, neural checkpoint, or GPU is involved in
generation.

## Frame layout

The binary is little-endian unsigned-16 data in this order:

```text
clip
  frame
    10-word physiology summary
    anatomical cells [x delta, y delta, health, alive]
    160 fluid slots [x, y, vx, vy, radius, remaining life]
```

The summary order is:

```text
integrity, neural, circulation, respiration, digestion,
senses, energy, hydration, dead, fluid_count
```

Normalized values use `[0, 65535]`. Cell position deltas and fluid positions or
velocities use a bias of 32768 and scale of 256 units per pixel. Fluid radius
and life use scale 1024. Alive/dead flags are exactly zero or 65535. Unused
fluid slots are all-zero; active slot count is stored exactly, so padding
cannot masquerade as matter.

Each chassis manifest entry binds the family, structural morphotype, genome,
seed, ordered grid coordinates, tissues, organs, appendage ownership, sides,
and an ordered cell-identity SHA-256. Every clip binds its intervention,
target, event schedule, hit count, maximum fluid count, exact byte range, and
trajectory SHA-256.

## Independent semantic replay

```powershell
python -m forge.creature_stage_intervention_corpus `
  outputs/creature_stage_intervention_corpus_v1_final_a
```

Validation is deliberately stronger than checking hashes. It rejects:

- duplicate-key, non-finite, oversized, malformed, or schema-invalid JSON;
- symlinks, missing/extra members, stale producer source, bad paths, byte gaps,
  overlap, truncation, trailing data, or altered hashes;
- wrong family/chassis/intervention order or duplicated anatomical grids;
- non-binary life state, health remaining on a dead cell, or noncanonical
  unused fluid slots;
- a reported integrity or organ capacity that disagrees with the exact alive
  cells belonging to that organ group;
- a control that loses cells, organs, or fluids;
- a non-control intervention that hits nothing or produces no spreading fluid;
- wounds that kill cells, healing that does not improve mean health, or cuts
  that remove no cells;
- an organ ablation that leaves its target capacity nonzero;
- neural ablation that does not enter a terminal state;
- any pre-intervention divergence from the matching control;
- non-finite or excessive local displacement and corpus-identity drift.

`load_intervention_clip(path, clip_id)` returns immutable float32/boolean NumPy
arrays for summaries, local cell positions, health, alive state, fluid state,
and exact fluid counts. A training process cannot mutate the decoded authority
in place.

## Reproducibility evidence

Two final native exports were produced independently and compared byte for
byte:

| field | value |
| --- | --- |
| format | `nullvector-creature-stage-intervention-corpus-v1` |
| manifest SHA-256 | `b6c8dc3be4c84a28cf7e1701e39cb8e3b0946fe8e8edcf7d5dfaa16c74a1cfef` |
| binary SHA-256 | `cbe2628827ceb68c80987e9d187243c1efc59f3d34275c6e1fa56013a6797b65` |
| producer source SHA-256 | `3fa5118e57182fe11eb62a5d1c6eee4dc04dde9f23ea2e631f4637f90ec56941` |
| corpus identity SHA-256 | `59e15ed22ff3b492b96afdf03d04589852aeefb639519c17c04c51f7b405e01e` |
| binary bytes | 121,292,640 |
| maximum local displacement | 7.715762 pixels |
| minimum surviving integrity | 0.569131 |
| maximum simultaneous fluid particles | 134 of 160 slots |
| quantization clipping | zero values |

The authoritative roots are
`outputs/creature_stage_intervention_corpus_v1_final_a` and
`outputs/creature_stage_intervention_corpus_v1_final_b`. Their manifests and
binary payloads are separately byte-identical.

The first bounded attempt used 64 fluid slots and correctly refused to publish
when an anomaly respiratory ablation created 134 droplets. That payload is
preserved as diagnostic evidence. A later full output exposed that plantlike
and anomaly central tissue may carry non-vital structural organ labels; the
wound target was corrected to exclude the five vital capacity groups rather
than requiring the literal label `none`. That pre-fix result is likewise
preserved but is not an authority.

## Honest scope

Version 1 records cell health/alive state, local physical response, normalized
organ function, death, energy/hydration, and finite ground-plane fluid
particles. It does not yet encode:

- explicit bond connectivity, fracture propagation, or reattachment;
- detached-piece and polyp identity;
- scar tissue, infection, necrosis, or cellular regrowth;
- resource ingestion, digestion products, reproduction, or lineage updates;
- collision contacts, projectiles, world displacement, or rendered layers.

Those are the next aligned causal fields. They should be added as explicit
teacher state before training the physiology NCA and before reverse-distilling
the ensemble into the recurrent action-DiT/VAE. Clean reconstruction of this
corpus is necessary evidence for a dynamics model, not sufficient evidence of
a complete organism simulation.
