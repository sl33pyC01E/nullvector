# Learned cellular motion v1

`forge.neural_cell_motion` is the learned temporal successor to the authored
14-driver motion bank. It does not replace the existing bank yet. It builds a
strict training corpus from that proven authority so a neural network can learn
the same idles, locomotion, emotes, and actions on the actual breakable cellular
anatomies.

The production corpus covers all 45 authoritative symmetry-refined organisms
across humanoid, animalian, plantlike, anomaly, and machine families. The
breeding grammar's real primary-family census is intentionally 11/10/9/8/7,
not a fabricated nine-per-family grid. Each family holds out its final two
identities for validation and test, and training must use family-balanced
sampling. Each identity contributes all 13
motions, eight top-down travel directions, and 944 recurrent frames: 42,480
examples in total. The body stays vertically aligned. Travel direction rotates
the local displacement vectors, never the underlying chassis.

Each anatomy is represented by a 60-channel 48x48 field:

- occupancy and body-relative XY;
- mass, stiffness, health capacity, fluid, energy, and nutrient state;
- emission level;
- exact tissue, material, and part-owner vocabularies;
- eleven motor/organ channels including paired appendages and locomotors.

The target has four channels: XY displacement, motor activation, and emission
activation. Every example also records its exact predecessor, so the model is
trained as a recurrent motion field rather than a disconnected frame lookup.

The production network is a phase-conditioned, recurrent three-level U-Net.
It uses family, motion, travel direction, and continuous phase embeddings,
12x12 spatial attention, and feedback from the previous predicted motion
state. The default configuration is intentionally substantial (tens of
millions of parameters); focused tests use a smaller geometry only to prove
shape, gradient, and contract behavior on CPU.

The current milestone is corpus/model foundation. It deliberately does not
claim trained motion quality or runtime authority. Before integration it still
needs segmented CUDA training, family-balanced held-out evaluation, loop and
event-pose gates, recurrent drift tests, ONNX export, and native Godot/WebGPU
parity. Until those gates pass, the deterministic cellular motion bank remains
the runtime authority.

Corpus loading is fail-closed: manifests must be bounded canonical JSON with no
duplicate keys; every source and shard path must remain within its authority
root; NPZ members have an exact registry and bounded NPY headers; categorical
features remain one-hot on occupied cells; and features/targets must be exactly
zero outside the cellular chassis. Rehashing a malformed artifact does not make
it acceptable.

Build the full 45-identity corpus:

```powershell
python -m forge.neural_cell_motion build-corpus
python -m forge.neural_cell_motion validate-corpus --replay
python -m forge.neural_cell_motion model-info
```

`build-corpus` is process-isolated by default: at most two CPU workers, three
bounded attempts per identity, a ten-minute worker deadline, native-crash
classification, atomic publication, and an independently spawned final
validator. An interrupted monolithic staging tree can be recovered without
trusting its bytes; every candidate shard is regenerated and exact-compared in
a fresh worker before it is reused:

```powershell
python -m forge.neural_cell_motion build-corpus `
  --recover-from outputs/neural_cell_motion/.corpus_v1.tmp-13360
```

The build telemetry beside the published corpus records every attempt, reuse,
timeout, and Windows access violation. A failed worker never publishes a shard.
`validate-corpus` uses the same containment policy, attesting each identity in
a fresh bounded process and requiring exact aggregate coverage. `--replay`
regenerates each identity's 944 recurrent targets inside those isolated
validators rather than risking one long native process.

Build a five-family smoke corpus without changing the production destination:

```powershell
python -m forge.neural_cell_motion build-corpus `
  --output outputs/neural_cell_motion/smoke_v1 `
  --identities-per-family 1
```
