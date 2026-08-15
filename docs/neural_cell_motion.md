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

The current milestone includes the corpus, model, segmented trainer, and a
strict recurrent held-out evaluator. It deliberately does not claim trained
motion quality or runtime authority. Before integration it still needs a clear
CUDA window for the declared training schedule, a checkpoint that passes the
calibrated gates below, ONNX export, and native Godot/WebGPU parity. Until
those gates pass, the deterministic cellular motion bank remains the runtime
authority.

Corpus loading is fail-closed: manifests must be bounded canonical JSON with no
duplicate keys; every source and shard path must remain within its authority
root; NPZ members have an exact registry and bounded NPY headers; categorical
features remain one-hot on occupied cells; and features/targets must be exactly
zero outside the cellular chassis. Rehashing a malformed artifact does not make
it acceptable.

Build the full 45-identity corpus. The default authority is additive
`corpus_v2`; historical `corpus_v1` remains untouched. Corpus provenance hashes
only the tensor-construction contract and implementation, so later package
exports, trainers, or evaluators cannot invalidate unchanged verified shards:

```powershell
python -m forge.neural_cell_motion build-corpus
python -m forge.neural_cell_motion validate-corpus --replay
python -m forge.neural_cell_motion model-info
```

The published v2 corpus contains 45 identities and 42,480 frames. It was
rebuilt from the v1 bytes only after every shard regenerated exactly, then
exact-replayed again in isolated validators. Build recovery contained two
native worker failures; final replay contained one retry. All three events are
preserved in telemetry rather than erased by the successful retries.

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

## Segmented production training

Production training is deliberately restartable and fail-closed. The default
schedule is 12,000 family-balanced updates in immutable 500-step checkpoints.
Every batch contains the same number of humanoid, animalian, plantlike,
anomaly, and machine examples even though the source family sizes differ. A
sample coordinate is a pure function of the frozen seed, global step, and
batch slot, so a resumed segment reads the same identities and frames.

The supervisor launches each segment in a fresh Python process, permits at
most three attempts, records native access violations and timeouts, and accepts
a checkpoint only after reloading and hashing its model and EMA states. Its
telemetry is canonical, source-bound, sequential, and rejects duplicated,
skipped, or incoherent attempts. The trainer also refuses to start unless CUDA
BF16 deterministic mode is available and at least 14 GiB of VRAM is free; it
will not crowd an unrelated GPU workload.

Inspect the deterministic sampler without using CUDA:

```powershell
python -m forge.neural_cell_motion sampler-report --steps 100
```

Prepare or run the immutable production schedule only in a clear GPU window:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m forge.neural_cell_motion prepare-production
python -m forge.neural_cell_motion train-production
```

## Held-out recurrent evaluation

Every immutable training segment is evaluated in a fresh process. Evaluation
feeds each prediction back as the next frame's recurrent state rather than
teacher-forcing the authoritative preceding frame. It covers five held-out
families, all 13 motions, all eight travel directions, 520 clips, and 4,720
frames. Metrics are macro-averaged across families so a large chassis cannot
dominate the quality result merely by containing more cells.

The report binds the exact checkpoint bytes, model and EMA state hashes,
training contract, corpus semantic identity, and a separate evaluation-source
hash. Internal validation derives headline metrics independently from both
family and frame-weighted motion breakdowns, rejecting fully rehashed nested
metric tampering. It also checks response energy for displacement, activation,
and emission, exact zero outside the chassis, bounded recurrent output,
previous-frame baseline improvement, loop closure, action endpoints, and every
family/motion metric.

Non-final checkpoints evaluate validation only. The final checkpoint also
evaluates the sealed test split and is promotion-eligible only when every gate
passes on both splits:

```powershell
python -m forge.neural_cell_motion evaluate-production `
  --output outputs/neural_cell_motion/production_v1 --step 12000
python -m forge.neural_cell_motion validate-evaluation `
  --output outputs/neural_cell_motion/production_v1 --step 12000
```

The production runner is still not runtime authority. A successful process and
a low scalar loss are insufficient; only a final checkpoint with
`promotion_eligible: true`, followed by ONNX export and runtime parity, may
replace the authored cellular motion bank.

## ONNX runtime bundle

An immutable checkpoint can be exported as a bounded, self-verifying ONNX
bundle without CUDA. Export always loads the checkpoint's EMA weights and
records the training contract, corpus identity, checkpoint bytes, model/EMA
state hashes, model geometry, tensor interface, and ONNX artifact hash.

The graph keeps batch size dynamic for every input and the output. Export runs
three deterministic CPU probes at batch sizes 1, 2, and 5 through both PyTorch
and ONNX Runtime. It rejects non-finite output, numerical disagreement above
`2e-5`, or any motion outside the authoritative cellular support. Replay reloads
the original checkpoint, reruns all probes, and exact-compares the recorded
evidence. A rehashed manifest cannot authorize changed model bytes.

```powershell
python -m forge.neural_cell_motion export-onnx `
  --output outputs/neural_cell_motion/production_v1 `
  --step 12000 `
  --destination outputs/neural_cell_motion/runtime_v1

python -m forge.neural_cell_motion validate-onnx `
  --bundle outputs/neural_cell_motion/runtime_v1 `
  --output outputs/neural_cell_motion/production_v1 `
  --replay
```

This bundle proves portable graph integrity and CPU numerical parity. It does
not by itself promote a checkpoint, prove visual quality, or replace the
recurrent held-out gates. Native Godot/WebGPU integration must consume this
same tensor contract and add its own backend parity evidence before runtime
authority changes.
