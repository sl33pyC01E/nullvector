# Masked Latent Map Prior

Status: exact-replayed CPU foundation. It proves the latent masking, condition,
checkpoint, raw-artifact, and replay contracts. It does not claim generative
quality and does not publish maps.

## Architecture

`forge.map_topology_neural_prior` loads the accepted 500-step topology codec by
exact checkpoint, source, and EMA hashes. The codec is CPU-loaded, frozen, and
used only to encode terrain/hazard/elevation into scale-four categorical code
grids. The prior cannot update the codec.

The compact spatial prior combines:

- a 513-entry token vocabulary: 512 frozen codec tokens plus one absorbing
  mask token;
- theme, global map-condition, downsampled mission-point, mask-ratio, and
  deterministic coordinate conditioning;
- residual 2-D convolutions over variable latent grids;
- four deterministic training mask families: random, rectangle, half-plane,
  and corridor;
- masked-only categorical cross entropy;
- parallel highest-confidence token reveal with stable row-major tie breaking.

The sampler begins fully masked and emits raw latent proposals plus quantized
uncertainty and a hash of every reveal step. It never invokes the topology
compiler. A raw latent is not a semantic map, a compiled map, or a runtime map
pack.

## Safety and replay

The CPU checkpoint uses `weights_only=True`, strict source/corpus/codec
authority, exact model loading, recursive tensor/container bounds, canonical
sidecars, immutable publication, and the 100 GiB disk floor. The smoke manifest
and raw bank are canonical JSON with self and file hashes. Validation rebuilds
the frozen latent batch, retrains the two-step prior from its fixed seed,
recreates the checkpoint semantics, resamples all six conditions, and compares
the complete raw bank exactly.

Fully rehashed claim-boundary or extra safety-gate fields fail closed. CUDA is
disabled and must remain uninitialized for the foundation path.

## Frozen smoke

Artifact:

`outputs/map_topology_neural_prior/smoke_v1`

Identities:

- prior source SHA-256:
  `76fcbce48e1ce20f5e1f28c20a38cc9c9d8c98be2cedccd221e7f95bb6145e15`
- frozen codec checkpoint SHA-256:
  `536d7e54e1da9f35ca9200353774121a59da69d9ea12853a5271b89fe06bce64`
- frozen codec EMA SHA-256:
  `0d90d210505fbda8fa3a319cc3a6d55ca1094252781159c4447f51bac6121d72`
- manifest SHA-256:
  `dfcf841f51885ec4d63ead50c3cdab5c9b77c0bccf1222d2b80d832284c3d31d`
- checkpoint file SHA-256:
  `c8842c237ced063d0a69ead027d8a6d5757cd407d15db5fc3b0640bc520f8fa9`
- model / EMA state SHA-256:
  `bd099840bf61173b965acf194c99720923fdd220cc3fce6f398ec781f6728b38` /
  `f96e6040362108de7f720dcbc3175470adf8f3ba5581923f266d36b7a8b2e8f4`

The six source conditions cover arena, rooms, caves, archipelago, garden, and
anomaly. All six raw token grids are unique and every masked token is revealed.
The deliberately tiny two-step loss moved 6.225658 to 6.091688. Fixed-mask
accuracy is only 0.009434 raw / 0.004717 EMA; these numbers are recorded to make
the absence of a quality claim explicit, not as an acceptance result.

## Commands

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
python -m forge.map_topology_neural_prior smoke `
  --corpus outputs/map_decorator_corpus_v1 `
  --output outputs/map_topology_neural_prior/UNIQUE
```

Exact replay uses the same environment and replaces `smoke` with `validate`.

## Next production slice

Before CUDA training, build a frozen latent corpus for all 3,096 maps in fresh
bounded worker shards. Each record must bind its source map, codec checkpoint,
EMA, shape/padding, conditions, token grid, and exact token hash. The production
trainer should use fresh short segments with optimizer/EMA/RNG resume, held-out
masked-token metrics, conditional counterfactuals, raw-sample diversity, and
repair-cost evaluation after deterministic compilation. No raw proposal should
become a map pack solely because its token loss is low.

