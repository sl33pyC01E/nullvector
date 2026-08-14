# Frozen Neural Map Latent Corpus

Status: complete and independently source-replayed. This is a training corpus
of frozen codec tokens, not a trained prior, semantic map bank, compiled map
bank, or runtime asset.

## Contract

`forge.map_topology_neural_prior_corpus` mirrors the 216 authoritative source
shards and encodes every one of the 3,096 topology-v2 maps with the accepted
codec EMA. Each output shard stores:

- uint16 codec tokens in `[0, 511]`;
- the exact latent valid mask;
- four downsampled binary mission-point condition channels;
- fourteen float32 global conditions;
- theme and source sample indices;
- the complete ordered source-reference vector.

Arrays use deterministic ZIP/NPY bytes and carry artifact plus canonical
semantic hashes. Manifests bind the latent-corpus source, frozen codec
checkpoint/source/EMA, frozen source corpus and root manifest, sample identity,
shape, theme, member dtype/shape metadata, and exact array hashes.

## Failure containment

The supervisor allows at most two CPU workers and three attempts per shard.
Every attempt runs in a new Python process with CUDA disabled and one numerical
thread. It records exit code, unsigned Windows exit code, access-violation flag,
timeout, duration, and bounded stdout/stderr. A shard is published by atomic
directory replacement only after its local artifact validation succeeds.

After all build shards exist, a second 216-worker phase independently reloads
the source corpus and frozen codec, re-encodes every source map, and compares
all stored arrays plus the complete canonical manifest. The root manifest is
written only after this phase passes. Partial staging cannot be loaded as a
corpus.

## Published corpus

Artifact:

`outputs/map_topology_neural_prior_corpus/v1`

- corpus-builder source SHA-256:
  `bf321bdb745cbf70107ef6f0390b6c1d86339180935f1bd88e0961084008c2c8`
- root manifest SHA-256:
  `01df481c1b3300e41c0e9a70153679e48a9483fd30b0ac3b4e800cff3d198359`
- aggregate latent-corpus identity SHA-256:
  `bbcce0606f12d04d53e15e50c16852a8ee3d0e7262146e4c85c5965cf10f4d56`
- frozen codec checkpoint SHA-256:
  `536d7e54e1da9f35ca9200353774121a59da69d9ea12853a5271b89fe06bce64`
- frozen codec EMA SHA-256:
  `0d90d210505fbda8fa3a319cc3a6d55ca1094252781159c4447f51bac6121d72`
- frozen source corpus SHA-256:
  `16ed5f3b1a661e2bfc2abe9e16c39e9b8caaecba81f50fed6658cc4f73cffab8`

Exact census:

- 216/216 shards;
- 3,096 maps: 2,496 train, 576 validation, 24 test sentinels;
- all six themes and all eight shape/aspect profiles;
- 216/216 fresh-process source replays;
- 432 attempts total, zero retries, zero access violations, zero timeouts;
- 865 files / 9.34 MiB.

The fast root validator rechecks canonical manifests, every shard artifact and
semantic hash, the complete aggregate identity, claim boundary, gates, and
census. The build telemetry is the durable evidence that every shard also
passed the expensive frozen-source re-encoding phase.

## Commands

Build only to a new destination:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
python -m forge.map_topology_neural_prior_corpus build `
  --corpus outputs/map_decorator_corpus_v1 `
  --output outputs/map_topology_neural_prior_corpus/UNIQUE `
  --workers 2 --timeout-seconds 180
```

Fast root validation:

```powershell
python -m forge.map_topology_neural_prior_corpus validate `
  --corpus outputs/map_decorator_corpus_v1 `
  --output outputs/map_topology_neural_prior_corpus/v1
```

## Next boundary

The production masked-prior trainer can now stream homogeneous token shards
without repeatedly running the codec. It must remain segmented and immutable,
keep train/validation/test identity separation, measure masked-token quality by
mask type and theme/shape, test conditional counterfactuals, and keep raw
generation plus deterministic repair cost separate from safety acceptance.

