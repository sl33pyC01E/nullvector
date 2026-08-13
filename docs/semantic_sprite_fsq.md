# Semantic sprite FSQ latent codec

This subsystem is the next neural construction stage after the production
multi-field diffusion model. It learns a compact discrete latent representation
of native 48px categorical sprite fields while preserving their separate
part-owner, material, and emission heads.

The design follows the original [VQ-VAE](https://arxiv.org/abs/1711.00937)
motivation but uses [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505):
four bounded scalar dimensions with levels `[8, 5, 5, 5]`, yielding an implicit
1,000-code vocabulary at a 12x12 latent grid. Fixed scalar levels avoid learned
codebook dead entries. The training path includes a continuous-autoencoder
warm-up before enabling quantization, motivated by recent work on avoiding
dimensional collapse in discrete autoencoders
([AE warm-up analysis](https://arxiv.org/abs/2605.06870)).

Important limits:

- The codec is a representation and reconstruction model. It does not yet have
  a generative prior, MaskGIT model, or diffusion model over its code indices.
- Decoding always projects aligned logits through the train-only legal
  `(part, material, emission)` tuple table. This guarantees categorical legality
  but does not itself guarantee topology, riggability, or aesthetic quality.
- The CPU smoke uses 20 deterministic procedural specimens to prove the model,
  quantizer, checkpoint, and artifact contracts. Production training uses the
  existing 32,768-specimen corpus with its frozen stratified split.
- No RGB/alpha refiner can modify the categorical authority.

Run the CPU foundation smoke:

```powershell
python -m forge.sprite_latent smoke --output outputs/sprite_latent/smoke_v1
python -m forge.sprite_latent validate outputs/sprite_latent/smoke_v1/smoke_manifest.json
```

The production sequence is intentionally staged: continuous warm-up,
quantized reconstruction, held-out topology/legality evaluation, immutable
checkpoint publication, and only then a learned masked prior over 12x12 codes.
