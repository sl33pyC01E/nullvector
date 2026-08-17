# Neural city layout v1

This stage replaces spatial settlement layout with a conditional categorical
network while retaining the existing architecture scaffold as a validator and
safety compiler.

## Contract

- 64×64 categorical city field.
- Eight classes: empty, road, wall, floor, door, utility, garden, storage.
- Conditions: family, culture, current project, biome, population, wealth,
  technology, building target, and a 16-value city style latent.
- The style latent controls the teacher geometry directly. It is not an opaque
  random seed the network would be forced to reverse.
- Full-blank training examples teach generation; partial masks teach editing,
  repair, and future image-to-image growth.
- Raw neural output is immutable. The compiler removes isolated structural
  hazards and connects unsupported occupied components with minimal roads.

## Current result

The selected 3,000-step model has 5.67M parameters. It trained in 111.2 seconds
on the development GPU, peaking at 5.10 GB reserved VRAM. On 512 held-out city
fields it reaches 99.39% masked-cell accuracy and 98.30% foreground accuracy.

The all-blank generation audit produces ten unique layouts across all five
families. The structural compiler changes 1.26% of cells on average and 2.66%
at worst. These are plausible alternative layouts rather than exact teacher
reconstructions. The model remains experimental until broader city-growth and
playability audits are added.

The inference-only EMA checkpoint is stored at
`examples/models/neural_city_layout_v1_ema.pt`. It excludes optimizer state and
is 22.7 MB.
