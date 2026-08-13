# Production neural latent genetics

`forge.neural_fusion_production` is the learned successor to the earlier smoke-codec fusion experiment. It consumes only the independently validated, accepted epoch-24 EMA checkpoint from `sprite_latent_production_v1_run3`; the old experimental output remains immutable.

## Crossover operators

- `linear`: continuous global interpolation
- `spatial_weave`: sinusoidal spatial ownership bands
- `voronoi_mosaic`: deterministic cellular latent grafts with softened seams
- `radial_graft`: localized donor organ/limb graft
- `channel_crossover`: independent inheritance weights for all six FSQ dimensions
- `spectral_splice`: low-frequency anatomy from one blend and high-frequency semantic detail from the complementary blend

## Mutation operators

- `latent_gaussian`: bounded seeded latent noise
- `spatial_burst`: localized mutation focus
- `channel_phase`: selected FSQ dimension inversion
- `donor_transplant`: discrete donor patch insertion
- `phase_wave`: coherent field-wide phase modulation
- `none`: exact crossover control

Every result is quantized through the learned six-dimensional FSQ, decoded under blended parent conditioning, projected onto the frozen 69-row legal tuple table, cropped to the four-pixel safety margin, connected deterministically, and assigned a fresh graph rig. Lineage binds the parent IDs, operator parameters, field hash, accepted production manifest, checkpoint file, and EMA semantic hash.

The pilot compiles twelve varied cross-family/role pairings, all six crossover and mutation modes, three motion clips per child, seven presentation atlases, semantic/provenance archives, and a comparison contact sheet. A result is rejected if either parent contributes fewer than eight decoded pixels, occupancy leaves `[0.02,0.60]`, topology is disconnected after repair, any tuple is illegal, or fresh rig construction fails.

```powershell
python -m forge.neural_fusion_production compile
python -m forge.neural_fusion_production validate outputs/neural_fusion_production_v1/production_fusion_manifest.json
```
