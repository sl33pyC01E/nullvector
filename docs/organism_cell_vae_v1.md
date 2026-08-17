# Continuous organism cell VAE

This rasterizer treats organism pixels as living cells. The morphology and physics systems supply continuous posed cell centers plus tissue, family, traits, appendage class, and phase. A conditional VAE predicts per-cell color, opacity, footprint, and bounded positional correction, then renders the cells through a differentiable splat field.

The architecture preserves sub-cell locomotion that a rounded 48×48 convolutional input cannot recover. The splat primitive is explicit scaffolding; a later action-model student can distill the complete renderer.
