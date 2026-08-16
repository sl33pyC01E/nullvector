# Neural pixel-cell refinement

The continuous world VAE keeps its frozen 48x32x32 latent coordinate system, preserving compatibility with the action-conditioned world DiT. A 597,443-parameter fully convolutional neural refiner now operates after the decoder. It sees only the decoder's RGB output and learns the missing local correction for cell boundaries, neon contours, organs, shadows, and geometric structures.

The refiner was trained for 5,000 CUDA updates on 300 native teacher frames and evaluated on all 60 frames of a held-out world seed. Unlike the previous display-only unsharp filter, this stage is trained, source-bound, replayable, and part of the neural raster path.

| Held-out metric | Base VAE | Refined VAE |
|---|---:|---:|
| RGB MAE | 0.015568 | 0.007090 |
| RGB MSE | 0.000884 | 0.000627 |
| PSNR | 30.54 dB | 32.03 dB |
| Edge MAE | 0.015779 | 0.014654 |

That is a 54.46% MAE improvement and a 7.13% edge-error improvement on the unseen seed. The compact BF16 refiner is 1,210,338 bytes and runs after the existing 33.9 MB continuous VAE. The native demo's F7 mode now uses this learned composition directly; the former deterministic sharpening pass has been removed.
