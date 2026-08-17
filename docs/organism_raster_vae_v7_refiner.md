# Anatomical VAE neural cell refiner

V7 keeps the current-corpus V6 anatomical VAE frozen and learns a compact 96×96 refinement stage. It consumes the VAE render and the 48×48 living cell field, then predicts continuous RGB and alpha corrections. This is a neural decoder stage, not an occupancy-mask projection.

The parent render cache is immutable and hash-bound. Training publishes small 200-update checkpoints. Promotion requires held-out silhouette IoU, appendage recall, RGBA error, and improvement over the V6 parent.
