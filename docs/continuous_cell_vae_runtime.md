# Continuous cell VAE runtime

The promoted rasterizer consumes the current continuous position and semantic
features of every living cell. It does not select a frozen sprite frame. The
physics, injury, and motion systems can move or remove cells, then ask the VAE
to render that exact posed body.

The runtime is bound to the calibrated held-out release. Its current boundary
is deliberate: morphology and physics supply cell coordinates; the neural
decoder supplies cell color, opacity, footprint, sub-pixel offset, and the
continuous 96x96 organism image. That deterministic coordinate supplier is a
teacher interface for the later recurrent student, not a claim that the final
renderer is already monolithic.
