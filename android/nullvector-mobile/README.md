# Nullvector Mobile

Android deployment foundation for the Samsung Galaxy S25 Ultra.

The APK runs the complete compact neural loop through ONNX Runtime: structured world context, recurrent action/state prediction, and the 91k-parameter mobile frame VAE. It continuously advances a latent world, displays decoded neural frames, and reports the selected execution provider plus measured context/action/raster latency.

The current FP32 bundle is 52.9 MiB. The action student retains 74% of the desktop teacher's raw counter-action sensitivity at 32.7% of its parameters; its executed response is independently gated for nonzero magnitude and absolute teacher parity. The packaged APK is arm64-only and approximately 78 MiB.

Runtime order:

1. QNN/Hexagon when a custom QNN-enabled ONNX Runtime build is available.
2. NNAPI.
3. XNNPACK or standard ONNX Runtime CPU kernels.

The desktop monolithic foundation remains authoritative until rollout parity and 24–30 FPS are measured on a physical Galaxy S25 Ultra. QNN FP16/INT8 partitioning follows that baseline measurement.
