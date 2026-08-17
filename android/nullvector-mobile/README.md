# Nullvector Mobile

Android deployment foundation for the Samsung Galaxy S25 Ultra.

The APK runs the complete compact neural loop through ONNX Runtime: structured world context, recurrent action/state prediction, and the 91k-parameter mobile frame VAE. It continuously advances a latent world, displays decoded neural frames, and reports the selected execution provider plus measured context/action/raster latency.

The recovery FP32 bundle is 53.1 MiB. The current candidate splits the runtime into a 13.2 MiB INT8 QDQ action graph and a 2.3 MiB FP32 physiology graph. It preserves actor outputs to numerical precision and diverges from the FP32 action rollout by 0.0090 latent units after 16 prediction-fed steps. The action student retains 74% of the desktop teacher's raw counter-action sensitivity at 32.7% of its parameters.

Runtime order:

1. QNN/Hexagon when a custom QNN-enabled ONNX Runtime build is available.
2. NNAPI.
3. XNNPACK or standard ONNX Runtime CPU kernels.

The desktop monolithic foundation remains authoritative until rollout parity and 24–30 FPS are measured on a physical Galaxy S25 Ultra. QNN FP16/INT8 partitioning follows that baseline measurement.
