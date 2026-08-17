# Nullvector Mobile

Android deployment foundation for the Samsung Galaxy S25 Ultra.

The APK runs the distilled structured-world encoder and the 91k-parameter mobile frame VAE through ONNX Runtime, displays a real neural frame, and reports the selected Android execution provider plus measured context/raster latency. The action core remains an external profiling artifact until its device partitioning is verified.

Runtime order:

1. QNN/Hexagon when a custom QNN-enabled ONNX Runtime build is available.
2. NNAPI.
3. XNNPACK or standard ONNX Runtime CPU kernels.

The desktop monolithic foundation remains authoritative until Android output parity and 24–30 FPS are measured on the phone.
