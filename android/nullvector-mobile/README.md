# Nullvector Mobile

Android deployment foundation for the Samsung Galaxy S25 Ultra.

The first APK runs the distilled structured-world encoder through ONNX Runtime and reports the selected Android execution provider and measured latency. The action core and continuous frame VAE are exported beside it for physical-device profiling before they are bundled or downloaded as an asset pack.

Runtime order:

1. QNN/Hexagon when a custom QNN-enabled ONNX Runtime build is available.
2. NNAPI.
3. XNNPACK or standard ONNX Runtime CPU kernels.

The desktop monolithic foundation remains authoritative until Android output parity and 24–30 FPS are measured on the phone.
