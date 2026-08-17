# Android build

`nullvector-mobile-context-v1-debug.apk` is the first arm64 Android foundation build for the Samsung Galaxy S25 Ultra.

It runs the distilled structured-world encoder locally through ONNX Runtime, attempts NNAPI first, and displays the selected backend and measured inference latency. The action model and frame VAE have valid Android ONNX exports but are not bundled into this APK until physical-device profiling confirms their execution-provider partitioning and frame rate.

Current APK SHA-256: `4127b9c48452bb83907b23eda9758ad485bd124e0ae9f09e3d70d99e57934dd1`.
