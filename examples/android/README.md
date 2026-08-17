# Android build

`nullvector-mobile-context-v1-debug.apk` is the first arm64 Android foundation build for the Samsung Galaxy S25 Ultra.

It runs the distilled structured-world encoder and 91k-parameter mobile neural rasterizer locally through ONNX Runtime, attempts NNAPI first, displays a real decoded game frame, and reports measured inference latency. The action model has a valid external Android ONNX export but is not bundled until physical-device profiling confirms its execution-provider partitioning and frame rate.

Current APK SHA-256: `164c719f80bbb2c92d3af16b1200ecdaa85e3773206cbf8bc2bed22718d40681`.
