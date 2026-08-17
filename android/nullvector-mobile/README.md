# Nullvector Mobile

Android deployment foundation for the Samsung Galaxy S25 Ultra.

Version 0.4 opens into the playable neural habitat rather than the model diagnostic screen. The cellular organism stays vertically aligned in a camera-following 2.5D world; touch controls movement and aim, thrown material follows a height/shadow ballistic arc, and persistent nutrient, fluid, and mineral clumps react to contact or impact. Absorbed resources are written into the organism's nutrient and energy channels before the next cellular-NCA tick. Model timings and the VAE diagnostic frame are available through `MODEL INFO`, but remain hidden during normal play.

The APK runs the compact neural loop through ONNX Runtime: structured world context, recurrent action/state prediction, a 492k-parameter per-cell physiology NCA, and the 91k-parameter mobile frame VAE. It advances the latent world and a held-out 48×48 cellular organism continuously, displays both neural outputs, and feeds organism health/neural state back into action control.

Two side-by-side preview flavors are available:

- `fp32`: full FP32 action graph. Package `world.nullvector.mobile.fp32`.
- `int8`: 13.2 MiB INT8 QDQ action graph plus 2.3 MiB FP32 actor graph. Package `world.nullvector.mobile.int8`.

Both flavors share the 2.0 MiB cellular NCA and mobile VAE. The current recovery build deliberately uses ONNX Runtime CPU with two intra-op threads; model-by-model QNN partitioning waits for physical S25 validation.

The INT8 candidate preserves actor outputs to numerical precision and diverges from the FP32 action rollout by 0.0090 latent units after 16 prediction-fed steps. The action student retains 74% of the desktop teacher's raw counter-action sensitivity at 32.7% of its parameters.

The desktop ensemble remains authoritative until rollout parity and 24–30 FPS are measured on a physical Galaxy S25 Ultra. QNN FP16/INT8 partitioning follows that baseline measurement.

Build both arm64 preview APKs with:

```powershell
.\gradlew.bat assembleFp32Release assembleInt8Release --no-daemon
```

Preview APKs are development-signed for sideloading. Production distribution will use a dedicated release key after physical-device validation.
