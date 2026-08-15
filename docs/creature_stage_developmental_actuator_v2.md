# Muscle-causal developmental actuator v2

V2 is a separately reproducible successor to the first developmental neural
actuator. It warm-starts 172 compatible tensors from the exact update-2,000 v1
EMA, then inserts six new tensors that turn predicted muscle contraction into
a same-frame antagonistic force on skeleton joints. Dependence on the previous
muscle state is gated below ten percent at initialization. A node-motion loss
therefore has a nonzero gradient into the muscle head in the same frame.

The production authority is
`outputs/creature_stage_developmental_actuator_v2/production_v1`. It uses
24-frame recurrent unrolls, 0.18-to-zero teacher forcing, balanced five-family
batches, exact 50-update checkpoints, and 1,200 CUDA updates. Independent
0-to-100 runs produced identical model-state and EMA hashes at both sealed
boundaries. The final EMA SHA-256 is
`04e33328697bae90d31e92d176e44000845e13bf494d28ed6e7a83f7c373de77`.

## Result

The update-1,200 autonomous evaluation improves on v1:

- cell RMSE: 0.366 px to 0.237 px
- node RMSE: 0.543 px to 0.411 px
- muscle MAE: 0.206 to 0.087
- appendage energy ratio: 0.776 to 0.991
- total motion energy ratio: 0.790 to 0.994
- loop seam RMSE: 0.204 px to 0.186 px
- worst-family cell RMSE: 0.447 px to 0.273 px

It passes ten of eleven v2 quality gates. It remains `failed-quality` because
p99 bone strain is 0.302 against a 0.18 gate. The reviewed teacher's own p99
bone strain is only 0.008, so this is a genuine learned-geometry defect, not an
unfair threshold. The direct causal force path must be followed by a
differentiable length-preserving skeleton projection or an equivalently strong
learned constraint before this model can be promoted.
