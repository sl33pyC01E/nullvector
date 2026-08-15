# Length-projected developmental actuator v3

V3 closes the remaining physical-validity defect in the muscle-causal V2
actuator. It retains V2's learned same-frame muscle-to-joint force path, then
applies two differentiable, relaxation-limited skeleton projection passes to
the predicted joint state before raster cells are reconstructed. The
projection is part of the forward model and training graph rather than a
post-processing repair.

The immutable production authority is
`outputs/creature_stage_developmental_actuator_v3/production_v1`. It starts
from the exact V2 update-1,200 EMA, uses 24-frame recurrent unrolls, reduces
teacher forcing from 0.05 to zero, and performs 400 CUDA fine-tuning updates in
exact 50-update segments. A terminal resume is a verified no-op. The final
checkpoint SHA-256 is
`72c3120c1f37a26a83aa3e34c54b178ce646870b02ebf00ae17515f71c1371e5`
and its EMA state SHA-256 is
`8fbc6612094e19c452dd37643c9b7f6e01344d0a3e1d9c82ebbdeeb55574bbe9`.

## Accepted result

Update 400 is the selected V3 authority. It passes all twelve autonomous
quality and safety gates across both reviewed identities in each of the five
families:

- cell RMSE: 0.171 px
- node RMSE: 0.265 px
- muscle MAE: 0.033
- appendage motion-energy ratio: 0.973
- total motion-energy ratio: 0.975
- p99 bone strain: 0.082
- maximum bone strain: 0.116
- loop-seam RMSE: 0.118 px
- worst-family cell RMSE: 0.197 px
- copy-collapse fraction: 0.0053

The milestone curve improves monotonically from updates 100 through 400.
Relative to V2, V3 cuts cell error by 27.9%, node error by 35.5%, muscle error
by 62.5%, p99 bone strain by 72.8%, and loop-seam error by 36.7%, while
preserving 97.3% of the reviewed appendage motion energy.

The exact evaluation lives at
`outputs/creature_stage_developmental_actuator_v3/production_v1/evaluation_0000400`.
Its semantic SHA-256 is
`da1652c683f5bf03206e8f6f84e8fce98d2ada19efc594ab5c8cc860b775db24`.
The reviewed 72-frame neural-only loop is under `visual_rollout_0000400`; its
GIF SHA-256 is
`cf31c0b57c3cd367d2557c0366d72a26897382ae1c0fc0866f3ce81477bcd272`
and its MP4 SHA-256 is
`d6a2aca62dba49c8b3f983d9803d5b643e484cf4b7c0b815670b96fed2fc1fb0`.

This actuator predicts motion over the approved V7 developmental morphology
authority. It does not yet replace morphology generation, behavior selection,
world interaction, or final rendering. Those remain separate ensemble stages
until their own human and quantitative review gates are satisfied.
