# Developmental neural actuator

This subsystem is the first neural successor trained directly on the reviewed
v7 creature anatomy. It is deliberately narrower than a final world model: it
learns periodic locomotion over explicit cells, skeletons, muscles, contacts,
family mixtures, morphotypes, and traits. The procedural developmental system
remains the falsifiable teacher.

## Frozen authorities

- Morphology and locomotion review: `outputs/creature_stage_developmental/review_v7`
- Ten-specimen actuator corpus: `outputs/creature_stage_developmental_motion/corpus_v7_final_authority`
- Distilled rollout-1000 parent prior: `outputs/creature_stage_developmental_motion/rollout1000_prior_v2`
- Production v1: `outputs/creature_stage_developmental_motion/production_v3`

The corpus contains ten organisms, five families, two construction roles, 72
frames per loop, up to 560 active cells, 43 skeleton nodes, and 60 muscles. Its
semantic SHA-256 is
`f834a93f2a6ef17b049552e056ace630fd1d7a9bf48e7d2f712c99ba3b492984`.
The parent-prior semantic SHA-256 is
`4a4f4d4199e203a73791722ed0695599045e5c5ac15227ee1e6bb0765c74cf46`.

## Model

The 5.79-million-parameter successor has three coupled levels:

1. A condition encoder represents family, morphotype, continuous traits, and
   cyclic phase.
2. A graph-aware transformer predicts skeleton-node state and muscle
   activation from the current body state and the sealed parent prior.
3. Skeleton skinning moves every cell, followed by one topology-local neural
   residual block for soft tissue. The cell residual is bounded and cannot
   teleport anatomy independently of the skeleton.

Training uses 12-frame recurrent unrolls, balanced five-family batches, a
one-third seam quota, declining teacher forcing, explicit bone-length and
appendage losses, anti-copy pressure, BF16 CUDA, float32 losses, EMA weights,
and immutable 50-update checkpoints. A terminal resume at update 2,000 is an
exact no-op.

## Evaluation result

The update-2,000 EMA is the best v1 milestone. Over ten independent 72-frame
autoregressive loops it reaches 0.366 px mean cell RMSE, 0.543 px mean node
RMSE, 0.204 px seam RMSE, 0.156 p99 bone strain, 0.790 motion-energy ratio, and
0.776 appendage-energy ratio. Every family remains below 0.45 px mean cell
error. It beats the sealed rollout prior by 12.7 percent and does not collapse
to copied frames.

V1 remains honestly classified as `failed-quality`: muscle activation MAE is
0.206 against the 0.18 gate, and it does not beat an oracle baseline that is
given the exact previous target frame. The moving preview is stored at
`outputs/creature_stage_developmental_motion/production_v3/visual_rollout_0002000/developmental_actuator_0002000.gif`.
The next version should make predicted muscle contraction a stronger
same-frame cause of joint motion instead of treating it mainly as an auxiliary
and delayed recurrent signal.
