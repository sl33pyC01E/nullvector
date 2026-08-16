# Learned living-body dynamics

This specialist graph network is the first learned replacement for causal body
state, rather than image presentation. It consumes per-cell tissue, organ,
appendage, current health/fluid/scar/connectivity, family, intervention fields,
literal feeder/digestive masks, physical mouth contact, food composition,
reserve/fullness, activity, and step duration. Bidirectional cellular adjacency
carries messages. The outputs are the next health/fluid/scar state for every
cell, seven whole-body capacities, absorption, nutrition, reserve, fullness,
energy release, contact/route confidence, and conserved clump mass.

Training pairs come from deterministic cuts, impacts, healing, idle ticks,
physical mouth contact, incompatible diets, full reserves, metabolism, feeder
ablation, digestive ablation, and structural severing across all 30 reviewed
organisms. Held-out identities are split by organism, not by individual
transition, preventing the same body from leaking across training and
validation. Successful absorption is oversampled only in training; validation
keeps its naturally sparse frequency.

Acceptance is intentionally causal: low held-out cell and organ-system error is
not enough. Healthy cells must not drift, invalid contact may not create food,
damaging a mouth or digestive route must suppress absorption, and recurrent
32-step reserve/fullness/energy/cell rollouts must remain bounded. Pixel contact
and mass/capacity conservation remain hard physical projections; the learned
network owns the state transition only after every one-step, causal, and
recurrent gate passes.

## Production-v2 baseline

The 5,000-step EMA checkpoint at
`outputs/living_body_dynamics_nn/production_v2_feeding/segment_020/` is a
measured rejected baseline, not a runtime authority. It reached held-out health
MAE 0.00188, fluid MAE 0.00208, and feeding-vector MAE 0.01945. It nevertheless
failed the causal gate because incompatible/full-reserve cases retained as much
as 0.0615 predicted absorption, and a 32-step rollout accumulated reserve MAE
0.2097 and normalized fullness MAE 0.2975. The immutable evaluation is at
`outputs/living_body_dynamics_nn/evaluation_v2_feeding_step5000/`.

## Production-v3 formulation

V3 does not extend the rejected flat head. It factors feeding into a learned
organ-route logit, learned conditional uptake/release efficiencies, and a hard
physical envelope supplied by the pixel simulation: explicit feeder contact,
family diet compatibility, available reserve capacity, material mass, and the
current feed/metabolize action. Sparse feeder and digestive populations receive
their own health/fluid/connectivity summaries instead of disappearing into a
whole-body mean pool.

Absorbed mass is the single conserved event. Nutrition, remaining clump mass,
reserve, fullness, and release are derived from it and the authored material
chemistry; they are no longer unrelated regression outputs. The energy head is
a bounded per-tick cost on top of conserved release rather than an absolute
recurrent prediction. This makes missed contact, incompatible food, a full
reserve, and non-feeding actions exactly incapable of creating uptake even in
an untrained network. Promotion remains forbidden until the new checkpoint
passes every one-step, ablation, and 32-step recurrent gate.
