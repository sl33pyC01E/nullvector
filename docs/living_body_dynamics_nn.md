# Learned living-body dynamics

This specialist graph network is the first learned replacement for causal body
state, rather than image presentation. It consumes per-cell tissue, organ,
appendage, current health/fluid/scar/connectivity, family, and intervention
fields. Bidirectional cellular adjacency carries messages. The outputs are the
next health/fluid/scar state for every cell and seven whole-body capacities.

Training pairs come from deterministic cuts, impacts, healing, and idle ticks of
all 30 reviewed organisms. Held-out identities are split by organism, not by
individual transition, preventing the same body from leaking across training
and validation.

Acceptance is intentionally causal: low held-out cell and organ-system error is
not enough. Healthy cells must not drift, distant cells must not respond to a
local intervention, damage must alter the relevant system, and fluid may not be
created. The deterministic substrate remains authoritative until those gates
hold over recurrent rollouts.
