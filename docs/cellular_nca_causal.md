# Cellular NCA causal curriculum v2

The v1 neural cellular automaton accurately reproduces ordinary 32-step wound,
fluid, neural, clotting, and scar trajectories, but its held-out organ ablation
probe exposed a specific failure: circulation and neural damage were causal,
while respiration and digestion were averaged away. V2 is an additive
fine-tuning curriculum for that missing capability. It does not alter the
frozen v1 package, checkpoint, corpus, manifest, or evaluation evidence.

Each update uses matched healthy/damaged copies of the same anatomy. The batch
is balanced over circulation, respiration, digestion, and neural systems.
Damage is applied only to existing organ cells, and bounded teacher pre-rolls
of 0, 4, 8, or 16 steps expose both onset and developed systemic consequences.
The ordinary two-step cellular loss preserves general dynamics. A second loss
matches the direction, magnitude, and spatial distribution of the selected
organ's counterfactual readout, normalized per system so subtle oxygen and
energy effects cannot be hidden by stronger fluid or neural signals.

Production defaults are deliberately compact: 512 updates, four immutable
128-step segments, batch eight (sixteen paired trajectories), BF16 CUDA with
float32 losses, and fresh-process bounded retries. Each segment has a 900-second
deadline. Canonical telemetry survives power loss, records access violations,
rejects invalid checkpoints before publication, and can explicitly attest a
valid checkpoint recovered between child publication and supervisor logging.
The parent EMA initializes the model; v2 uses a lower learning rate and retains
the same architecture.

```powershell
python -m forge.cellular_nca_causal train
python -m forge.cellular_nca_causal evaluate
python -m forge.cellular_nca_causal validate
```

If a validator-only source correction lands after a child has atomically
published a valid segment, `rebind` can copy that segment into a new output
authority.  It validates the old model, optimizer, RNG, history, and hashes,
requires the semantic training contract to remain identical, and records the
old/new checkpoint ancestry.  It never edits the interrupted output in place.

Evaluation and validation default to deterministic single-threaded CPU so the
published causal measurements can be replayed independently of the training
GPU. `validate` reruns all 32-step general trajectories and all four 16-step
organ counterfactuals, reconstructs the PNG byte-for-byte, and exact-compares
the complete manifest. `validate --metadata-only` is a faster artifact and
provenance inspection; it is not an exact evaluation replay.

Acceptance requires all four ablations to reduce their intended readout,
counterfactual error to improve materially over v1, and general wound/fluid/
neural rollout accuracy to remain bounded. A failed causal gate remains an
experimental result; the validator never promotes it by relaxing thresholds.
The strict schema fixes the organ census, all 12 rollout channels, all seven
acceptance gates, telemetry counts, runtime fields, and artifact bounds.
