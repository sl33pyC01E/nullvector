# Neural grounded graspers

The grasper controller extends the accepted grounded locomotion lineage with physical manipulation commands. It selects one of up to eight appendages, predicts contact engagement, a two-axis reach target, grip force, target type, and whole-body bracing. Targets include organisms, discrete objects, and raw material clumps; goals include inspection, consumption, carrying, and tearing.

The model does not directly teleport held things. Its outputs drive an equal-and-opposite point constraint. Excess strain can tear a weak material clump, while release removes the constraint. This preserves the project rule that learned intentions still act through physical cells, appendages, mass, and cohesion.

Throwing is a learned release command with a two-axis impulse. The constraint solver applies that impulse to an actually attached target and applies equal-and-opposite recoil to the organism; body bracing explicitly transfers the residual impulse into the ground.

Feeding is deliberately physical but forgiving. Every family derives a cell-level feeder aperture: literal mouth/jaw cells for animals, a lower head aperture for humanoids, terminal root feeders for plants, a transmuter aperture for anomalies, and a fuel port for machines. A food or fuel clump must overlap the small contact field around a live feeder cell, and those cells must still have a live cellular path to a digestive organ. Proximity to the body alone never creates nutrition. Severing a mouth, gut, or their connecting tissue therefore stops intake.

The neural controller receives the exact centroid of the organism's physical feeder cells, not an approximate component anchor. Consumption is a two-stage learned manipulation: first reach and attach to a food clump, then keep the constraint engaged while bringing the clump to that anchor. This separates intentional transport from the authoritative cell-contact test while ensuring every family aims at the same cells the intake collision actually checks.

The production fit uses the larger desktop-quality controller (384 channels, six conditioned blocks) and gives extra loss authority to appendage selection, feeder reach, and throw impulse. Raw and EMA weights are both evaluated; the runtime is published from the better validation candidate and the quality gates are never relaxed to make a run pass.

Absorbed nutrition enters a bounded reserve instead of becoming an immediate hunger reset. The default reserve and 90-second fullness buffer make physical ingestion compatible with stable unattended ecosystems; exact durations remain balance parameters. Graspers can guide held food into the contact field, so this is not intended as a sub-pixel aiming challenge.

## Accepted desktop baseline

`outputs/creature_stage_neural_grasper_v1/production_v2` is the accepted neural controller. It contains an 8,010,637-parameter, 16.0 MB BF16 runtime selected from the EMA weights. On 960 held-out manipulation cases it reached 95.255% appendage accuracy, 0.988 engagement F1, 0.0163 normalized reach MAE, 0.0144 force MAE, 0.0257 brace MAE, perfect target-type and release classification, and 0.0148 throw-impulse MAE. Every declared gate passed. The earlier `production_v1` experiment is retained locally as failed evidence and is not a runtime candidate.
