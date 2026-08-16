# Neural grounded graspers

The grasper controller extends the accepted grounded locomotion lineage with physical manipulation commands. It selects one of up to eight appendages, predicts contact engagement, a two-axis reach target, grip force, target type, and whole-body bracing. Targets include organisms, discrete objects, and raw material clumps; goals include inspection, consumption, carrying, and tearing.

The model does not directly teleport held things. Its outputs drive an equal-and-opposite point constraint. Excess strain can tear a weak material clump, while release removes the constraint. This preserves the project rule that learned intentions still act through physical cells, appendages, mass, and cohesion.

Throwing is a learned release command with a two-axis impulse. The constraint solver applies that impulse to an actually attached target and applies equal-and-opposite recoil to the organism; body bracing explicitly transfers the residual impulse into the ground.

Feeding is deliberately physical but forgiving. Every family derives a cell-level feeder aperture: literal mouth/jaw cells for animals, a lower head aperture for humanoids, terminal root feeders for plants, a transmuter aperture for anomalies, and a fuel port for machines. A food or fuel clump must overlap the small contact field around a live feeder cell, and those cells must still have a live cellular path to a digestive organ. Proximity to the body alone never creates nutrition. Severing a mouth, gut, or their connecting tissue therefore stops intake.

The neural controller receives the organism's own feeder anchor. Consumption is a two-stage learned manipulation: first reach and attach to a food clump, then keep the constraint engaged while bringing the clump to that anchor. This separates intentional transport from the authoritative cell-contact test.

Absorbed nutrition enters a bounded reserve instead of becoming an immediate hunger reset. The default reserve and 90-second fullness buffer make physical ingestion compatible with stable unattended ecosystems; exact durations remain balance parameters. Graspers can guide held food into the contact field, so this is not intended as a sub-pixel aiming challenge.
