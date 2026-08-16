# Neural manipulation in the persistent ecology

`forge.nature_neural_feeding_v1` connects the accepted 8.01M-parameter
grasper controller to `NatureWorld` without replacing the legacy ecology path.
When explicitly enabled, flora and biomass stop being abstract calories.
They become persistent top-down material clumps that must be approached at a
body-sized standoff, reached by a selected appendage, grasped, carried to the
organism's live feeder cells, and passed through an intact digestive route.

The bridge uses two coordinate scales. World motion remains top-down and
continuous. The last reach, tether, feeder collision, and anatomy query run in
cell coordinates (`12 cells == 1 world unit`). The controller chooses the
appendage, reach target, grip force, brace, release, and throw vector; the
constraint solver applies momentum and cohesion. A failed or severed feeder
cannot be bypassed by chassis overlap.

Nutrition is family-specific. Animalian and humanoid organisms use flora and
biomass; machines favor mineral/charge; anomalies favor phase material; plants
can root-feed flora while retaining continuous light/water physiology. Injury
and death create tangible family-appropriate matter rather than transferring
energy instantly. Stored reserve is metabolized gradually, and fullness has a
long duration so ecosystem survival does not depend on perfect mouth contact
every few seconds.

The compatibility contract is intentional: constructing `NatureWorld`
without `feeding_system=` preserves the older abstract teacher behavior.
Constructing it with `NatureNeuralFeedingSystem` activates physical feeding
and includes the clump/controller state in the world semantic hash.
