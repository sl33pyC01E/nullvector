# Infinite nature atlas

The playable 64×64 cell ecology is a streamed region, not the world boundary.
Every signed 64-bit region coordinate maps to a deterministic seed, biome,
resource pressures, ruins, danger, and background populations. Visited chunks
store exact departure snapshots; unvisited chunks cost no memory.

In the native demo the controlled organism can cross a region edge. Its exact
genome, cellular body, organ damage, scars, grafts, age, energy, inventory, and
objectives move into the newly generated ecology. Other organisms remain in the
departed region summary, permitting future conservative LOD simulation rather
than forcing one process to hold an astronomical cell grid.

