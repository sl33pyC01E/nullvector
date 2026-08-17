# Bound world-frame refiner V2

V2 retrains the lightweight pixel refiner against the exact high-fidelity world
VAE artifact used by Composite Build 1. The base checkpoint hash and original
source hash are part of every segment and the final report; a refiner cannot be
silently attached to a different decoder.
