# Cellular world decoder adaptation

The recurrent Action-DiT improved held-out latent prediction, but the promoted world decoder erased most of that gain on newer cellular scenes. This additive adapter trains only the decoder half of the promoted world VAE.

The encoder and 48x32x32 latent contract remain byte-exact. Training mixes the four cellular training worlds with the original curriculum so the decoder can learn dense cellular scenes without silently abandoning its original domain. The fifth cellular world selects raw versus EMA weights. The sixth cellular world and the original held-out episode are used once for final gates.

Promotion requires at least 20% cellular reconstruction improvement, no more than 35% original-domain MAE regression, and an exact frozen encoder. A passing decoder still requires a new exact-parent pixel refiner before entering the playable composite.
