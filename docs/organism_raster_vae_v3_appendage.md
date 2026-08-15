# Appendage-aware VAE continuation experiment

This isolated branch tested whether explicit appendage-kind channels and a dedicated high-resolution appendage head could improve the 114M-parameter VAE v3 without changing its accepted chassis/organ latent hierarchy.

It did not produce an acceptable successor. Three 600-step warm-start continuations exposed a precision/recall tradeoff:

- the parent held-out alpha IoU is 0.56317, appendage recall 0.92048, and neighborhood F1 0.68562;
- the precision-oriented continuation improved alpha IoU to 0.58525 but reduced appendage recall to 0.87581;
- the recall-oriented continuation raised recall to 0.97523 but reduced alpha IoU to 0.53716 and neighborhood F1 to 0.65991.

No continuation passed all declared gates and none is eligible for production. The sealed 1,200-step v3 calibration remains the balanced review authority.

The experiment demonstrates that scalar alpha weighting is the wrong next mechanism. The next model should give appendages structural latent tokens tied to their skeleton owner, root, joints, contact state, and paired partner. That representation can preserve topology and movement without inflating a soft silhouette around every thin structure.

The calibration writer now validates every target/parent/continuation panel independently before publishing its comparison artifact. Historical failed outputs remain evidence only and are not current-source authorities.
