# Neural grounded locomotion

This slice connects the approved developmental creature bodies to learned motion without discarding the physical scaffold.

## Authority boundary

- Cells, tissue identity, connectivity, skeletons, muscles, ground contacts, and collision geometry remain authoritative simulation state.
- The grounded teacher uses persistent world-space contact anchors, PBD limb tethers, muscle forces, and explicit body reaction. It never rotates or mirrors a creature to face its travel direction.
- Locomotion belongs to inherited components. Legs step, roots drag, wheels roll, and native anomaly phase fields float. A graft can move any of these components into another family.
- The neural motion model predicts local cell pose, local cell velocity, and body travel from those states. It is warm-started from the exact sealed rollout update-1000 EMA.
- The 384-token learned transformer weights are preserved exactly and evaluated on an expanded 560-cell canvas. A contact-aware graph refiner anchors recurrent poses and prevents velocity accumulation.

## VAE raster boundary

The trained 35.6M-parameter hierarchical organism VAE is the appearance rasterizer. Predicted cells are projected into its 74-channel living field: occupancy, 15 tissues, materials, parts, emission, eight physiology systems, system roles, ten cell-state channels, and RGBA.

The VAE does not invent physics cells. Simulation selection, damage, organs, bonds, collision, severing, and fluids continue to address the source cells. The raster framing may center or fit a pose for display, but it never changes simulation coordinates.

## Evaluation

Production evaluation must use all five grafted organisms for complete 72-frame prediction-fed cycles. It reports position and velocity error, appendage and planted-contact error, motion energy, loop seam, body travel, and frozen-VAE transfer IoU. Safety and quality gates remain separate; an artifact is not promoted merely because it runs.

Commands:

```powershell
python -m forge.creature_stage_neural_grounded train --output outputs/creature_stage_neural_grounded/production_v1 --updates 1600 --batch-size 10 --device cuda
python -m forge.creature_stage_neural_grounded evaluate --checkpoint outputs/creature_stage_neural_grounded/production_v1/grounded_motion_0001600.pt --output outputs/creature_stage_neural_grounded/evaluation_v1 --device cuda
```

## Frozen production lineage

The grounded successor was trained as an additive lineage. Every stage binds
the exact parent checkpoint, EMA state, teacher semantic hash, configuration,
and source hash; none of the earlier checkpoints are overwritten.

1. `production_v1`: 1,600 CUDA updates from the sealed rollout update-1000
   authority. This introduced the 560-cell contact-aware graph refiner and
   prediction-fed six-frame training.
2. `production_v2_800`: an 800-update low-rate refinement emphasizing planted
   contacts, appendages, and direct pose prediction.
3. `production_v3`: continuation through update 1,100, retaining the best
   copy-baseline improvement of the lineage.
4. `production_v4`: continuation through update 1,350 with additional
   top-tail wrap-transition pressure. This is the best pose/contact model and
   the selected research checkpoint, but it is deliberately not promoted as a
   finished locomotion authority.

The final five-family, 360-frame prediction-fed evaluation records:

- cell position MAE: `0.200376660 px`
- appendage MAE: `0.227660581 px`
- planted-contact MAE: `0.187670901 px`
- velocity MAE: `0.054985121 px`
- motion-energy ratio: `0.909301109`
- body advance ratio: `1.022521060`
- frozen-VAE silhouette IoU: `0.476076975`
- improvement over copying the previous frame: `9.568%`
- worst single-cell loop-transition error: `2.802919388 px`

Ten of twelve strict gates pass. The copy-baseline gate requires at least 10%
improvement and the worst-cell wrap gate requires at most 0.35 px, so the
evaluation remains `failed-quality` and `promotion_eligible=false`. The result
is usable as a neural pose proposal inside the deterministic contact solver;
it is not yet allowed to replace that solver.

The next model iteration should replace the worst-cell seam objective with a
cyclic phase representation plus a short closed-loop discriminator, and should
train the VAE raster bridge on motion-conditioned fields rather than only
measuring frozen transfer. This preserves the scaffold while moving the visible
creature and eventually the contact controller toward the intended learned
engine.
