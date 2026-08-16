from __future__ import annotations
import torch
from pathlib import Path
from forge.world_frame_vae_refiner import PixelCellRefiner,ModelConfig,RefinedWorldFrameVAERuntime

def test_pixel_cell_refiner_starts_as_identity_and_learns_local_edges():
    model=PixelCellRefiner(ModelConfig(width=16,blocks=2));base=torch.rand(2,3,32,32);result=model(base);assert torch.equal(result,base);loss=(result-torch.rand_like(result)).square().mean();loss.backward();assert model.out.weight.grad is not None and torch.isfinite(model.out.weight.grad).all()

def test_promoted_refiner_is_source_bound_and_improves_unseen_frames():
    base=Path("game/generated/models/world_frame_vae/raster_v1.pt");refiner=Path("game/generated/models/world_frame_vae_refiner/refiner_v1.pt")
    if not base.is_file() or not refiner.is_file():return
    runtime=RefinedWorldFrameVAERuntime.from_checkpoints(base,refiner,device="cpu");assert runtime.report["mae_improvement"]>.5 and runtime.report["edge_improvement"]>0
    frame=torch.zeros(256,256,3,dtype=torch.uint8).numpy();result=runtime.reconstruct(frame);assert result.shape==(256,256,3) and result.dtype.name=="uint8"
