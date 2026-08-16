from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from forge.world_frame_vae import ModelConfig,WorldFrameVAE,WorldFrameVAERuntime

def test_world_frame_vae_is_continuous_differentiable_and_exact_shape():
    model=WorldFrameVAE(ModelConfig(base=48,latent_channels=8));image=torch.rand(2,3,256,256);result,mean,logvar=model(image);assert result.shape==image.shape and mean.shape==(2,8,32,32) and logvar.shape==mean.shape;assert torch.isfinite(result).all() and 0<=float(result.min())<=float(result.max())<=1;result.mean().backward();assert model.statistics.weight.grad is not None

def test_compact_production_world_vae_reconstructs_native_viewport():
    checkpoint=Path(__file__).resolve().parents[1]/"game/generated/models/world_frame_vae/raster_v1.pt";runtime=WorldFrameVAERuntime.from_checkpoint(checkpoint,device="cpu");frame=np.zeros((256,256,3),np.uint8);frame[48:210,60:196]=(31,70,95);result=runtime.reconstruct(frame);assert result.shape==frame.shape and result.dtype==np.uint8;assert runtime.report["heldout_psnr_db"]>30 and runtime.report["heldout_edge_mae"]<.02
