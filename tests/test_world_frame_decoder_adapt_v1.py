from __future__ import annotations

import copy

import torch

from forge.world_frame_decoder_adapt_v1.training import _decoder_parameters
from forge.world_frame_vae.contract import ModelConfig
from forge.world_frame_vae.model import WorldFrameVAE


def test_decoder_parameter_partition_preserves_encoder():
    model = WorldFrameVAE(ModelConfig(base=32, latent_channels=8))
    before = copy.deepcopy(model.state_dict())
    model.requires_grad_(False)
    decoder = _decoder_parameters(model)
    for parameter in decoder:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(decoder, lr=1e-4)
    loss = model.decode(torch.randn(1, 8, 4, 4)).mean()
    loss.backward()
    optimizer.step()
    after = model.state_dict()
    frozen = [name for name in after if not name.startswith(("from_latent.", "decoder.", "out."))]
    assert frozen
    assert all(torch.equal(before[name], after[name]) for name in frozen)
    assert any(not torch.equal(before[name], after[name]) for name in after if name not in frozen)
