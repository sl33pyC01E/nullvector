from __future__ import annotations

import torch

from forge.mobile_viewport_decoder_v1.contract import ModelConfig
from forge.mobile_viewport_decoder_v1.model import MobileViewportDecoder


def test_mobile_decoder_is_full_frame_and_mobile_sized():
    model = MobileViewportDecoder(ModelConfig()).eval()
    output = model(torch.zeros(2, 48, 32, 32))
    assert output.shape == (2, 3, 256, 256)
    assert torch.isfinite(output).all()
    assert 0 <= float(output.min()) <= float(output.max()) <= 1
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000


def test_mobile_decoder_rejects_wrong_latent_geometry():
    model = MobileViewportDecoder(ModelConfig()).eval()
    try:
        model(torch.zeros(1, 48, 16, 16))
    except ValueError as error:
        assert "latent drifted" in str(error)
    else:
        raise AssertionError("wrong latent geometry was accepted")
