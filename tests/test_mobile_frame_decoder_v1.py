import pytest
import torch

from forge.mobile_frame_decoder_v1.contract import MobileDecoderConfig
from forge.mobile_frame_decoder_v1.model import build_model


def test_mobile_decoder_shape_and_parameter_budget() -> None:
    model = build_model(MobileDecoderConfig()).eval()
    with torch.inference_mode(): output = model(torch.zeros(1, 48, 32, 32))
    assert output.shape == (1, 3, 256, 256)
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000


def test_mobile_decoder_rejects_wrong_latent() -> None:
    with pytest.raises(ValueError): build_model(MobileDecoderConfig())(torch.zeros(1, 48, 16, 16))
