import torch

from forge.recurrent_world_context_v1.contract import ContextModelConfig
from forge.recurrent_world_context_v1.model import WorldContextStateAdapter


def test_context_adapter_shape_and_gradients() -> None:
    model = WorldContextStateAdapter(ContextModelConfig(input_features=84, width=32, output_features=64)); value = torch.randn(3, 84); result = model(value); assert result.shape == (3, 64); result.square().mean().backward(); assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_context_adapter_rejects_nonfinite_input() -> None:
    model = WorldContextStateAdapter(); value = torch.zeros(1, 84); value[0, 2] = float("nan")
    try: model(value)
    except ValueError: pass
    else: raise AssertionError("nonfinite world context was accepted")
