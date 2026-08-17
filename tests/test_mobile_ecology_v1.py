import torch

from forge.mobile_ecology_v1 import MobileEcologyConfig, MobileEcologyPolicy
from forge.mobile_ecology_v1.model import MobileEcologyGraph


def test_mobile_ecology_shapes_budget_and_masked_neighbors() -> None:
    model = MobileEcologyPolicy(MobileEcologyConfig()).eval(); batch = 3
    self_features = torch.randn(batch, 94); resource = torch.randn(batch, 10, 4); neighbor = torch.randn(batch, 12, 14); mask = torch.zeros(batch, 12, dtype=torch.bool); mask[1, :3] = True; mask[2] = True
    value = model(self_features, resource, neighbor, mask)
    assert value.intent_logits.shape == (batch, 12)
    assert value.direction.shape == (batch, 2)
    assert value.urgency.shape == (batch,)
    assert model.parameter_count < 250_000
    assert torch.isfinite(value.intent_logits).all() and torch.isfinite(value.direction).all()


def test_mobile_ecology_graph_accepts_float_runtime_mask() -> None:
    graph = MobileEcologyGraph(MobileEcologyPolicy().eval()); inputs = (torch.zeros(1, 94), torch.zeros(1, 10, 4), torch.zeros(1, 12, 14), torch.zeros(1, 12))
    intent, direction, urgency = graph(*inputs)
    assert intent.shape == (1, 12) and direction.shape == (1, 2) and urgency.shape == (1,)
