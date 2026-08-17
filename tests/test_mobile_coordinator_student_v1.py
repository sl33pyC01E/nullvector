from __future__ import annotations

import torch

from forge.mobile_coordinator_student_v1 import MobileCoordinatorStudent, ModelConfig


def _inputs(batch: int = 2) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(0x4D4F42494C45)
    current = torch.rand(batch, 32, 32, 32, generator=generator)
    previous = torch.rand(batch, 32, 32, 32, generator=generator)
    global_state = torch.rand(batch, 44, generator=generator)
    previous_global = torch.rand(batch, 44, generator=generator)
    members = torch.rand(batch, 16, 64, generator=generator)
    member_mask = torch.ones(batch, 16, dtype=torch.bool)
    member_mask[:, -3:] = False
    society = torch.rand(batch, 64, generator=generator)
    sequence = torch.rand(batch, 24, 64, generator=generator)
    return current, previous, global_state, previous_global, members, member_mask, society, sequence


def test_mobile_coordinator_forward_contract_and_gradients() -> None:
    model = MobileCoordinatorStudent(ModelConfig(shared_width=64, macro_width=16, macro_blocks=2, member_width=48))
    outputs = model(*_inputs())
    assert [tuple(value.shape) for value in outputs] == [
        (2, 32, 32, 32), (2, 44), (2, 16, 6), (2, 16, 3),
        (2, 16), (2, 6), (2, 3), (2, 9), (2, 64), (2, 10),
        (2,), (2, 5, 64), (2, 5), (2, 5),
    ]
    loss = sum(value.float().mean() for value in outputs)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_mobile_coordinator_outputs_are_bounded_where_required() -> None:
    model = MobileCoordinatorStudent(ModelConfig(shared_width=64, macro_width=16, macro_blocks=1, member_width=48)).eval()
    outputs = model(*_inputs(batch=1))
    for index in (0, 1, 3, 8, 10, 11, 12, 13):
        assert torch.all(outputs[index] >= 0)
        assert torch.all(outputs[index] <= 1)
