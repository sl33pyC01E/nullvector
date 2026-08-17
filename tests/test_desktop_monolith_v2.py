from __future__ import annotations

import torch

from forge.desktop_monolith_v2 import DesktopWorldMonolith, ModelConfig
from forge.world_latent_dit.contract import ModelConfig as ActionConfig


def _inputs(batch=1):
    g=torch.Generator().manual_seed(31337)
    return (torch.rand(batch,48,32,32,generator=g),torch.rand(batch,48,32,32,generator=g),torch.zeros(batch,dtype=torch.long),torch.rand(batch,4,generator=g),torch.rand(batch,64,generator=g),torch.rand(batch,128,generator=g),torch.rand(batch,32,32,32,generator=g),torch.rand(batch,32,32,32,generator=g),torch.rand(batch,44,generator=g),torch.rand(batch,44,generator=g),torch.rand(batch,16,64,generator=g),torch.ones(batch,16,dtype=torch.bool),torch.rand(batch,64,generator=g),torch.rand(batch,24,64,generator=g))


def test_identity_fusion_preserves_action_parent_at_initialization():
    action=ActionConfig(width=64,layers=1,heads=4,patch=4)
    model=DesktopWorldMonolith(ModelConfig(width=64,heads=4,fusion_layers=2,macro_patch=4),action).eval(); values=_inputs()
    with torch.no_grad():
        expected=model.visual(*values[:6]); actual=model(*values)[0]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_desktop_monolith_output_contract():
    action=ActionConfig(width=64,layers=1,heads=4,patch=4)
    model=DesktopWorldMonolith(ModelConfig(width=64,heads=4,fusion_layers=1,macro_patch=4),action)
    outputs=model(*_inputs(batch=2))
    assert [tuple(value.shape) for value in outputs] == [(2,48,32,32),(2,32,32,32),(2,44),(2,16,6),(2,16,3),(2,16),(2,6),(2,3),(2,9),(2,64),(2,10),(2,),(2,5,64),(2,5),(2,5)]
