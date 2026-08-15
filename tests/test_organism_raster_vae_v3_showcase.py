from __future__ import annotations

import torch

from forge.organism_raster_vae_v3_showcase import _render


def test_showcase_frame_geometry() -> None:
    target=torch.zeros((5,4,96,96)); prediction=torch.zeros_like(target); target[:,3,20:70,30:60]=1; prediction[:,3,22:68,31:59]=1
    image=_render(target,prediction,3)
    assert image.size==(466,1203)
