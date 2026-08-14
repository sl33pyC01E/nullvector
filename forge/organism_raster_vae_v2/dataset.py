from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..organism_raster_vae.dataset import OrganismRasterCorpus


def style_vector(rgba: Tensor, occupancy: Tensor, emission: Tensor) -> Tensor:
    if rgba.shape != (4, 48, 48) or occupancy.shape != (48, 48) or emission.shape != (48, 48): raise ValueError("Organism VAE v2 style inputs drifted.")
    visible = occupancy.bool(); count = visible.sum().clamp_min(1); rgb = rgba[:3]
    mean = (rgb * occupancy[None]).sum(dim=(1, 2)) / count
    variance = ((rgb - mean[:, None, None]).square() * occupancy[None]).sum(dim=(1, 2)) / count
    return torch.cat((mean, variance.sqrt(), occupancy.mean()[None], (emission * occupancy).sum()[None] / count)).float()


class OrganismRasterCorpusV2(Dataset[dict[str, Tensor | str]]):
    def __init__(self) -> None:
        self.base = OrganismRasterCorpus(); self.samples = self.base.samples; self.indices_by_family = self.base.indices_by_family

    def __len__(self) -> int: return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        result = dict(self.base[index]); result["style"] = style_vector(result["rgba"], result["occupancy"], result["emission"])
        # The v1 living-field contract stores eight normalized role maps at
        # channels 52:60.  Re-expose their exact four-class identities so rare
        # cores/conduits cannot disappear inside a low aggregate MAE.
        result["system_role"] = (result["living_field"][52:60] * 3).round().long()
        return result
