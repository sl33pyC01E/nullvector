from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..creature_stage_developmental import TISSUES
from ..organism_raster_vae_v2.contract import CHECKPOINT_FORMAT as VAE_CHECKPOINT_FORMAT, OrganismVAEV2Config
from ..organism_raster_vae_v2.dataset import style_vector
from ..organism_raster_vae_v2.model import HierarchicalOrganismRasterVAE, HierarchicalOutput
from .contract import VAE_AUTHORITY, sha256_file


VAE_CHECKPOINT_SHA256 = "3a9673d95a7744e51c18a71249192ce15e840fc57996f1c0c3d67fdb663f69be"
TISSUE_TO_VAE = np.asarray((1, 3, 2, 2, 10, 4, 6, 6, 7, 5, 9, 14, 12, 3, 11), dtype=np.uint8)
PALETTE = np.asarray((
    (238, 104, 118), (228, 225, 191), (250, 69, 108), (240, 168, 120),
    (91, 145, 171), (55, 222, 245), (224, 44, 93), (105, 207, 238),
    (239, 158, 49), (247, 237, 91), (151, 86, 219), (94, 217, 78),
    (176, 68, 255), (165, 188, 203), (255, 117, 49),
), dtype=np.float32) / 255.0


@dataclass(slots=True)
class RasterCondition:
    living: Tensor
    family: Tensor
    subtype: Tensor
    role: Tensor
    genes: Tensor
    style: Tensor


def living_field_from_cells(
    cells: np.ndarray,
    tissue: np.ndarray,
    trait_fields: np.ndarray,
    appendage_owner: np.ndarray,
    family: int,
    *,
    health: np.ndarray | None = None,
) -> RasterCondition:
    """Project authoritative cells into the frozen 74-channel VAE language.

    Cell positions and state remain authoritative.  The VAE is a continuous
    appearance rasterizer; it never creates collision or anatomy cells.
    """
    count = len(cells)
    if cells.shape != (count, 2) or tissue.shape != (count,) or trait_fields.shape != (count, 15) or appendage_owner.shape != (count,):
        raise ValueError("grounded living-field inputs drifted")
    if not 0 <= family < 5 or not np.isfinite(cells).all() or not np.isfinite(trait_fields).all():
        raise ValueError("grounded living-field values drifted")
    # Framing belongs to the camera/rasterizer, not to cellular physics.  Fit
    # only the display projection when a predicted pose wanders near an edge;
    # collision cells and all metric calculations retain their native units.
    minimum, maximum = cells.min(0), cells.max(0)
    span = np.maximum(maximum - minimum, 1e-6)
    display_scale = min(1.0, 43.0 / float(span.max()))
    framed = (cells - (minimum + maximum) * .5) * display_scale
    center = np.asarray((23.5, 23.5), np.float32)
    coords = np.rint(framed + center).astype(np.int16)
    valid = (coords[:, 0] >= 0) & (coords[:, 0] < 48) & (coords[:, 1] >= 0) & (coords[:, 1] < 48)
    if float(valid.mean()) < .96:
        raise ValueError("grounded organism escaped the VAE raster frame")
    coords = coords[valid]; source_tissue = tissue[valid]; traits = trait_fields[valid]; owners = appendage_owner[valid]
    living = np.zeros((74, 48, 48), np.float32)
    counts = np.zeros((48, 48), np.float32)
    # Average collocated cells rather than allowing order-dependent overwrites.
    for index, (x_raw, y_raw) in enumerate(coords):
        x, y = int(x_raw), int(y_raw); counts[y, x] += 1
        vae_tissue = int(TISSUE_TO_VAE[int(source_tissue[index])])
        living[1 + vae_tissue, y, x] += 1
        material = min(9, max(1, vae_tissue % 10)); living[16 + material, y, x] += 1
        part = 1 if owners[index] < 0 else 2 + int(owners[index]) % 15; living[26 + part, y, x] += 1
        emission = .8 if int(source_tissue[index]) in (5, 9, 12, 14) else .08
        living[43, y, x] += emission
        # Eight soft organ/system fields derive from the actual diffused traits.
        systems = np.asarray((traits[index, 8], traits[index, 9], traits[index, 10], traits[index, 13], traits[index, 11], traits[index, 6], traits[index, 14], traits[index, 12]), np.float32)
        living[44:52, y, x] += systems
        living[52:60, y, x] += np.clip(systems * 3, 0, 3) / 3
        state_health = 1.0 if health is None else float(health[valid][index])
        living[60:70, y, x] += np.asarray((state_health, .92, .55, .60, .48, traits[index, 3], traits[index, 11], .05, traits[index, 11], .2), np.float32)
        rgb = PALETTE[int(source_tissue[index])]
        living[70:73, y, x] += rgb * (1 - .35 * emission) + np.asarray((.65, .95, 1.0)) * (.35 * emission)
        living[73, y, x] += 1
    occupied = counts > 0; living[0, occupied] = 1
    for channel in range(1, 74):
        living[channel, occupied] /= counts[occupied]
    # Restore categorical one-hot authority after collision averaging.
    for start, width in ((1, 15), (16, 10), (26, 17)):
        values = living[start:start + width]
        winner = values.argmax(0)
        values[:] = 0
        yy, xx = np.nonzero(occupied); values[winner[yy, xx], yy, xx] = 1
    living = np.clip(living, 0, 1)
    rgba = torch.from_numpy(living[70:74])
    occupancy = torch.from_numpy(living[0])
    emission = torch.from_numpy(living[43])
    style = style_vector(rgba, occupancy, emission)[None]
    genes = np.asarray((
        traits[:, 10].mean(), .65, traits[:, 11].mean(), traits[:, 11].mean() * .5,
        .65, .78, .55, .45, .08, .10, .35, traits[:, 1].mean(), traits[:, 1].mean(),
        traits[:, 1].mean(), min(1.0, count / 560), .55,
    ), np.float32)
    return RasterCondition(
        torch.from_numpy(living)[None], torch.tensor([family]), torch.tensor([family * 4]),
        torch.tensor([0]), torch.from_numpy(genes)[None], style,
    )


def load_frozen_vae(path: Path = VAE_AUTHORITY, *, device: str | torch.device = "cpu") -> tuple[HierarchicalOrganismRasterVAE, dict[str, object]]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 512 * 1024**2 or sha256_file(path) != VAE_CHECKPOINT_SHA256:
        raise ValueError("frozen organism VAE authority drifted")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != VAE_CHECKPOINT_FORMAT or payload.get("steps") != 2048:
        raise ValueError("frozen organism VAE checkpoint contract drifted")
    config = OrganismVAEV2Config(**payload["config"])
    model = HierarchicalOrganismRasterVAE(config)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval(), payload


@torch.no_grad()
def neural_raster(model: HierarchicalOrganismRasterVAE, condition: RasterCondition) -> HierarchicalOutput:
    device = next(model.parameters()).device
    return model(
        condition.living.to(device), condition.family.to(device), condition.subtype.to(device),
        condition.role.to(device), condition.genes.to(device), condition.style.to(device), sample=False,
    )
