from __future__ import annotations

from typing import Final

import torch
from torch import Tensor
import torch.nn.functional as F


POINT_NAMES: Final[tuple[str, ...]] = ("start", "exit", "objective", "spawn")
CONDITION_CHANNELS: Final[tuple[str, ...]] = (
    *(f"point_{name}" for name in POINT_NAMES),
    *(f"near1_{name}" for name in POINT_NAMES),
    *(f"near2_{name}" for name in POINT_NAMES),
    "coord_x", "coord_y", "boundary_distance", "radial_distance", "mission_corridor",
)


def _centroid(field: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return normalized x/y centroid and a presence bit for B,H,W fields."""
    batch, height, width = field.shape
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32, device=field.device)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=field.device)
    mass = field.sum(dim=(1, 2)).clamp_min(1.0)
    cx = (field * x[None, None, :]).sum(dim=(1, 2)) / mass
    cy = (field * y[None, :, None]).sum(dim=(1, 2)) / mass
    present = (field.sum(dim=(1, 2)) > 0).to(torch.float32)
    return cx, cy, present


def build_spatial_conditions(point_conditions: Tensor, valid_mask: Tensor) -> Tensor:
    """Expand sparse sockets into deterministic local and global spatial hints.

    The corridor is a soft distance-to-segment field between the start and exit
    centroids.  It is a hint, not an authoritative route or topology mask.
    """
    if point_conditions.ndim != 4 or point_conditions.shape[1] != len(POINT_NAMES):
        raise ValueError("Prior-v2 point conditions must be B,4,H,W.")
    batch, _, height, width = point_conditions.shape
    if valid_mask.shape != (batch, 1, height, width) or valid_mask.dtype != torch.bool:
        raise ValueError("Prior-v2 valid mask disagrees with point conditions.")
    points = point_conditions.to(torch.float32)
    if not bool(torch.isfinite(points).all()) or bool(((points < 0) | (points > 1)).any()):
        raise ValueError("Prior-v2 point conditions must be finite in [0,1].")
    valid = valid_mask.to(torch.float32)
    near1 = F.max_pool2d(points, kernel_size=3, stride=1, padding=1)
    near2 = F.max_pool2d(points, kernel_size=5, stride=1, padding=2)
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32, device=points.device)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=points.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coord_x = xx[None, None].expand(batch, -1, -1, -1)
    coord_y = yy[None, None].expand(batch, -1, -1, -1)
    edge_y = 1.0 - yy.abs()
    edge_x = 1.0 - xx.abs()
    boundary = torch.minimum(edge_x, edge_y).clamp(0, 1)[None, None].expand(batch, -1, -1, -1)
    radial = torch.sqrt(xx.square() + yy.square()).clamp(0, 1)[None, None].expand(batch, -1, -1, -1)

    start_x, start_y, start_present = _centroid(points[:, 0])
    exit_x, exit_y, exit_present = _centroid(points[:, 1])
    ax = xx[None] - start_x[:, None, None]
    ay = yy[None] - start_y[:, None, None]
    vx = exit_x - start_x
    vy = exit_y - start_y
    length2 = (vx.square() + vy.square()).clamp_min(1.0e-6)
    projection = ((ax * vx[:, None, None] + ay * vy[:, None, None]) / length2[:, None, None]).clamp(0, 1)
    nearest_x = start_x[:, None, None] + projection * vx[:, None, None]
    nearest_y = start_y[:, None, None] + projection * vy[:, None, None]
    distance = torch.sqrt((xx[None] - nearest_x).square() + (yy[None] - nearest_y).square())
    corridor = torch.exp(-distance.square() / 0.08)
    corridor *= (start_present * exit_present)[:, None, None]
    expanded = torch.cat((
        points, near1, near2, coord_x, coord_y, boundary, radial, corridor[:, None]
    ), dim=1)
    if expanded.shape != (batch, len(CONDITION_CHANNELS), height, width):
        raise RuntimeError("Prior-v2 spatial-condition channel census drifted.")
    return expanded * valid
