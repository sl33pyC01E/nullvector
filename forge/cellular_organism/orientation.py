from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Mapping

import numpy as np

from ..map_decorator.hashing import json_sha256


ORIENTATION_FORMAT: Final[str] = "nullvector-top-down-surface-physics-v1"
ORIENTATION_CONTRACT: Final[dict[str, object]] = {
    "format": ORIENTATION_FORMAT,
    "projection": "top_down_dorsal",
    "world_axes": ["screen_x", "screen_y"],
    "uniform_acceleration_xy": [0.0, 0.0],
    "scalar_screen_gravity_disabled": True,
    "living_motion_model": "planar_organ_actuation",
    "detached_motion_model": "planar_inertia_with_weak_reconnection_attraction",
    "external_fluid_model": "isotropic_surface_diffusion",
    "puddle_kernel": "mass_conserving_3x3_then_eight_neighbor_diffusion",
    "directional_trails_only_from_momentum": True,
    "python_runtime_required": False,
}
ORIENTATION_CONTRACT_SHA256: Final[str] = json_sha256(ORIENTATION_CONTRACT)


def orientation_manifest() -> dict[str, object]:
    return {
        "contract": {
            key: list(value) if isinstance(value, list) else value
            for key, value in ORIENTATION_CONTRACT.items()
        },
        "contract_sha256": ORIENTATION_CONTRACT_SHA256,
    }


def validate_orientation(value: Mapping[str, object]) -> None:
    if dict(value) != orientation_manifest():
        raise ValueError("Top-down surface-orientation contract differs.")


def top_down_simulation_defaults(source_defaults: Mapping[str, object]) -> dict[str, object]:
    """Project legacy anatomy-bank defaults into the active surface runtime."""

    projected = dict(source_defaults)
    projected["gravity"] = 0.0
    projected["legacy_scalar_gravity_disabled"] = True
    projected["orientation_contract_sha256"] = ORIENTATION_CONTRACT_SHA256
    return projected


@dataclass(slots=True)
class SurfacePuddleField:
    """Deterministic mass field for leaked fluid on a top-down surface.

    Deposits use a symmetric 3x3 kernel. Subsequent updates mix each pixel with
    its eight neighbors, so a stationary wound cannot acquire a screen-down
    bias. Momentum belongs to the emitting tissue; diffusion itself is radial.
    """

    width: int
    height: int
    diffusion_rate: float = 2.4
    evaporation_rate: float = 0.018
    amount: np.ndarray = field(init=False, repr=False)
    deposited_total: float = 0.0
    evaporated_total: float = 0.0

    def __post_init__(self) -> None:
        if self.width < 8 or self.height < 8:
            raise ValueError("Surface puddle field must be at least 8x8.")
        if not np.isfinite([self.diffusion_rate, self.evaporation_rate]).all() or self.diffusion_rate <= 0 or self.evaporation_rate < 0:
            raise ValueError("Surface puddle rates must be finite and non-negative.")
        self.amount = np.zeros((self.height, self.width), dtype=np.float32)

    def deposit(self, positions_xy: np.ndarray, amounts: np.ndarray) -> float:
        positions = np.asarray(positions_xy, dtype=np.float32)
        values = np.asarray(amounts, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 2 or values.shape != (len(positions),):
            raise ValueError("Surface deposit arrays have incompatible shapes.")
        if not np.isfinite(positions).all() or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Surface deposits must be finite and non-negative.")
        kernel = (
            (-1, -1, 0.05), (0, -1, 0.10), (1, -1, 0.05),
            (-1, 0, 0.10), (0, 0, 0.40), (1, 0, 0.10),
            (-1, 1, 0.05), (0, 1, 0.10), (1, 1, 0.05),
        )
        deposited = 0.0
        for (x_value, y_value), amount_value in zip(positions, values, strict=True):
            value = float(amount_value)
            if value <= 0:
                continue
            x = int(np.clip(np.rint(x_value), 1, self.width - 2))
            y = int(np.clip(np.rint(y_value), 1, self.height - 2))
            for dx, dy, weight in kernel:
                self.amount[y + dy, x + dx] += np.float32(value * weight)
            deposited += value
        self.deposited_total += deposited
        return deposited

    def step(self, dt: float) -> None:
        if not np.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("Surface diffusion dt must be finite in (0,0.1].")
        padded = np.pad(self.amount, 1, mode="edge")
        neighbors = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) * np.float32(0.125)
        mixing = np.float32(min(0.24, self.diffusion_rate * dt))
        self.amount += (neighbors - self.amount) * mixing
        before_evaporation = float(self.amount.sum(dtype=np.float64))
        retention = np.float32(max(0.0, 1.0 - self.evaporation_rate * dt))
        self.amount *= retention
        self.amount[self.amount < np.float32(1e-9)] = 0
        self.evaporated_total += before_evaporation - float(self.amount.sum(dtype=np.float64))

    @property
    def total(self) -> float:
        return float(self.amount.sum(dtype=np.float64))

    def centroid_xy(self) -> tuple[float, float] | None:
        total = self.total
        if total <= 0:
            return None
        yy, xx = np.indices(self.amount.shape, dtype=np.float64)
        weights = self.amount.astype(np.float64)
        return float((xx * weights).sum() / total), float((yy * weights).sum() / total)

    def rms_radius(self) -> float:
        centroid = self.centroid_xy()
        if centroid is None:
            return 0.0
        yy, xx = np.indices(self.amount.shape, dtype=np.float64)
        squared = (xx - centroid[0]) ** 2 + (yy - centroid[1]) ** 2
        return float(np.sqrt((squared * self.amount.astype(np.float64)).sum() / self.total))
