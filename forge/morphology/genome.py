from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

import numpy as np

from .constants import FAMILIES, GENOME_VERSION


UINT32_MASK = 0xFFFFFFFF


def mix32(value: int) -> int:
    value &= UINT32_MASK
    value = (value ^ (value >> 16)) & UINT32_MASK
    value = (value * 0x7FEB352D) & UINT32_MASK
    value = (value ^ (value >> 15)) & UINT32_MASK
    value = (value * 0x846CA68B) & UINT32_MASK
    return (value ^ (value >> 16)) & UINT32_MASK


def stream_value(seed: int, slot: int) -> int:
    """Stable counter-based randomness with no hidden mutable RNG state."""
    return mix32((seed & UINT32_MASK) ^ mix32((slot + 1) * 0x9E3779B1))


def _integer(seed: int, slot: int, minimum: int, maximum: int) -> int:
    if maximum < minimum:
        raise ValueError("maximum must be greater than or equal to minimum")
    return minimum + stream_value(seed, slot) % (maximum - minimum + 1)


@dataclass(frozen=True, slots=True)
class MorphologyGenome:
    version: str
    seed: int
    family: int
    body_width: int
    body_height: int
    head_radius: int
    limb_length: int
    limb_thickness: int
    stance_width: int
    appendage_length: int
    appendage_count: int
    armor_depth: int
    weapon_length: int
    core_radius: int
    detail_count: int
    posture: int
    asymmetry: int
    dorsal_bias: int
    taper: int
    segmentation: int
    x_offset: int
    y_offset: int
    silhouette_variant: int
    subtype_id: int
    role_id: int
    palette_id: int
    palette_shift: int
    topology_seed: int
    detail_seed: int

    @property
    def family_name(self) -> str:
        return FAMILIES[self.family]

    def to_dict(self) -> dict[str, int | str]:
        payload = asdict(self)
        payload["family_name"] = self.family_name
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MorphologyGenome":
        values = dict(payload)
        values.pop("family_name", None)
        genome = cls(**values)
        genome.validate()
        return genome

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def condition_vector(self) -> np.ndarray:
        """A complete bounded conditioning vector for future neural models."""
        values = (
            (self.body_width - 10.0) / 10.0,
            (self.body_height - 14.0) / 12.0,
            (self.head_radius - 3.0) / 3.0,
            (self.limb_length - 6.0) / 6.0,
            (self.limb_thickness - 2.0) / 2.0,
            (self.stance_width - 4.0) / 6.0,
            (self.appendage_length - 5.0) / 8.0,
            (self.appendage_count - 1.0) / 3.0,
            (self.armor_depth - 1.0) / 3.0,
            (self.weapon_length - 5.0) / 8.0,
            (self.core_radius - 2.0) / 2.0,
            (self.detail_count - 3.0) / 6.0,
            (self.posture + 2.0) / 4.0,
            (self.asymmetry + 2.0) / 4.0,
            (self.dorsal_bias + 2.0) / 4.0,
            self.taper / 4.0,
            (self.segmentation - 1.0) / 4.0,
            (self.x_offset + 1.0) / 2.0,
            (self.y_offset + 1.0) / 2.0,
            self.silhouette_variant / 3.0,
            self.palette_id / 9.0,
            self.palette_shift / 7.0,
            self.topology_seed / float(UINT32_MASK),
            self.detail_seed / float(UINT32_MASK),
        )
        return np.asarray(values, dtype=np.float32)

    def validate(self) -> None:
        if self.version != GENOME_VERSION:
            raise ValueError(f"Unsupported genome version: {self.version!r}")
        if not 0 <= self.seed <= UINT32_MASK:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if not 0 <= self.family < len(FAMILIES):
            raise ValueError("family index is outside the morphology contract")
        ranges = {
            "body_width": (10, 20),
            "body_height": (14, 26),
            "head_radius": (3, 6),
            "limb_length": (6, 12),
            "limb_thickness": (2, 4),
            "stance_width": (4, 10),
            "appendage_length": (5, 13),
            "appendage_count": (1, 4),
            "armor_depth": (1, 4),
            "weapon_length": (5, 13),
            "core_radius": (2, 4),
            "detail_count": (3, 9),
            "posture": (-2, 2),
            "asymmetry": (-2, 2),
            "dorsal_bias": (-2, 2),
            "taper": (0, 4),
            "segmentation": (1, 5),
            "x_offset": (-1, 1),
            "y_offset": (-1, 1),
            "silhouette_variant": (0, 3),
            "subtype_id": (0, 19),
            "role_id": (0, 7),
            "palette_id": (0, 9),
            "palette_shift": (0, 7),
            "topology_seed": (0, UINT32_MASK),
            "detail_seed": (0, UINT32_MASK),
        }
        for name, (minimum, maximum) in ranges.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name}={value} is outside [{minimum}, {maximum}]")


def genome_from_seed(
    seed: int,
    family: int | str | None = None,
) -> MorphologyGenome:
    normalized_seed = int(seed) & UINT32_MASK
    if family is None:
        family_index = stream_value(normalized_seed, 0) % len(FAMILIES)
    elif isinstance(family, str):
        family_index = FAMILIES.index(family)
    else:
        family_index = int(family)
        if not 0 <= family_index < len(FAMILIES):
            raise ValueError(f"Unsupported family index: {family_index}")

    # Family-specific offsets widen the corpus without hiding any sampled trait.
    width_bias = (0, 1, -1, 0, 2)[family_index]
    height_bias = (1, 2, 2, -1, 0)[family_index]
    genome = MorphologyGenome(
        version=GENOME_VERSION,
        seed=normalized_seed,
        family=family_index,
        body_width=_integer(normalized_seed, 1, 11, 18) + width_bias,
        body_height=_integer(normalized_seed, 2, 15, 23) + height_bias,
        head_radius=_integer(normalized_seed, 3, 3, 6),
        limb_length=_integer(normalized_seed, 4, 6, 12),
        limb_thickness=_integer(normalized_seed, 5, 2, 4),
        stance_width=_integer(normalized_seed, 6, 4, 10),
        appendage_length=_integer(normalized_seed, 7, 5, 13),
        appendage_count=_integer(normalized_seed, 8, 1, 4),
        armor_depth=_integer(normalized_seed, 9, 1, 4),
        weapon_length=_integer(normalized_seed, 10, 5, 13),
        core_radius=_integer(normalized_seed, 11, 2, 4),
        detail_count=_integer(normalized_seed, 12, 3, 9),
        posture=_integer(normalized_seed, 13, -2, 2),
        asymmetry=_integer(normalized_seed, 14, -2, 2),
        dorsal_bias=_integer(normalized_seed, 23, -2, 2),
        taper=_integer(normalized_seed, 24, 0, 4),
        segmentation=_integer(normalized_seed, 25, 1, 5),
        x_offset=_integer(normalized_seed, 15, -1, 1),
        y_offset=_integer(normalized_seed, 16, -1, 1),
        silhouette_variant=_integer(normalized_seed, 17, 0, 3),
        subtype_id=0,
        role_id=_integer(normalized_seed, 22, 0, 7),
        palette_id=_integer(normalized_seed, 18, 0, 9),
        palette_shift=_integer(normalized_seed, 19, 0, 7),
        topology_seed=stream_value(normalized_seed, 20),
        detail_seed=stream_value(normalized_seed, 21),
    )
    # Subtype IDs are a stable dense product of family and silhouette variant.
    genome = MorphologyGenome(
        **{
            **asdict(genome),
            "subtype_id": family_index * 4 + genome.silhouette_variant,
        }
    )
    genome.validate()
    return genome
