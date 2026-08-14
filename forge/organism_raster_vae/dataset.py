from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .contract import ANATOMY_MANIFEST, FROZEN_UPSTREAM, PHYSIOLOGY_MANIFEST, TRAUMA_MANIFEST, sha256_file


SYSTEM_COUNT: Final[int] = 8
TISSUE_COUNT: Final[int] = 15
MATERIAL_COUNT: Final[int] = 10
PART_COUNT: Final[int] = 17
GENE_NAMES: Final[tuple[str, ...]] = (
    "metabolic_rate", "digestion_efficiency", "fluid_regeneration_rate", "tissue_regeneration_rate",
    "basal_energy_capacity", "reproduction_energy_threshold", "gestation_seconds", "offspring_energy_fraction",
    "mutation_rate", "mutation_scale", "litter_size", "symmetry_score", "chassis_symmetry",
    "appendage_symmetry", "cell_density", "organ_density",
)
ANATOMY_KEYS: Final[frozenset[str]] = frozenset({
    "format", "position_xy", "part_owner", "material", "emission", "tissue", "organ_id", "cell_flags",
    "max_health", "fluid_capacity", "fluid_initial", "nutrient_initial", "energy_initial", "mass", "stiffness",
    "bond_ab", "bond_kind", "bond_rest", "bond_strength", "bond_conductance",
})
PHYSIOLOGY_KEYS: Final[frozenset[str]] = frozenset({"system_membership", "system_role", "system_weight"})
TRAUMA_KEYS: Final[frozenset[str]] = frozenset({"heal_class", "clotting_weight", "scar_bias", "regrowth_weight", "bond_repair_weight", "bond_magnetic_weight"})


def _read_manifest(path: Path, expected_sha: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha:
        raise ValueError(f"Frozen organism VAE upstream drifted: {path}")
    payload = json.loads(path.read_text("utf-8"))
    if not isinstance(payload, dict) or payload.get("status") not in {"passed", "ready"} or not isinstance(payload.get("gates"), dict) or not all(payload["gates"].values()):
        raise ValueError(f"Frozen organism VAE upstream is not fully accepted: {path}")
    return payload


def _load_npz(path: Path, artifact: dict[str, Any], expected_keys: frozenset[str]) -> dict[str, np.ndarray]:
    if artifact.get("path") is None or artifact.get("sha256") != sha256_file(path) or artifact.get("bytes") != path.stat().st_size:
        raise ValueError(f"Organism artifact identity failed: {path}")
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= 32 * 1024 * 1024:
        raise ValueError(f"Organism artifact is missing or oversized: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError(f"Organism artifact member census drifted: {path}")
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def _one_hot(values: np.ndarray, classes: int) -> np.ndarray:
    if values.dtype.kind not in "ui" or int(values.min(initial=0)) < 0 or int(values.max(initial=0)) >= classes:
        raise ValueError("Organism categorical raster exceeds its vocabulary.")
    return np.moveaxis(np.eye(classes, dtype=np.float32)[values], -1, 0)


def _normalized_genes(entry: dict[str, Any]) -> np.ndarray:
    genome = entry["genome"]; symmetry = entry["symmetry"]["after"]; summary = entry["summary"]
    raw = np.asarray([
        float(genome["metabolic_rate"]) / 1.5, float(genome["digestion_efficiency"]), float(genome["fluid_regeneration_rate"]) / .06,
        float(genome["tissue_regeneration_rate"]) / .03, float(genome["basal_energy_capacity"]) / 160,
        float(genome["reproduction_energy_threshold"]) / 180, float(genome["gestation_seconds"]) / 25,
        float(genome["offspring_energy_fraction"]), float(genome["mutation_rate"]) / .12, float(genome["mutation_scale"]) / .2,
        float(genome["litter_size"]) / 3, float(symmetry["weighted_score"]), float(symmetry["chassis_match"]),
        float(symmetry["paired_appendage_match"]), float(summary["physical_cell_count"]) / (48 * 48), float(summary["organ_count"]) / 32,
    ], dtype=np.float32)
    return np.clip(raw, 0, 1)


@dataclass(slots=True)
class OrganismRasterSample:
    sample_id: str
    family: int
    subtype: int
    role: int
    living_field: np.ndarray
    rgba: np.ndarray
    occupancy: np.ndarray
    tissue: np.ndarray
    material: np.ndarray
    part: np.ndarray
    emission: np.ndarray
    physiology: np.ndarray
    cell_state: np.ndarray
    genes: np.ndarray


class OrganismRasterCorpus(Dataset[dict[str, Tensor | str]]):
    """Strict 45-identity cellular corpus projected into aligned living rasters."""

    def __init__(self) -> None:
        anatomy = _read_manifest(ANATOMY_MANIFEST, FROZEN_UPSTREAM["anatomy_manifest_sha256"])
        physiology = _read_manifest(PHYSIOLOGY_MANIFEST, FROZEN_UPSTREAM["physiology_manifest_sha256"])
        trauma = _read_manifest(TRAUMA_MANIFEST, FROZEN_UPSTREAM["trauma_manifest_sha256"])
        anatomy_rows = {row["sample_id"]: row for row in anatomy["offspring"]}
        physiology_rows = {row["sample_id"]: row for row in physiology["identities"]}
        trauma_rows = {row["sample_id"]: row for row in trauma["identities"]}
        if len(anatomy_rows) != 45 or set(anatomy_rows) != set(physiology_rows) or set(anatomy_rows) != set(trauma_rows):
            raise ValueError("Organism VAE upstream identity census drifted.")
        self.samples = [self._compile(anatomy_rows[key], physiology_rows[key], trauma_rows[key]) for key in sorted(anatomy_rows)]
        families = [sample.family for sample in self.samples]
        self.indices_by_family = {family: [index for index, sample in enumerate(self.samples) if sample.family == family] for family in range(5)}
        if [families.count(family) for family in range(5)] != [11, 10, 9, 8, 7]:
            raise ValueError("Organism VAE frozen family census drifted.")

    @staticmethod
    def _compile(entry: dict[str, Any], physiology_entry: dict[str, Any], trauma_entry: dict[str, Any]) -> OrganismRasterSample:
        anatomy_path = ANATOMY_MANIFEST.parent / entry["arrays"]["path"]
        anatomy = _load_npz(anatomy_path, entry["arrays"], ANATOMY_KEYS)
        physiology_path = PHYSIOLOGY_MANIFEST.parent / physiology_entry["arrays"]["path"]
        phys = _load_npz(physiology_path, physiology_entry["arrays"], PHYSIOLOGY_KEYS)
        trauma_path = TRAUMA_MANIFEST.parent / trauma_entry["arrays"]["path"]
        trauma = _load_npz(trauma_path, trauma_entry["arrays"], TRAUMA_KEYS)
        positions = anatomy["position_xy"]
        count = len(positions)
        if positions.shape != (count, 2) or len(np.unique(positions, axis=0)) != count or count != entry["summary"]["physical_cell_count"]:
            raise ValueError("Organism cell coordinates/census drifted.")
        if phys["system_role"].shape != (SYSTEM_COUNT, count) or phys["system_weight"].shape != (SYSTEM_COUNT, count):
            raise ValueError("Organism physiology alignment drifted.")
        if any(trauma[name].shape != (count,) for name in ("heal_class", "clotting_weight", "scar_bias", "regrowth_weight")):
            raise ValueError("Organism trauma alignment drifted.")
        x = positions[:, 0].astype(np.int64); y = positions[:, 1].astype(np.int64)
        if bool(np.any(x < 0) or np.any(x >= 48) or np.any(y < 0) or np.any(y >= 48)):
            raise ValueError("Organism cells exceed native raster bounds.")
        occupancy = np.zeros((48, 48), dtype=np.float32); occupancy[y, x] = 1
        categorical: dict[str, np.ndarray] = {}
        for name in ("tissue", "material", "part_owner"):
            field = np.zeros((48, 48), dtype=np.uint8); field[y, x] = anatomy[name]; categorical[name] = field
        emission = np.zeros((48, 48), dtype=np.float32); emission[y, x] = anatomy["emission"].astype(np.float32) / 3
        physiology_field = np.zeros((SYSTEM_COUNT, 48, 48), dtype=np.float32); physiology_field[:, y, x] = phys["system_weight"]
        role_field = np.zeros((SYSTEM_COUNT, 48, 48), dtype=np.float32); role_field[:, y, x] = phys["system_role"].astype(np.float32) / 3
        state_values = np.stack((
            anatomy["max_health"] / 2.2, anatomy["fluid_initial"] / np.maximum(anatomy["fluid_capacity"], 1e-6), anatomy["nutrient_initial"],
            anatomy["energy_initial"], anatomy["mass"] / 1.55, anatomy["stiffness"], trauma["clotting_weight"], trauma["scar_bias"],
            trauma["regrowth_weight"], trauma["heal_class"].astype(np.float32) / 5,
        )).astype(np.float32)
        cell_state = np.zeros((10, 48, 48), dtype=np.float32); cell_state[:, y, x] = np.clip(state_values, 0, 1)
        palette = entry["palette"]; mids = np.asarray(palette["material_mid_rgb"], dtype=np.float32) / 255
        rgb = np.zeros((3, 48, 48), dtype=np.float32); rgb[:, y, x] = mids[anatomy["material"]].T
        glow = np.asarray(palette["emission_rgb"], dtype=np.float32)[:, None] / 255
        rgb[:, y, x] = rgb[:, y, x] * (1 - .55 * emission[y, x]) + glow * (.55 * emission[y, x])
        rgba = np.concatenate((rgb, occupancy[None]), axis=0)
        # RGBA is part of the living observation, not merely a display target:
        # identity palettes, fluid tint and emission color must survive the
        # continuous posterior instead of being guessed from class IDs.
        living = np.concatenate((occupancy[None], _one_hot(categorical["tissue"], TISSUE_COUNT), _one_hot(categorical["material"], MATERIAL_COUNT), _one_hot(categorical["part_owner"], PART_COUNT), emission[None], physiology_field, role_field, cell_state, rgba), axis=0)
        if living.shape != (74, 48, 48) or not np.isfinite(living).all() or float(living.min()) < 0 or float(living.max()) > 1:
            raise ValueError("Organism living-field contract failed.")
        return OrganismRasterSample(entry["sample_id"], int(entry["family_id"]), int(entry["subtype_id"]), int(entry["role_id"]), living, rgba, occupancy, categorical["tissue"], categorical["material"], categorical["part_owner"], emission, physiology_field, cell_state, _normalized_genes(entry))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        return {
            "sample_id": sample.sample_id, "family": torch.tensor(sample.family), "subtype": torch.tensor(sample.subtype), "role": torch.tensor(sample.role),
            "living_field": torch.from_numpy(sample.living_field), "rgba": torch.from_numpy(sample.rgba), "occupancy": torch.from_numpy(sample.occupancy),
            "tissue": torch.from_numpy(sample.tissue).long(), "material": torch.from_numpy(sample.material).long(), "part": torch.from_numpy(sample.part).long(),
            "emission": torch.from_numpy(sample.emission), "physiology": torch.from_numpy(sample.physiology), "cell_state": torch.from_numpy(sample.cell_state), "genes": torch.from_numpy(sample.genes),
        }
