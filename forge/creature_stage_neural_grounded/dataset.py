from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..creature_stage_developmental import TISSUES as DEVELOPMENTAL_TISSUES
from ..creature_stage_developmental import TRAITS
from ..creature_stage_developmental.genomes import review_genomes
from ..creature_stage_developmental.development import develop
from ..creature_stage_grounded_locomotion.contract import LOCOMOTOR_MODES, source_sha256 as grounded_source_sha256
from ..creature_stage_grounded_locomotion.review import validate_review
from ..creature_stage_neural_motion.contract import CONTROL_FEATURES, GENE_NAMES, ORGANS, STATIC_FEATURES, TISSUES
from ..creature_stage_developmental_motion.compiler import ORGAN_MAP, TISSUE_MAP
from .contract import BODY_SPEED_SCALE, DYNAMIC_FEATURES, GROUND_AUTHORITY, MAX_CELLS, POSITION_SCALE, sha256_file


EXPECTED_ARRAYS = frozenset({
    "family", "grafted", "ground_y", "body_world_x", "body_velocity_x",
    "nodes_local", "node_velocity", "node_mask", "cells_local", "cell_mask",
    "rest_cells", "tissue", "appendage_owner", "trait_fields", "component_weights",
    "component_mask", "contact_active", "contact_anchor_local", "contact_force",
    "appendage_mask", "locomotor_mode", "muscle_activation", "muscle_mask",
})


def _array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256(b"nullvector-grounded-teacher-arrays-v1\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0" + value.dtype.str.encode("ascii") + b"\0")
        digest.update(str(value.shape).encode("ascii") + b"\0")
        digest.update(memoryview(value))
    return digest.hexdigest()


class GroundedMotionTeacher:
    """Strict tensor projection of the visually approved contact/PBD authority."""

    def __init__(self, root: Path = GROUND_AUTHORITY) -> None:
        self.root = Path(root).resolve()
        validation = validate_review(self.root, replay=False)
        if not validation["passed"] or not validation["promotion_ready"]:
            raise ValueError("grounded motion authority is not promoted")
        self.manifest_path = self.root / "grounded_locomotion_manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text("utf-8"))
        artifact = self.manifest["artifacts"]["arrays"]
        self.archive_path = self.root / artifact["path"]
        if (
            sha256_file(self.archive_path) != artifact["sha256"]
            or self.archive_path.stat().st_size != artifact["bytes"]
        ):
            raise ValueError("grounded motion authority archive drifted")
        with np.load(self.archive_path, allow_pickle=False) as archive:
            if set(archive.files) != EXPECTED_ARRAYS:
                raise ValueError("grounded motion authority member census drifted")
            self.arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        if self.arrays["cells_local"].shape != (10, 72, MAX_CELLS, 2):
            raise ValueError("grounded motion teacher cell shape drifted")
        self.organisms = tuple(develop(genome) for genome in review_genomes())
        if [item.cell_count for item in self.organisms] != self.arrays["cell_mask"].sum(1).tolist():
            raise ValueError("grounded organism census drifted")
        self.static = tuple(self._static(index) for index in range(10))
        self.semantic_sha256 = hashlib.sha256(
            (
                self.manifest["semantic_sha256"]
                + sha256_file(self.archive_path)
                + _array_digest(self.arrays)
                + grounded_source_sha256()
            ).encode("ascii")
        ).hexdigest()

    @staticmethod
    def split_indices(split: str) -> tuple[int, ...]:
        if split == "train":
            return (0, 2, 4, 6, 8)
        if split == "validation":
            return (1, 3, 5, 7, 9)
        if split == "all":
            return tuple(range(10))
        raise ValueError("grounded teacher split drifted")

    def _static(self, identity: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        organism = self.organisms[identity]
        count = organism.cell_count
        features = np.zeros((MAX_CELLS, STATIC_FEATURES), dtype=np.float32)
        mask = self.arrays["cell_mask"][identity].astype(np.bool_)
        adjacency = np.zeros((MAX_CELLS, MAX_CELLS), dtype=np.bool_)
        coordinates = self.arrays["rest_cells"][identity, :count]
        dominant_component = np.argmax(organism.component_weights, axis=1)
        span = np.ptp(coordinates, axis=0)
        ti = {name: TRAITS.index(name) for name in TRAITS}
        genes = {
            "width": min(1.0, float(span[0]) / 48.0),
            "height": min(1.0, float(span[1]) / 48.0),
            "asymmetry": 1.0 - float(organism.genome.traits[ti["symmetry"]]),
            "symmetry": float(organism.genome.traits[ti["symmetry"]]),
            "repair": float(organism.genome.traits[ti["regeneration"]]),
            "metabolism": float(organism.genome.traits[ti["metabolism"]]),
            "fertility": float(np.clip(.25 + organism.genome.traits[ti["metabolism"]] * .35, 0, 1)),
            "bond_strength": float(np.clip(organism.genome.traits[ti["stiffness"]] * .55 + organism.genome.traits[ti["bone_density"]] * .45, 0, 1)),
        }
        for index in range(count):
            x, y = map(float, coordinates[index])
            features[index, 0:4] = (x / 16, y / 16, math.hypot(x, y) / 24, y / 16)
            source_tissue = DEVELOPMENTAL_TISSUES[int(organism.tissue[index])]
            target_tissue = TISSUE_MAP[source_tissue]
            features[index, 4 + TISSUES.index(target_tissue)] = 1
            component = organism.genome.components[int(dominant_component[index])]
            organ = ORGAN_MAP.get(component.organ, component.organ if component.organ in ORGANS else "none")
            features[index, 17 + ORGANS.index(organ)] = 1
            side = int(organism.side[index])
            features[index, 47 + side] = 1
            features[index, 49] = max(0.0, 1.0 - math.hypot(x, y) / 28)
            owner = int(organism.appendage_index[index])
            if owner >= 0:
                features[index, 50:53] = (1, math.sin(owner * math.tau / 32), math.cos(owner * math.tau / 32))
            for gene_index, name in enumerate(GENE_NAMES):
                features[index, 53 + gene_index] = genes[name]
        grids = coordinates.astype(np.int16)
        delta = grids[:, None] - grids[None]
        local = np.max(np.abs(delta), axis=2) <= 1
        adjacency[:count, :count] = local
        seen, frontier = {0}, [0]
        while frontier:
            current = frontier.pop()
            for neighbor in np.flatnonzero(local[current]).tolist():
                if neighbor not in seen:
                    seen.add(neighbor); frontier.append(neighbor)
        if len(seen) != count:
            raise ValueError("grounded teacher cell graph disconnected")
        return features, mask, adjacency

    def sample(self, identity: int, frame: int) -> dict[str, np.ndarray | int | float]:
        if not 0 <= identity < 10 or not 0 <= frame < 72:
            raise ValueError("grounded teacher coordinate drifted")
        previous = (frame - 1) % 72
        before = (frame - 2) % 72
        count = int(self.arrays["cell_mask"][identity].sum())
        rest = self.arrays["rest_cells"][identity]
        current_cells = self.arrays["cells_local"][identity, previous]
        before_cells = self.arrays["cells_local"][identity, before]
        target_cells = self.arrays["cells_local"][identity, frame]
        state = np.zeros((MAX_CELLS, 4), np.float32)
        target = np.zeros_like(state)
        state[:count, :2] = (current_cells[:count] - rest[:count]) / POSITION_SCALE
        state[:count, 2:] = (current_cells[:count] - before_cells[:count]) / POSITION_SCALE
        target[:count, :2] = (target_cells[:count] - rest[:count]) / POSITION_SCALE
        target[:count, 2:] = (target_cells[:count] - current_cells[:count]) / POSITION_SCALE
        dynamic = np.zeros((MAX_CELLS, DYNAMIC_FEATURES), np.float32)
        owners = self.arrays["appendage_owner"][identity, :count]
        modes = self.arrays["locomotor_mode"][identity]
        active = self.arrays["contact_active"][identity, frame].astype(bool)
        for cell_index, owner in enumerate(owners):
            mode = int(modes[owner]) if owner >= 0 else 0
            dynamic[cell_index, mode] = 1.0
            if owner >= 0 and active[owner]:
                dynamic[cell_index, 5] = 1.0
                dynamic[cell_index, 6:8] = np.clip(
                    (self.arrays["contact_anchor_local"][identity, frame, owner] - target_cells[cell_index]) / POSITION_SCALE,
                    -1, 1,
                )
                dynamic[cell_index, 8:10] = np.clip(self.arrays["contact_force"][identity, frame, owner], -1, 1)
        dynamic[:count, 10] = self.arrays["body_velocity_x"][identity, previous] / BODY_SPEED_SCALE
        dynamic[:count, 11] = np.clip((self.arrays["body_world_x"][identity, frame] - self.arrays["body_world_x"][identity, 0]) / 16, -1, 1)
        dynamic[:count, 12] = np.clip((self.arrays["ground_y"][identity] - target_cells[:count, 1]) / 32, -1, 1)
        dynamic[:count, 13] = float(self.arrays["grafted"][identity])
        phase = frame / 72.0
        dynamic[:count, 14] = math.sin(math.tau * phase)
        dynamic[:count, 15] = math.cos(math.tau * phase)
        static, mask, adjacency = self.static[identity]
        controls = np.zeros(CONTROL_FEATURES, np.float32); controls[:4] = (1, 0, 1, 0)
        for value in (state, target, dynamic, controls):
            value.setflags(write=False)
        return {
            "static": static, "state": state, "target": target, "dynamic": dynamic,
            "mask": mask, "adjacency": adjacency, "controls": controls,
            "family": int(self.arrays["family"][identity]),
            "morphotype": int(self.arrays["family"][identity]) * 4 + int(self.arrays["grafted"][identity]),
            "motion": 2, "phase": phase, "identity": identity, "frame": frame,
            "body_target": float(self.arrays["body_velocity_x"][identity, frame] / BODY_SPEED_SCALE),
            "body_previous": float(self.arrays["body_velocity_x"][identity, previous] / BODY_SPEED_SCALE),
            "cell_count": count,
        }

    @staticmethod
    def _mix64(value: int) -> int:
        value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return value ^ (value >> 31)

    def batch(self, step: int, batch_size: int, device: torch.device, *, split: str = "train", frame_offset: int = 0) -> dict[str, Tensor]:
        identities = self.split_indices(split)
        if batch_size % len(identities):
            raise ValueError("grounded batch must remain family-balanced")
        rows: list[dict[str, Any]] = []
        for slot in range(batch_size):
            family_slot = slot % len(identities)
            identity = identities[family_slot]
            token = self._mix64(0x47524F554E444544 ^ step * 0xD1342543DE82EF95 ^ slot * 0xA24BAED4963EE407)
            frame = int((token % 72 + frame_offset) % 72)
            rows.append(self.sample(identity, frame))
        result: dict[str, Tensor] = {}
        for name in ("static", "state", "target", "dynamic", "mask", "adjacency", "controls"):
            result[name] = torch.from_numpy(np.stack([row[name] for row in rows]).copy()).to(device)
        for name in ("family", "morphotype", "motion", "identity", "frame"):
            result[name] = torch.tensor([int(row[name]) for row in rows], dtype=torch.long, device=device)
        for name in ("phase", "body_target", "body_previous"):
            result[name] = torch.tensor([float(row[name]) for row in rows], dtype=torch.float32, device=device)
        return result

    def sequence(self, step: int, batch_size: int, frames: int, device: torch.device, *, split: str = "all") -> list[dict[str, Tensor]]:
        if not 2 <= frames <= 12:
            raise ValueError("grounded teacher sequence length drifted")
        return [self.batch(step, batch_size, device, split=split, frame_offset=offset) for offset in range(frames)]
