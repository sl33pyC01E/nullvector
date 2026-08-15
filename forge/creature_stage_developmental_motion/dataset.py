from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..config import PROJECT_ROOT
from ..multifield_style_motion.hashing import canonical_json_bytes, sha256_file
from .compiler import ARRAY_FILE, MANIFEST_FILE, validate_candidate_corpus
from .contract import (
    DEFAULT_CORPUS,
    DEFAULT_PRIOR,
    FRAME_COUNT,
    MAX_DISPLACEMENT,
    MAX_MUSCLES,
)


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


@dataclass(frozen=True, slots=True)
class SequenceCoordinate:
    specimen: int
    start: int
    forced_seam: bool


class DevelopmentalMotionTeacher:
    """Strict tensor view over the approved v7 cell/skeleton/muscle limit cycles."""

    def __init__(
        self,
        root: Path = DEFAULT_CORPUS,
        *,
        prior: Path | None = None,
        replay: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.validation = validate_candidate_corpus(self.root, replay=replay)
        manifest_path = self.root / MANIFEST_FILE
        archive_path = self.root / ARRAY_FILE
        raw = manifest_path.read_bytes()
        self.manifest: dict[str, Any] = json.loads(raw)
        if raw != canonical_json_bytes(self.manifest):
            raise ValueError("developmental motion teacher manifest is not canonical")
        if not self.manifest["approval"]["training_permitted"]:
            raise ValueError("developmental motion teacher is not approved for training")
        with np.load(archive_path, allow_pickle=False) as archive:
            self.arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        for value in self.arrays.values():
            value.setflags(write=False)
        self.semantic_sha256 = hashlib.sha256(
            b"nullvector-developmental-motion-teacher-v2\0"
            + self.manifest["semantic_sha256"].encode("ascii")
            + sha256_file(manifest_path).encode("ascii")
            + sha256_file(archive_path).encode("ascii")
        ).hexdigest()
        self.specimen_count = int(self.manifest["contract"]["specimen_count"])
        if self.specimen_count != 10:
            raise ValueError("developmental motion teacher specimen count drifted")
        self._validate_arrays()
        self.parent_prior: np.ndarray | None = None
        self.prior_semantic_sha256: str | None = None
        self.prior_root: Path | None = None
        if prior is not None:
            from .parent_prior import ARRAY_FILE as PRIOR_ARRAY_FILE
            from .parent_prior import validate_parent_prior

            self.prior_root = Path(prior).resolve()
            validation = validate_parent_prior(self.prior_root)
            with np.load(self.prior_root / PRIOR_ARRAY_FILE, allow_pickle=False) as archive:
                self.parent_prior = np.ascontiguousarray(archive["parent_prior"])
            self.parent_prior.setflags(write=False)
            self.prior_semantic_sha256 = validation["semantic_sha256"]

    def _validate_arrays(self) -> None:
        required = {
            "static", "mask", "adjacency", "rest_xy", "trajectory",
            "muscle_features", "muscle_mask", "muscle_activation", "muscle_incidence",
            "node_mask", "node_features", "node_adjacency", "node_rest", "node_trajectory",
            "cell_node_weights", "appendage_mask", "planted_contacts", "family",
            "morphotype", "family_mix", "traits", "cell_count", "node_count", "muscle_count",
        }
        if set(self.arrays) != required:
            raise ValueError("developmental motion teacher array registry drifted")
        if not np.array_equal(self.arrays["family"], np.arange(10, dtype=np.uint8) // 2):
            raise ValueError("developmental motion family balance drifted")
        if not np.array_equal(self.arrays["morphotype"], (np.arange(10, dtype=np.uint8) // 2) * 4 + np.arange(10, dtype=np.uint8) % 2):
            raise ValueError("developmental motion morphotype registry drifted")
        weights = self.arrays["cell_node_weights"]
        mask = self.arrays["mask"]
        active_sums = weights.sum(axis=2)[mask]
        if active_sums.size == 0 or not np.allclose(active_sums, 1.0, atol=1e-6):
            raise ValueError("developmental motion skinning authority drifted")
        if np.any(weights[~mask] != 0.0):
            raise ValueError("developmental motion padded skinning weights drifted")

    @property
    def base_indices(self) -> tuple[int, ...]:
        return (0, 2, 4, 6, 8)

    @property
    def graft_indices(self) -> tuple[int, ...]:
        return (1, 3, 5, 7, 9)

    def sample(self, specimen: int, frame: int) -> dict[str, np.ndarray | int | float]:
        if not 0 <= specimen < self.specimen_count or not 0 <= frame < FRAME_COUNT:
            raise ValueError("developmental motion sample coordinate drifted")
        previous = (frame - 1) % FRAME_COUNT
        previous2 = (frame - 2) % FRAME_COUNT
        trajectory = self.arrays["trajectory"][specimen]
        node_trajectory = self.arrays["node_trajectory"][specimen]
        muscle = self.arrays["muscle_activation"][specimen]

        state = np.zeros_like(trajectory[0], dtype=np.float32)
        state = np.concatenate(
            (trajectory[previous] / MAX_DISPLACEMENT,
             (trajectory[previous] - trajectory[previous2]) / MAX_DISPLACEMENT),
            axis=1,
        ).astype(np.float32)
        target = np.concatenate(
            (trajectory[frame] / MAX_DISPLACEMENT,
             (trajectory[frame] - trajectory[previous]) / MAX_DISPLACEMENT),
            axis=1,
        ).astype(np.float32)
        node_state = np.concatenate(
            (node_trajectory[previous] / MAX_DISPLACEMENT,
             (node_trajectory[previous] - node_trajectory[previous2]) / MAX_DISPLACEMENT),
            axis=1,
        ).astype(np.float32)
        node_target = np.concatenate(
            (node_trajectory[frame] / MAX_DISPLACEMENT,
             (node_trajectory[frame] - node_trajectory[previous]) / MAX_DISPLACEMENT),
            axis=1,
        ).astype(np.float32)
        result: dict[str, np.ndarray | int | float] = {
            name: self.arrays[name][specimen]
            for name in (
                "static", "mask", "adjacency", "rest_xy", "muscle_features",
                "muscle_mask", "muscle_incidence", "node_mask", "node_features",
                "node_adjacency", "node_rest", "cell_node_weights", "appendage_mask",
                "traits",
            )
        }
        result.update({
            "state": state,
            "target": target,
            "node_state": node_state,
            "node_target": node_target,
            "muscle_state": muscle[previous].astype(np.float32),
            "muscle_target": muscle[frame].astype(np.float32),
            "planted_target": self.arrays["planted_contacts"][specimen, frame],
            "family": int(self.arrays["family"][specimen]),
            "morphotype": int(self.arrays["morphotype"][specimen]),
            "phase": frame / FRAME_COUNT,
            "specimen": specimen,
            "frame": frame,
        })
        if self.parent_prior is not None:
            result["parent_prior"] = self.parent_prior[specimen, frame]
        for value in result.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        return result


class DevelopmentalSequenceSampler:
    """Deterministic family-balanced cyclic sequences with an explicit seam quota."""

    def __init__(
        self,
        teacher: DevelopmentalMotionTeacher,
        *,
        batch_size: int = 10,
        sequence_frames: int = 12,
        seed: int = 0x4445564143545632,
        seam_numerator: int = 1,
        seam_denominator: int = 3,
    ) -> None:
        if type(batch_size) is not int or not 5 <= batch_size <= 40 or batch_size % 5:
            raise ValueError("developmental sequence batch must be family balanced")
        if type(sequence_frames) is not int or not 6 <= sequence_frames <= 24:
            raise ValueError("developmental sequence length drifted")
        if not 1 <= seam_numerator < seam_denominator <= 16:
            raise ValueError("developmental sequence seam quota drifted")
        self.teacher = teacher
        self.batch_size = batch_size
        self.sequence_frames = sequence_frames
        self.seed = seed
        self.seam_numerator = seam_numerator
        self.seam_denominator = seam_denominator

    def coordinates(self, step: int) -> tuple[SequenceCoordinate, ...]:
        if type(step) is not int or step < 0:
            raise ValueError("developmental sequence step drifted")
        forced = step % self.seam_denominator < self.seam_numerator
        rows: list[SequenceCoordinate] = []
        for slot in range(self.batch_size):
            family = slot % 5
            token = _mix64(self.seed ^ (step * 0xD1342543DE82EF95) ^ (slot * 0xA24BAED4963EE407))
            specimen = family * 2 + int(token & 1)
            if forced:
                start = FRAME_COUNT - 1 - int(token % max(1, self.sequence_frames // 2))
            else:
                start = int(_mix64(token ^ 0xDB4F0B9175AE2165) % FRAME_COUNT)
            rows.append(SequenceCoordinate(specimen, start, forced))
        return tuple(rows)

    @staticmethod
    def _stack(rows: list[dict[str, np.ndarray | int | float]], device: str | torch.device) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        array_names = (
            "static", "mask", "adjacency", "rest_xy", "state", "target",
            "muscle_features", "muscle_mask", "muscle_incidence", "muscle_state",
            "muscle_target", "node_mask", "node_features", "node_adjacency", "node_rest",
            "node_state", "node_target", "cell_node_weights", "appendage_mask",
            "planted_target", "traits",
        )
        if "parent_prior" in rows[0]:
            array_names = (*array_names, "parent_prior")
        for name in array_names:
            result[name] = torch.from_numpy(np.stack([np.asarray(row[name]) for row in rows]).copy()).to(device)
        for name in ("family", "morphotype", "specimen", "frame"):
            result[name] = torch.tensor([int(row[name]) for row in rows], dtype=torch.long, device=device)
        result["phase"] = torch.tensor([float(row["phase"]) for row in rows], dtype=torch.float32, device=device)
        return result

    def sequence(self, step: int, device: str | torch.device = "cpu") -> tuple[list[dict[str, Tensor]], tuple[SequenceCoordinate, ...]]:
        coordinates = self.coordinates(step)
        frames: list[dict[str, Tensor]] = []
        for offset in range(self.sequence_frames):
            rows = [self.teacher.sample(item.specimen, (item.start + offset) % FRAME_COUNT) for item in coordinates]
            frames.append(self._stack(rows, device))
        return frames, coordinates


def project_path(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("developmental motion path escaped the project") from error
