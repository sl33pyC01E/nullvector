from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..creature_stage_motion_corpus import validate_motion_corpus
from .contract import (
    CONTROL_FEATURES,
    DEFAULT_TEACHER,
    GENE_NAMES,
    MAX_CELLS,
    MAX_DISPLACEMENT,
    ORGANS,
    STATIC_FEATURES,
    TISSUES,
)


SPLIT_MORPHOTYPES = {"train": (0, 1), "validation": (2,), "test": (3,)}
EVENTS = ("none", "impact", "terminal")


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeMotionTeacher:
    """Strict, padded tensor view over the native cell-motion authority."""

    def __init__(self, root: Path = DEFAULT_TEACHER) -> None:
        self.root = Path(root).resolve()
        self.validation = validate_motion_corpus(self.root)
        self.manifest_path = self.root / "manifest.json"
        self.binary_path = self.root / "motion_frames.u16le"
        self.manifest: dict[str, Any] = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.binary = self.binary_path.read_bytes()
        if _sha256(self.manifest_path) != self.validation["manifest_sha256"]:
            raise ValueError("native motion teacher manifest changed after validation")
        if hashlib.sha256(self.binary).hexdigest() != self.validation["binary_sha256"]:
            raise ValueError("native motion teacher binary changed after validation")
        self.chassis = self.manifest["chassis"]
        self.clips = self.manifest["clips"]
        self._static: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._trajectory: OrderedDict[int, np.ndarray] = OrderedDict()
        self.semantic_sha256 = hashlib.sha256(
            (
                self.validation["manifest_sha256"]
                + self.validation["binary_sha256"]
                + self.validation["corpus_identity_sha256"]
            ).encode("ascii")
        ).hexdigest()
        self._validate_split()

    def _validate_split(self) -> None:
        seen = {name: [0] * 5 for name in SPLIT_MORPHOTYPES}
        for chassis in self.chassis:
            family = int(chassis["family_id"])
            morphotype = int(chassis["morphotype_id"])
            matching = [name for name, values in SPLIT_MORPHOTYPES.items() if morphotype in values]
            if len(matching) != 1:
                raise ValueError("native motion split is not exhaustive and disjoint")
            seen[matching[0]][family] += 1
        if seen != {
            "train": [2, 2, 2, 2, 2],
            "validation": [1, 1, 1, 1, 1],
            "test": [1, 1, 1, 1, 1],
        }:
            raise ValueError("native motion family-balanced split drifted")

    def split_chassis(self, split: str, family_id: int | None = None) -> list[int]:
        if split not in SPLIT_MORPHOTYPES:
            raise ValueError("unknown native motion split")
        result = [
            int(record["chassis_id"])
            for record in self.chassis
            if int(record["morphotype_id"]) in SPLIT_MORPHOTYPES[split]
            and (family_id is None or int(record["family_id"]) == family_id)
        ]
        if not result:
            raise ValueError("native motion split selection is empty")
        return result

    def _static_for(self, chassis_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if chassis_id in self._static:
            return self._static[chassis_id]
        record = self.chassis[chassis_id]
        cells = record["cells"]
        count = len(cells)
        if not 1 <= count <= MAX_CELLS:
            raise ValueError("native motion chassis exceeds padded cell bound")
        features = np.zeros((MAX_CELLS, STATIC_FEATURES), dtype=np.float32)
        mask = np.zeros((MAX_CELLS,), dtype=np.bool_)
        grids = np.asarray([cell["grid"] for cell in cells], dtype=np.int16)
        mask[:count] = True
        for index, cell in enumerate(cells):
            x, y = (float(cell["grid"][0]), float(cell["grid"][1]))
            features[index, 0:4] = (x / 16.0, y / 16.0, math.hypot(x, y) / 24.0, y / 16.0)
            tissue = str(cell["tissue"])
            organ = str(cell["organ"])
            if tissue not in TISSUES or organ not in ORGANS:
                raise ValueError("native motion cell vocabulary drifted")
            features[index, 4 + TISSUES.index(tissue)] = 1.0
            features[index, 17 + ORGANS.index(organ)] = 1.0
            side = int(cell["side"])
            if side not in (-1, 0, 1):
                raise ValueError("native motion side vocabulary drifted")
            features[index, 46 + side + 1] = 1.0
            features[index, 49] = max(0.0, 1.0 - math.hypot(x, y) / 24.0)
            appendage = int(cell["appendage"])
            if appendage >= 0:
                features[index, 50] = 1.0
                features[index, 51] = math.sin(appendage * math.tau / 32.0)
                features[index, 52] = math.cos(appendage * math.tau / 32.0)
            for gene_index, name in enumerate(GENE_NAMES):
                features[index, 53 + gene_index] = float(record["genes"][name])
        delta = grids[:, None, :] - grids[None, :, :]
        adjacency_small = np.max(np.abs(delta), axis=2) <= 1
        adjacency = np.zeros((MAX_CELLS, MAX_CELLS), dtype=np.bool_)
        adjacency[:count, :count] = adjacency_small
        visited = {0}
        frontier = [0]
        while frontier:
            current = frontier.pop()
            for neighbor in np.flatnonzero(adjacency_small[current]).tolist():
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        if len(visited) != count:
            raise ValueError("native motion cell graph is disconnected")
        features.setflags(write=False)
        mask.setflags(write=False)
        adjacency.setflags(write=False)
        self._static[chassis_id] = (features, mask, adjacency)
        return self._static[chassis_id]

    def _clip(self, clip_id: int) -> np.ndarray:
        if clip_id in self._trajectory:
            self._trajectory.move_to_end(clip_id)
            return self._trajectory[clip_id]
        record = self.clips[clip_id]
        offset = int(record["byte_offset"])
        length = int(record["byte_length"])
        count = int(record["cell_count"])
        raw = self.binary[offset : offset + length]
        if hashlib.sha256(raw).hexdigest() != record["trajectory_sha256"]:
            raise ValueError("native motion clip changed after validation")
        values = np.frombuffer(raw, dtype="<u2").reshape(72, count, 2)
        trajectory = (values.astype(np.float32) - 32768.0) / 256.0
        trajectory.setflags(write=False)
        self._trajectory[clip_id] = trajectory
        if len(self._trajectory) > 24:
            self._trajectory.popitem(last=False)
        return trajectory

    def sample(self, chassis_id: int, motion_id: int, frame: int) -> dict[str, np.ndarray | int | float]:
        if not 0 <= chassis_id < 20 or not 0 <= motion_id < 13 or not 0 <= frame < 72:
            raise ValueError("native motion sample coordinate drifted")
        chassis = self.chassis[chassis_id]
        clip_id = chassis_id * 13 + motion_id
        clip = self.clips[clip_id]
        if clip["chassis_id"] != chassis_id or clip["motion_id"] != motion_id:
            raise ValueError("native motion clip registry drifted")
        trajectory = self._clip(clip_id)
        previous_index = max(frame - 1, 0)
        previous2_index = max(frame - 2, 0)
        previous_delta = trajectory[previous_index]
        previous_velocity = previous_delta - trajectory[previous2_index]
        target_delta = trajectory[frame]
        target_velocity = target_delta - previous_delta
        static, mask, adjacency = self._static_for(chassis_id)
        state = np.zeros((MAX_CELLS, 4), dtype=np.float32)
        target = np.zeros((MAX_CELLS, 4), dtype=np.float32)
        count = int(chassis["cell_count"])
        state[:count, :2] = previous_delta / MAX_DISPLACEMENT
        state[:count, 2:] = previous_velocity / MAX_DISPLACEMENT
        target[:count, :2] = target_delta / MAX_DISPLACEMENT
        target[:count, 2:] = target_velocity / MAX_DISPLACEMENT
        controls = np.zeros((CONTROL_FEATURES,), dtype=np.float32)
        controls[0:2] = np.asarray(clip["controls"]["move"], dtype=np.float32)
        controls[2:4] = np.asarray(clip["controls"]["aim"], dtype=np.float32)
        controls[4] = float(clip["controls"]["attack"])
        controls[5] = float(clip["controls"]["utility"])
        controls[6 + EVENTS.index(str(clip["controls"]["external_event"]))] = 1.0
        spec = self.manifest["motion_specs"][clip["motion"]]
        elapsed = frame / float(self.manifest["fixed_hz"])
        phase = elapsed / float(spec["cycle"])
        phase = phase % 1.0 if bool(spec["loop"]) else min(phase, 1.0)
        for array in (state, target, controls):
            array.setflags(write=False)
        return {
            "static": static,
            "state": state,
            "target": target,
            "mask": mask,
            "adjacency": adjacency,
            "controls": controls,
            "family": int(chassis["family_id"]),
            "morphotype": int(chassis["family_id"]) * 4 + int(chassis["morphotype_id"]),
            "motion": motion_id,
            "phase": float(phase),
            "cell_count": count,
        }


class MotionBatchSampler:
    def __init__(self, teacher: NativeMotionTeacher, *, batch_size: int = 10, seed: int = 0x4E4D4F54494F4E31) -> None:
        if type(batch_size) is not int or not 5 <= batch_size <= 40 or batch_size % 5:
            raise ValueError("neural motion batch must be family balanced")
        self.teacher = teacher
        self.batch_size = batch_size
        self.seed = seed
        self.by_family = {
            family: teacher.split_chassis("train", family)
            for family in range(5)
        }
        if [len(self.by_family[family]) for family in range(5)] != [2] * 5:
            raise ValueError("neural motion training family split drifted")

    def coordinates(self, step: int) -> list[tuple[int, int, int]]:
        if type(step) is not int or step < 0:
            raise ValueError("neural motion sampler step drifted")
        result: list[tuple[int, int, int]] = []
        for slot in range(self.batch_size):
            family = slot % 5
            token = _mix64(self.seed ^ (step * 0xD1342543DE82EF95) ^ (slot * 0xA24BAED4963EE407))
            chassis = self.by_family[family][token % 2]
            motion = _mix64(token ^ 0xC6BC279692B5CC83) % 13
            frame = _mix64(token ^ 0xDB4F0B9175AE2165) % 72
            result.append((int(chassis), int(motion), int(frame)))
        return result

    def batch(self, step: int, device: str | torch.device = "cpu") -> dict[str, Tensor]:
        rows = [self.teacher.sample(*coordinate) for coordinate in self.coordinates(step)]
        float_names = ("static", "state", "target", "controls")
        result: dict[str, Tensor] = {
            name: torch.from_numpy(np.stack([row[name] for row in rows]).copy()).to(device)
            for name in float_names
        }
        result["mask"] = torch.from_numpy(np.stack([row["mask"] for row in rows]).copy()).to(device)
        result["adjacency"] = torch.from_numpy(np.stack([row["adjacency"] for row in rows]).copy()).to(device)
        for name in ("family", "morphotype", "motion"):
            result[name] = torch.tensor([int(row[name]) for row in rows], dtype=torch.long, device=device)
        result["phase"] = torch.tensor([float(row["phase"]) for row in rows], dtype=torch.float32, device=device)
        return result
