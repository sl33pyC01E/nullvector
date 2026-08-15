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

from ..creature_stage_intervention_corpus import validate_intervention_corpus
from ..creature_stage_intervention_corpus.validation import (
    FRAMES,
    INTERVENTIONS,
    MAX_FLUIDS,
    POSITION_BIAS,
    POSITION_SCALE,
    UNIT_SCALE,
)
from ..creature_stage_neural_motion.contract import GENE_NAMES, ORGANS, TISSUES
from .contract import (
    CELL_POSITION_BOUND,
    CELL_STATE_FEATURES,
    DEFAULT_TEACHER,
    EVENT_FEATURES,
    FLUID_SCALAR_BOUND,
    FLUID_SLOTS,
    FLUID_STATE_FEATURES,
    MAX_CELLS,
    STATIC_FEATURES,
    SUMMARY_FEATURES,
)


SPLIT_MORPHOTYPES = {"train": (0, 1), "validation": (2,), "test": (3,)}


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


class NativeInterventionTeacher:
    """Strict padded tensor view over cellular damage, organ, and fluid trajectories."""

    def __init__(self, root: Path = DEFAULT_TEACHER) -> None:
        self.root = Path(root).resolve()
        self.validation = validate_intervention_corpus(self.root)
        self.manifest_path = self.root / "manifest.json"
        self.binary_path = self.root / "intervention_frames.u16le"
        self.manifest: dict[str, Any] = json.loads(self.manifest_path.read_bytes())
        self.binary = self.binary_path.read_bytes()
        if hashlib.sha256(self.binary).hexdigest() != self.validation["binary_sha256"]:
            raise ValueError("native intervention binary changed after validation")
        self.chassis = self.manifest["chassis"]
        self.clips = self.manifest["clips"]
        self.semantic_sha256 = hashlib.sha256(
            (
                self.validation["manifest_sha256"]
                + self.validation["binary_sha256"]
                + self.validation["corpus_identity_sha256"]
            ).encode("ascii")
        ).hexdigest()
        self._static: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._trajectory: OrderedDict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
        self._validate_split()

    def _validate_split(self) -> None:
        counts = {split: [0] * 5 for split in SPLIT_MORPHOTYPES}
        for chassis in self.chassis:
            matching = [name for name, values in SPLIT_MORPHOTYPES.items() if int(chassis["morphotype_id"]) in values]
            if len(matching) != 1:
                raise ValueError("native intervention split is not exhaustive")
            counts[matching[0]][int(chassis["family_id"])] += 1
        if counts != {"train": [2] * 5, "validation": [1] * 5, "test": [1] * 5}:
            raise ValueError("native intervention family split drifted")

    def split_chassis(self, split: str, family_id: int | None = None) -> list[int]:
        if split not in SPLIT_MORPHOTYPES:
            raise ValueError("unknown native intervention split")
        result = [
            int(chassis["chassis_id"])
            for chassis in self.chassis
            if int(chassis["morphotype_id"]) in SPLIT_MORPHOTYPES[split]
            and (family_id is None or int(chassis["family_id"]) == family_id)
        ]
        if not result:
            raise ValueError("native intervention split selection is empty")
        return result

    def _static_for(self, chassis_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if chassis_id in self._static:
            return self._static[chassis_id]
        record = self.chassis[chassis_id]
        cells = record["cells"]
        count = len(cells)
        if not 1 <= count <= MAX_CELLS:
            raise ValueError("native intervention chassis exceeds padded bound")
        features = np.zeros((MAX_CELLS, STATIC_FEATURES), dtype=np.float32)
        mask = np.zeros((MAX_CELLS,), dtype=np.bool_)
        grids = np.asarray([cell["grid"] for cell in cells], dtype=np.int16)
        mask[:count] = True
        for index, cell in enumerate(cells):
            x, y = float(cell["grid"][0]), float(cell["grid"][1])
            features[index, :4] = (x / 16.0, y / 16.0, math.hypot(x, y) / 24.0, y / 16.0)
            tissue, organ = str(cell["tissue"]), str(cell["organ"])
            if tissue not in TISSUES or organ not in ORGANS:
                raise ValueError("native intervention cell vocabulary drifted")
            features[index, 4 + TISSUES.index(tissue)] = 1.0
            features[index, 17 + ORGANS.index(organ)] = 1.0
            side = int(cell["side"])
            if side not in (-1, 0, 1):
                raise ValueError("native intervention side vocabulary drifted")
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
        small = np.max(np.abs(delta), axis=2) <= 1
        adjacency = np.zeros((MAX_CELLS, MAX_CELLS), dtype=np.bool_)
        adjacency[:count, :count] = small
        visited, frontier = {0}, [0]
        while frontier:
            current = frontier.pop()
            for neighbor in np.flatnonzero(small[current]).tolist():
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        if len(visited) != count:
            raise ValueError("native intervention cell graph is disconnected")
        for array in (features, mask, adjacency):
            array.setflags(write=False)
        self._static[chassis_id] = features, mask, adjacency
        return self._static[chassis_id]

    def _clip(self, clip_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if clip_id in self._trajectory:
            self._trajectory.move_to_end(clip_id)
            return self._trajectory[clip_id]
        record = self.clips[clip_id]
        start, length = int(record["byte_offset"]), int(record["byte_length"])
        raw = self.binary[start : start + length]
        if hashlib.sha256(raw).hexdigest() != record["trajectory_sha256"]:
            raise ValueError("native intervention clip changed after validation")
        count = int(record["cell_count"])
        stride = int(record["frame_stride_bytes"]) // 2
        frames = np.frombuffer(raw, dtype="<u2").reshape(FRAMES, stride)
        summary = np.zeros((FRAMES, SUMMARY_FEATURES), dtype=np.float32)
        summary[:, :9] = frames[:, :9].astype(np.float32) / float(UNIT_SCALE)
        summary[:, 9] = frames[:, 9].astype(np.float32) / float(FLUID_SLOTS)
        words = frames[:, 10 : 10 + count * 4].reshape(FRAMES, count, 4)
        cells = np.zeros((FRAMES, MAX_CELLS, CELL_STATE_FEATURES), dtype=np.float32)
        cells[:, :count, :2] = (
            (words[:, :, :2].astype(np.float32) - float(POSITION_BIAS)) / float(POSITION_SCALE) / CELL_POSITION_BOUND
        )
        cells[:, :count, 2] = words[:, :, 2].astype(np.float32) / float(UNIT_SCALE)
        cells[:, :count, 3] = (words[:, :, 3] == UNIT_SCALE).astype(np.float32)
        fluid_words = frames[:, 10 + count * 4 :].reshape(FRAMES, MAX_FLUIDS, 6)
        fluids = np.zeros((FRAMES, FLUID_SLOTS, FLUID_STATE_FEATURES), dtype=np.float32)
        counts = frames[:, 9].astype(np.int64)
        for frame, active in enumerate(counts.tolist()):
            if active:
                fluids[frame, :active, :4] = (
                    (fluid_words[frame, :active, :4].astype(np.float32) - float(POSITION_BIAS))
                    / float(POSITION_SCALE)
                    / CELL_POSITION_BOUND
                )
                fluids[frame, :active, 4:6] = (
                    fluid_words[frame, :active, 4:6].astype(np.float32)
                    / 1024.0
                    / FLUID_SCALAR_BOUND
                )
                fluids[frame, :active, 6] = 1.0
        for array in (summary, cells, fluids, counts):
            array.setflags(write=False)
        self._trajectory[clip_id] = summary, cells, fluids, counts
        if len(self._trajectory) > 12:
            self._trajectory.popitem(last=False)
        return self._trajectory[clip_id]

    def sample(self, chassis_id: int, intervention_id: int, frame: int) -> dict[str, Any]:
        if not 0 <= chassis_id < 20 or not 0 <= intervention_id < 9 or not 0 <= frame < FRAMES:
            raise ValueError("native intervention sample coordinate drifted")
        clip_id = chassis_id * 9 + intervention_id
        clip = self.clips[clip_id]
        if clip["chassis_id"] != chassis_id or clip["intervention_id"] != intervention_id:
            raise ValueError("native intervention clip registry drifted")
        summary, cells, fluids, _ = self._clip(clip_id)
        previous = max(frame - 1, 0)
        static, mask, adjacency = self._static_for(chassis_id)
        events = np.zeros((EVENT_FEATURES,), dtype=np.float32)
        events[0] = float(frame == 15)
        events[1] = float(intervention_id == 2 and frame == 75)
        events[2] = float(frame >= 15)
        events[3] = float(intervention_id == 2 and frame >= 75)
        events.setflags(write=False)
        chassis = self.chassis[chassis_id]
        return {
            "static": static,
            "mask": mask,
            "adjacency": adjacency,
            "cell_state": cells[previous],
            "summary_state": summary[previous],
            "fluid_state": fluids[previous],
            "cell_target": cells[frame],
            "summary_target": summary[frame],
            "fluid_target": fluids[frame],
            "family": int(chassis["family_id"]),
            "morphotype": int(chassis["family_id"]) * 4 + int(chassis["morphotype_id"]),
            "intervention": intervention_id,
            "phase": frame / float(FRAMES - 1),
            "events": events,
            "cell_count": int(chassis["cell_count"]),
            "fluid_count": int(round(float(summary[frame, 9]) * FLUID_SLOTS)),
        }


class PhysiologyBatchSampler:
    def __init__(self, teacher: NativeInterventionTeacher, *, batch_size: int = 5, seed: int = 0x50485953494F4C31) -> None:
        if type(batch_size) is not int or not 5 <= batch_size <= 25 or batch_size % 5:
            raise ValueError("neural physiology batch must be family balanced")
        self.teacher = teacher
        self.batch_size = batch_size
        self.seed = seed
        self.by_family = {family: teacher.split_chassis("train", family) for family in range(5)}

    def coordinates(self, step: int) -> list[tuple[int, int, int]]:
        if type(step) is not int or step < 0:
            raise ValueError("neural physiology sampler step drifted")
        result: list[tuple[int, int, int]] = []
        for slot in range(self.batch_size):
            family = slot % 5
            token = _mix64(self.seed ^ (step * 0xD1342543DE82EF95) ^ (slot * 0xA24BAED4963EE407))
            chassis = self.by_family[family][token % 2]
            intervention = _mix64(token ^ 0xC6BC279692B5CC83) % 9
            frame = _mix64(token ^ 0xDB4F0B9175AE2165) % FRAMES
            result.append((int(chassis), int(intervention), int(frame)))
        return result

    def batch(self, step: int, device: str | torch.device = "cpu") -> dict[str, Tensor]:
        rows = [self.teacher.sample(*coordinate) for coordinate in self.coordinates(step)]
        names = (
            "static", "mask", "adjacency", "cell_state", "summary_state",
            "fluid_state", "cell_target", "summary_target", "fluid_target", "events",
        )
        result = {
            name: torch.from_numpy(np.stack([row[name] for row in rows]).copy()).to(device)
            for name in names
        }
        for name in ("family", "morphotype", "intervention"):
            result[name] = torch.tensor([int(row[name]) for row in rows], dtype=torch.long, device=device)
        result["phase"] = torch.tensor([float(row["phase"]) for row in rows], dtype=torch.float32, device=device)
        return result
