from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..creature_stage_neural_motion.dataset import NativeMotionTeacher
from .contract import LoopTrainingConfig


SEED = 0x4C4F4F5044454C31


def _mix64(value: int) -> int:
    value &= (1 << 64) - 1
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


@dataclass(frozen=True)
class SequenceCoordinate:
    chassis: int
    motion: int
    start: int
    loop: bool
    forced_seam: bool


class LoopAwareRolloutBatchSampler:
    """Family-balanced sequences with an explicit quota of cyclic seam crossings."""

    def __init__(
        self,
        teacher: NativeMotionTeacher,
        *,
        batch_size: int = 5,
        config: LoopTrainingConfig | None = None,
        seed: int = SEED,
    ) -> None:
        if type(batch_size) is not int or not 5 <= batch_size <= 30 or batch_size % 5:
            raise ValueError("loop motion batch must be family balanced")
        self.teacher = teacher
        self.batch_size = batch_size
        self.config = config or LoopTrainingConfig()
        self.seed = seed
        self.by_family = {family: teacher.split_chassis("train", family) for family in range(5)}
        if [len(self.by_family[family]) for family in range(5)] != [2] * 5:
            raise ValueError("loop motion training split drifted")
        self.motion_names = tuple(teacher.manifest["motion_specs"])
        if len(self.motion_names) != 13:
            raise ValueError("loop motion vocabulary drifted")
        self.loop_by_motion = tuple(
            bool(teacher.manifest["motion_specs"][name]["loop"])
            for name in self.motion_names
        )
        if not any(self.loop_by_motion) or all(self.loop_by_motion):
            raise ValueError("loop motion cyclic/noncyclic coverage drifted")
        self.loop_motion_ids = tuple(index for index, loop in enumerate(self.loop_by_motion) if loop)

    def coordinates(self, update: int, *, force_seam: bool = False) -> list[SequenceCoordinate]:
        if type(update) is not int or update < 0:
            raise ValueError("loop motion sampler update drifted")
        result: list[SequenceCoordinate] = []
        for slot in range(self.batch_size):
            family = slot % 5
            token = _mix64(self.seed ^ (update * 0xD1342543DE82EF95) ^ (slot * 0xA24BAED4963EE407))
            chassis = self.by_family[family][token % 2]
            if force_seam:
                motion = self.loop_motion_ids[
                    int(_mix64(token ^ 0xC6BC279692B5CC83) % len(self.loop_motion_ids))
                ]
            else:
                motion = int(_mix64(token ^ 0xC6BC279692B5CC83) % len(self.motion_names))
            loop = self.loop_by_motion[motion]
            seam_selected = loop and (
                force_seam
                or int(_mix64(token ^ 0x45414D5345414D31) % self.config.seam_quota_denominator)
                < self.config.seam_quota_numerator
            )
            if seam_selected:
                # Every selected sequence crosses frame 71 -> 0 at least once.
                start = 72 - self.config.sequence_frames + 1 + int(
                    _mix64(token ^ 0x5345414D53544152) % (self.config.sequence_frames - 1)
                )
            elif loop:
                start = int(_mix64(token ^ 0xDB4F0B9175AE2165) % 72)
            else:
                start = int(
                    _mix64(token ^ 0xDB4F0B9175AE2165)
                    % (72 - self.config.sequence_frames + 1)
                )
            result.append(SequenceCoordinate(int(chassis), motion, start, loop, seam_selected))
        return result

    def frame_indices(self, coordinate: SequenceCoordinate) -> tuple[int, ...]:
        if coordinate.loop:
            return tuple((coordinate.start + offset) % 72 for offset in range(self.config.sequence_frames))
        return tuple(coordinate.start + offset for offset in range(self.config.sequence_frames))

    def sequence(
        self,
        update: int,
        device: str | torch.device = "cpu",
        *,
        force_seam: bool = False,
    ) -> tuple[list[dict[str, Tensor]], list[SequenceCoordinate]]:
        coordinates = self.coordinates(update, force_seam=force_seam)
        frame_indices = [self.frame_indices(coordinate) for coordinate in coordinates]
        frames: list[dict[str, Tensor]] = []
        for offset in range(self.config.sequence_frames):
            rows = [
                self.teacher.sample(coordinate.chassis, coordinate.motion, indices[offset])
                for coordinate, indices in zip(coordinates, frame_indices, strict=True)
            ]
            frame: dict[str, Tensor] = {
                name: torch.from_numpy(np.stack([row[name] for row in rows]).copy()).to(device)
                for name in ("static", "state", "target", "controls")
            }
            frame["mask"] = torch.from_numpy(np.stack([row["mask"] for row in rows]).copy()).to(device)
            frame["adjacency"] = torch.from_numpy(np.stack([row["adjacency"] for row in rows]).copy()).to(device)
            for name in ("family", "morphotype", "motion"):
                frame[name] = torch.tensor([int(row[name]) for row in rows], dtype=torch.long, device=device)
            frame["phase"] = torch.tensor([float(row["phase"]) for row in rows], dtype=torch.float32, device=device)
            frames.append(frame)
        return frames, coordinates
