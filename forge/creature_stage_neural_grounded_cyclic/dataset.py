from __future__ import annotations

from typing import Any

import numpy as np

from ..creature_stage_neural_grounded.dataset import GroundedMotionTeacher
from .contract import EVALUATION_IDENTITIES, POSITION_SCALE, TRAIN_IDENTITIES


class RuntimeHonestGroundedTeacher(GroundedMotionTeacher):
    """Grounded authority projected without next-frame position leakage.

    The contact solver may provide an active contact, its world anchor, and its
    reaction force.  It may not provide the target cell position.  Grafted
    bodies also share their family's morphotype token so locomotion must be
    inferred from the component and contact channels rather than a held-out
    graft embedding.
    """

    @staticmethod
    def split_indices(split: str) -> tuple[int, ...]:
        if split == "train":
            return TRAIN_IDENTITIES
        if split == "validation":
            return EVALUATION_IDENTITIES
        if split == "all":
            return TRAIN_IDENTITIES + EVALUATION_IDENTITIES
        raise ValueError("runtime-honest grounded split drifted")

    def sample(self, identity: int, frame: int) -> dict[str, Any]:
        row = dict(super().sample(identity, frame))
        count = int(row["cell_count"])
        previous = (frame - 1) % 72
        current_cells = self.arrays["cells_local"][identity, previous]
        owners = self.arrays["appendage_owner"][identity, :count]
        active = self.arrays["contact_active"][identity, frame].astype(bool)
        dynamic = np.asarray(row["dynamic"]).copy()
        dynamic[:count, 6:8] = 0
        for cell_index, owner in enumerate(owners):
            if owner >= 0 and active[owner]:
                dynamic[cell_index, 6:8] = np.clip(
                    (self.arrays["contact_anchor_local"][identity, frame, owner] - current_cells[cell_index]) / POSITION_SCALE,
                    -1,
                    1,
                )
        dynamic[:count, 11] = np.clip(
            (self.arrays["body_world_x"][identity, previous] - self.arrays["body_world_x"][identity, 0]) / 16,
            -1, 1,
        )
        dynamic[:count, 12] = np.clip(
            (self.arrays["ground_y"][identity] - current_cells[:count, 1]) / 32,
            -1, 1,
        )
        dynamic.setflags(write=False)
        row["dynamic"] = dynamic
        row["morphotype"] = int(row["family"]) * 4
        return row
