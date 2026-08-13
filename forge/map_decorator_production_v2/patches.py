from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Final

import numpy as np

from ..map_decorator.hashing import array_sha256, json_sha256, named_arrays_sha256
from ..map_decorator_ml.contract import HEAD_NAMES
from ..map_decorator_ml.dataset import Crop, TeacherSample
from .contract import ForegroundPatchConfig, V2_CONTRACT_SHA256


FOCUS_HEADS: Final[tuple[str, str]] = ("decal", "prop")


@dataclass(frozen=True, slots=True)
class ForegroundSampleStat:
    shard_index: int
    sample_index: int
    split: str
    map_id: str
    sample_identity_sha256: str
    full_map_identity_sha256: str
    decal_count: int
    prop_count: int

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("Foreground sample stat has an unknown split.")
        if min(self.shard_index, self.sample_index, self.decal_count, self.prop_count) < 0:
            raise ValueError("Foreground sample stat counts and indices cannot be negative.")
        for value in (self.sample_identity_sha256, self.full_map_identity_sha256):
            if len(value) != 64:
                raise ValueError("Foreground sample identities must be SHA-256 strings.")
            int(value, 16)

    def count(self, head: str) -> int:
        if head == "decal":
            return self.decal_count
        if head == "prop":
            return self.prop_count
        raise ValueError(f"Unsupported foreground focus head {head!r}.")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "shard_index": self.shard_index,
            "sample_index": self.sample_index,
            "split": self.split,
            "map_id": self.map_id,
            "sample_identity_sha256": self.sample_identity_sha256,
            "full_map_identity_sha256": self.full_map_identity_sha256,
            "decal_count": self.decal_count,
            "prop_count": self.prop_count,
        }


@dataclass(frozen=True, slots=True)
class PlannedForegroundSample:
    stat: ForegroundSampleStat
    focus_head: str

    def __post_init__(self) -> None:
        if self.focus_head not in FOCUS_HEADS or self.stat.count(self.focus_head) <= 0:
            raise ValueError("Planned focus must have at least one teacher foreground cell.")


def _stable_seed(*parts: object) -> int:
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def plan_foreground_batches(
    stats: tuple[ForegroundSampleStat, ...] | list[ForegroundSampleStat],
    *,
    steps: int,
    epoch: int,
    seed: int,
    config: ForegroundPatchConfig = ForegroundPatchConfig(),
) -> tuple[tuple[PlannedForegroundSample, ...], ...]:
    if isinstance(steps, bool) or not 1 <= steps <= 100_000:
        raise ValueError("steps must be an integer in [1,100000].")
    if isinstance(epoch, bool) or epoch < 0:
        raise ValueError("epoch cannot be negative.")
    if not stats:
        raise ValueError("Foreground quota planning requires non-empty sample statistics.")
    identities = [stat.sample_identity_sha256 for stat in stats]
    if len(identities) != len(set(identities)):
        raise ValueError("Foreground quota input contains duplicate sample identities.")
    splits = {stat.split for stat in stats}
    if len(splits) != 1:
        raise ValueError("A foreground quota plan may cover exactly one split.")
    split = next(iter(splits))

    pools: dict[str, list[ForegroundSampleStat]] = {
        head: [stat for stat in stats if stat.count(head) > 0] for head in FOCUS_HEADS
    }
    required = {"decal": config.decal_slots, "prop": config.prop_slots}
    for head in FOCUS_HEADS:
        if len(pools[head]) < required[head]:
            raise ValueError(f"Insufficient distinct {head} foreground maps for one batch.")
        rng = random.Random(_stable_seed(V2_CONTRACT_SHA256, seed, epoch, head, split))
        rng.shuffle(pools[head])

    cursors = {head: 0 for head in FOCUS_HEADS}
    batches: list[tuple[PlannedForegroundSample, ...]] = []
    focus_order = tuple(
        ["decal"] * config.decal_slots + ["prop"] * config.prop_slots
    )
    for step in range(steps):
        # Rotate the slot order so optimizer positions do not encode a head identity.
        offset = step % len(focus_order)
        rotated = focus_order[offset:] + focus_order[:offset]
        used: set[str] = set()
        planned: list[PlannedForegroundSample] = []
        for head in rotated:
            pool = pools[head]
            selected = None
            for _ in range(len(pool)):
                candidate = pool[cursors[head] % len(pool)]
                cursors[head] += 1
                if candidate.sample_identity_sha256 not in used:
                    selected = candidate
                    break
            if selected is None:
                raise ValueError("Bounded quota selection could not form a duplicate-free batch.")
            used.add(selected.sample_identity_sha256)
            planned.append(PlannedForegroundSample(selected, head))
        if len(planned) != config.batch_size:
            raise RuntimeError("Foreground quota plan failed to exhaust its batch contract.")
        batches.append(tuple(planned))
    return tuple(batches)


def foreground_centered_crop(
    sample: TeacherSample,
    *,
    focus_head: str,
    epoch: int,
    step: int,
    slot: int,
    seed: int,
    config: ForegroundPatchConfig = ForegroundPatchConfig(),
) -> TeacherSample:
    if focus_head not in FOCUS_HEADS:
        raise ValueError(f"Unsupported foreground focus head {focus_head!r}.")
    if sample.crop is not None:
        raise ValueError("V2 foreground crops must start from an uncropped full-map sample.")
    positions = np.argwhere(sample.targets[focus_head] != 0)
    if positions.size == 0:
        raise ValueError(f"Sample {sample.map_id!r} has no {focus_head} foreground to center.")
    rng = random.Random(
        _stable_seed(
            V2_CONTRACT_SHA256,
            sample.sample_identity_sha256,
            focus_head,
            seed,
            epoch,
            step,
            slot,
        )
    )
    center_y, center_x = positions[rng.randrange(len(positions))]
    full_h, full_w = sample.shape
    crop_h = min(config.patch_size, full_h)
    crop_w = min(config.patch_size, full_w)
    jitter_x = rng.randint(-config.jitter_radius, config.jitter_radius)
    jitter_y = rng.randint(-config.jitter_radius, config.jitter_radius)
    x = max(0, min(full_w - crop_w, int(center_x) - crop_w // 2 + jitter_x))
    y = max(0, min(full_h - crop_h, int(center_y) - crop_h // 2 + jitter_y))
    # Clamp again against the chosen teacher point; jitter may never push it outside.
    x = min(x, int(center_x))
    x = max(x, int(center_x) - crop_w + 1)
    y = min(y, int(center_y))
    y = max(y, int(center_y) - crop_h + 1)
    crop = Crop(x=x, y=y, width=crop_w, height=crop_h)
    ys, xs = crop.slices(sample.shape)
    features = np.ascontiguousarray(sample.features[:, ys, xs], dtype=np.float32)
    targets = {
        name: np.ascontiguousarray(sample.targets[name][ys, xs], dtype=np.uint8)
        for name in HEAD_NAMES
    }
    legal_masks = {
        name: np.ascontiguousarray(sample.legal_masks[name][:, ys, xs], dtype=bool)
        for name in HEAD_NAMES
    }
    hard_empty = np.ascontiguousarray(sample.hard_empty[ys, xs], dtype=bool)
    if not bool((targets[focus_head] != 0).any()):
        raise RuntimeError("Foreground-centered crop lost its required teacher foreground.")
    feature_hash = array_sha256(features)
    target_hash = named_arrays_sha256(targets)
    identity = json_sha256(
        {
            "contract_sha256": V2_CONTRACT_SHA256,
            "parent_sample_identity_sha256": sample.sample_identity_sha256,
            "focus_head": focus_head,
            "crop": crop.to_dict(),
            "epoch": epoch,
            "step": step,
            "slot": slot,
            "feature_tensor_sha256": feature_hash,
            "target_fields_sha256": target_hash,
        }
    )
    return TeacherSample(
        features=features,
        targets=targets,
        legal_masks=legal_masks,
        hard_empty=hard_empty,
        global_conditions=np.ascontiguousarray(sample.global_conditions, dtype=np.float32),
        theme_index=sample.theme_index,
        split=sample.split,
        full_map_identity_sha256=sample.full_map_identity_sha256,
        sample_identity_sha256=identity,
        source_semantic_sha256=sample.source_semantic_sha256,
        feature_tensor_sha256=feature_hash,
        target_fields_sha256=target_hash,
        map_id=sample.map_id,
        crop=crop,
    )
