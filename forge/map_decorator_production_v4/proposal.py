from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..map_art.hashing import bounded_hash
from ..map_art.styles import style_for
from ..map_decorator.catalog import MAX_DECAL_CLASSES, MAX_PROP_CLASSES, catalog_for
from ..map_decorator.hashing import named_arrays_sha256
from ..map_decorator_ml.contract import HEAD_NAMES
from ..map_decorator_ml.dataset import TeacherSample
from ..map_decorator_production.corpus import load_shard_array
from ..map_decorator_production.training import CorpusSampleRef
from ..map_decorator_production_v2.runner import V2CorpusAuthority
from ..maps.model import THEMES
from .contract import (
    DECAL_PROPOSAL_CHANNELS,
    OBJECT_PLACEMENT_SALT,
    PROP_PROPOSAL_CHANNELS,
    PROPOSAL_CHANNEL_MANIFEST_SHA256,
    V4_CONTRACT_SHA256,
)


_MASK64 = np.uint64((1 << 64) - 1)
_MIX_ADD = np.uint64(0x9E3779B97F4A7C15)
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_X_MUL = np.uint64(0xD6E8FEB86659FD93)
_Y_MUL = np.uint64(0xA5A3564E27F8862D)
_SALT_MUL = np.uint64(0x9E3779B97F4A7C15)


def _coordinate_hash_grid(seed: int, shape: tuple[int, int], salt: int) -> np.ndarray:
    height, width = shape
    if not 1 <= height <= 256 or not 1 <= width <= 256:
        raise ValueError("Proposal grid dimensions must remain in [1,256].")
    yy, xx = np.indices(shape, dtype=np.uint64)
    with np.errstate(over="ignore"):
        value = np.full(shape, np.uint64(seed), dtype=np.uint64)
        value ^= xx * _X_MUL
        value ^= yy * _Y_MUL
        value ^= np.uint64(salt) * _SALT_MUL
        value = (value + _MIX_ADD) & _MASK64
        value = ((value ^ (value >> np.uint64(30))) * _MIX_A) & _MASK64
        value = ((value ^ (value >> np.uint64(27))) * _MIX_B) & _MASK64
        value = (value ^ (value >> np.uint64(31))) & _MASK64
    return np.ascontiguousarray(value)


def _placement_mask(
    seed: int,
    shape: tuple[int, int],
    *,
    catalog_index: int,
    modulus: int,
    slots: tuple[int, ...],
) -> np.ndarray:
    hashed = _coordinate_hash_grid(
        seed,
        shape,
        OBJECT_PLACEMENT_SALT + int(catalog_index) * 17,
    )
    remainder = hashed % np.uint64(modulus)
    return np.ascontiguousarray(np.isin(remainder, np.asarray(slots, dtype=np.uint64)), dtype=bool)


@dataclass(slots=True)
class ProposalFields:
    decal: np.ndarray
    prop: np.ndarray
    map_seed: int
    theme: str
    channel_manifest_sha256: str
    fields_sha256: str

    def __post_init__(self) -> None:
        expected = {
            "decal": (DECAL_PROPOSAL_CHANNELS, *self.decal.shape[1:]),
            "prop": (PROP_PROPOSAL_CHANNELS, *self.decal.shape[1:]),
        }
        if self.decal.shape != expected["decal"] or self.prop.shape != expected["prop"]:
            raise ValueError("Proposal fields violate their fixed channel/shape contract.")
        if self.decal.dtype != np.bool_ or self.prop.dtype != np.bool_:
            raise TypeError("Proposal fields must be boolean arrays.")
        if self.channel_manifest_sha256 != PROPOSAL_CHANNEL_MANIFEST_SHA256:
            raise ValueError("Proposal channel manifest provenance drifted.")
        if self.fields_sha256 != named_arrays_sha256(self.arrays()):
            raise ValueError("Proposal field hash drifted.")
        self.decal.setflags(write=False)
        self.prop.setflags(write=False)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.decal.shape[1]), int(self.decal.shape[2])

    def arrays(self) -> dict[str, np.ndarray]:
        return {"decal": self.decal, "prop": self.prop}


def build_proposal_fields(
    *,
    map_seed: int,
    theme: str,
    shape: tuple[int, int],
    legal_masks: dict[str, np.ndarray],
    hard_empty: np.ndarray,
) -> ProposalFields:
    if isinstance(map_seed, bool) or not isinstance(map_seed, (int, np.integer)) or not 0 <= int(map_seed) < (1 << 64):
        raise ValueError("Proposal map_seed must be an unsigned 64-bit integer.")
    if theme not in THEMES:
        raise ValueError("Proposal theme is not authoritative.")
    if set(legal_masks) != set(HEAD_NAMES):
        raise ValueError("Proposal construction requires the complete legal-mask registry.")
    if hard_empty.shape != shape or hard_empty.dtype != np.bool_:
        raise TypeError("Proposal hard_empty must be a boolean map-shaped array.")
    expected_classes = {"decal": MAX_DECAL_CLASSES, "prop": MAX_PROP_CLASSES}
    for head, classes in expected_classes.items():
        value = legal_masks[head]
        if value.shape != (classes, *shape) or value.dtype != np.bool_:
            raise TypeError(f"Proposal legality for {head} has the wrong shape or dtype.")
    catalog = catalog_for(theme)
    style = style_for(theme)
    arrays = {
        "decal": np.zeros((DECAL_PROPOSAL_CHANNELS, *shape), dtype=bool),
        "prop": np.zeros((PROP_PROPOSAL_CHANNELS, *shape), dtype=bool),
    }
    for head, entries in (("decal", catalog.decal_classes), ("prop", catalog.prop_classes)):
        for entry in entries:
            spec = style.props[entry.catalog_index - 1]
            raw = _placement_mask(
                int(map_seed),
                shape,
                catalog_index=entry.catalog_index,
                modulus=spec.placement_modulus,
                slots=spec.placement_slots,
            )
            arrays[head][entry.class_id - 1] = (
                raw & legal_masks[head][entry.class_id] & ~hard_empty
            )
    decal = np.ascontiguousarray(arrays["decal"], dtype=bool)
    prop = np.ascontiguousarray(arrays["prop"], dtype=bool)
    return ProposalFields(
        decal=decal,
        prop=prop,
        map_seed=int(map_seed),
        theme=theme,
        channel_manifest_sha256=PROPOSAL_CHANNEL_MANIFEST_SHA256,
        fields_sha256=named_arrays_sha256({"decal": decal, "prop": prop}),
    )


def audit_proposal_targets(
    proposals: ProposalFields,
    targets: dict[str, np.ndarray],
) -> dict[str, Any]:
    if set(targets) != set(HEAD_NAMES):
        raise ValueError("Proposal audit requires all four target heads.")
    result: dict[str, Any] = {
        "v4_contract_sha256": V4_CONTRACT_SHA256,
        "proposal_fields_sha256": proposals.fields_sha256,
        "heads": {},
    }
    for head, fields in (("decal", proposals.decal), ("prop", proposals.prop)):
        target = targets[head]
        if target.shape != proposals.shape or target.dtype != np.uint8:
            raise TypeError(f"Proposal audit target {head} has the wrong shape or dtype.")
        target_count = int((target != 0).sum())
        proposal_count = int(fields.sum())
        hit_count = 0
        for channel in range(fields.shape[0]):
            hit_count += int((fields[channel] & (target == channel + 1)).sum())
        missing_count = target_count - hit_count
        result["heads"][head] = {
            "target_count": target_count,
            "proposal_count": proposal_count,
            "hit_count": hit_count,
            "missing_count": missing_count,
            "extra_count": proposal_count - hit_count,
            "recall": 1.0 if target_count == 0 else hit_count / target_count,
            "precision": 1.0 if proposal_count == 0 else hit_count / proposal_count,
        }
    result["passed"] = all(item["missing_count"] == 0 for item in result["heads"].values())
    return result


class ProposalAuthority:
    """Bind corpus samples to exact public map seeds without mutating v1/v2."""

    def __init__(self, authority: V2CorpusAuthority) -> None:
        self.authority = authority
        self._seed_arrays: dict[int, np.ndarray] = {}

    @classmethod
    def load(cls, corpus_root: Path, index_root: Path) -> "ProposalAuthority":
        return cls(V2CorpusAuthority.load(Path(corpus_root), Path(index_root)))

    def map_seed(self, ref: CorpusSampleRef) -> int:
        if ref.shard_index not in self._seed_arrays:
            spec, _ = self.authority.corpus.shards[ref.shard_index]
            self._seed_arrays[ref.shard_index] = np.ascontiguousarray(
                load_shard_array(self.authority.corpus.root, spec, "seeds"), dtype=np.uint64
            )
        return int(self._seed_arrays[ref.shard_index][ref.sample_index])

    def sample_and_proposals(self, ref: CorpusSampleRef) -> tuple[TeacherSample, ProposalFields]:
        sample = self.authority.corpus.sample(ref)
        theme = THEMES[sample.theme_index]
        proposals = build_proposal_fields(
            map_seed=self.map_seed(ref),
            theme=theme,
            shape=sample.shape,
            legal_masks=sample.legal_masks,
            hard_empty=sample.hard_empty,
        )
        return sample, proposals


def assert_vectorized_hash_exact() -> None:
    """Guard the NumPy hash projection against the authoritative scalar path."""
    for seed in (0, 1, 0xFFFF_FFFF_FFFF_FFFF, 0x1020304050607080):
        for salt in (0, OBJECT_PLACEMENT_SALT, OBJECT_PLACEMENT_SALT + 119):
            grid = _coordinate_hash_grid(seed, (9, 11), salt)
            for y in range(9):
                for x in range(11):
                    if int(grid[y, x]) != bounded_hash(seed, x, y, salt, 1 << 64):
                        raise RuntimeError("Vectorized proposal hash diverged from scalar authority.")
