from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch

from ..map_topology_neural.codec import collate_topology_tensors
from ..map_topology_neural.contract import encode_topology_tensor, inferred_conditions
from ..map_topology_neural.corpus import TopologyCorpus
from ..map_topology_neural.hashing import json_sha256
from .contract import TopologyCodecCalibrationConfig


SPLITS: Final[tuple[str, ...]] = ("train", "validation", "test")


def _split_for_identity(identity: str) -> str:
    if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("Topology reference identity is not canonical SHA-256.")
    bucket = int(identity[:16], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


@dataclass(frozen=True, slots=True)
class TopologyRef:
    shard_id: str
    sample_index: int
    split: str
    kind: str
    theme: str
    width: int
    height: int
    objective_count: int
    full_map_identity_sha256: str
    sample_identity_sha256: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def identity_payload(self) -> dict[str, object]:
        return {
            "shard_id": self.shard_id,
            "sample_index": self.sample_index,
            "split": self.split,
            "kind": self.kind,
            "theme": self.theme,
            "width": self.width,
            "height": self.height,
            "objective_count": self.objective_count,
            "full_map_identity_sha256": self.full_map_identity_sha256,
            "sample_identity_sha256": self.sample_identity_sha256,
        }


class TopologyProductionDataset:
    def __init__(self, corpus_root: Path) -> None:
        self.corpus = TopologyCorpus(Path(corpus_root))
        refs: list[TopologyRef] = []
        entries = self.corpus.manifest.get("shards")
        if not isinstance(entries, list):
            raise ValueError("Frozen topology corpus shard registry is malformed.")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Frozen topology corpus shard entry is malformed.")
            spec = entry.get("spec")
            full_ids = entry.get("full_map_identity_sha256")
            sample_ids = entry.get("sample_identity_sha256")
            if not isinstance(spec, dict) or not isinstance(full_ids, list) or not isinstance(sample_ids, list):
                raise ValueError("Frozen topology corpus reference registry is malformed.")
            if len(full_ids) != len(sample_ids):
                raise ValueError("Frozen topology corpus identity vectors disagree.")
            for index, (full_identity, sample_identity) in enumerate(zip(full_ids, sample_ids, strict=True)):
                split = _split_for_identity(str(full_identity))
                refs.append(
                    TopologyRef(
                        shard_id=str(entry["shard_id"]),
                        sample_index=index,
                        split=split,
                        kind=str(entry["kind"]),
                        theme=str(spec["theme"]),
                        width=int(spec["width"]),
                        height=int(spec["height"]),
                        objective_count=int(spec["objective_count"]),
                        full_map_identity_sha256=str(full_identity),
                        sample_identity_sha256=str(sample_identity),
                    )
                )
        refs.sort(key=lambda item: (item.split, item.kind, item.theme, item.height, item.width, item.objective_count, item.shard_id, item.sample_index))
        if len(refs) != 3_096 or len({item.full_map_identity_sha256 for item in refs}) != len(refs):
            raise ValueError("Frozen topology production reference census is not exact.")
        counts = {split: sum(item.split == split for item in refs) for split in SPLITS}
        if counts != {"train": 2496, "validation": 576, "test": 24}:
            raise ValueError("Frozen topology production split census drifted.")
        self.refs = tuple(refs)
        self.refs_by_split = {split: tuple(item for item in refs if item.split == split) for split in SPLITS}
        buckets: dict[tuple[int, int], list[TopologyRef]] = {}
        for item in self.refs_by_split["train"]:
            buckets.setdefault(item.shape, []).append(item)
        self.train_buckets = {shape: tuple(values) for shape, values in sorted(buckets.items())}
        if len(self.train_buckets) != 8:
            raise ValueError("Frozen topology training shape profile census drifted.")
        self.registry_sha256 = json_sha256([item.identity_payload() for item in refs])

    def evaluation_refs(self, split: str, count: int) -> tuple[TopologyRef, ...]:
        if split not in {"validation", "test"}:
            raise ValueError("Topology evaluation split must be validation or test.")
        available = self.refs_by_split[split]
        if split == "test":
            if len(available) != 24 or count not in {6, 24}:
                raise ValueError("Topology test evaluation must use 6 theme sentinels or all 24 sentinels.")
            if count == 24:
                return available
            groups: dict[str, list[TopologyRef]] = {}
            for item in available:
                groups.setdefault(item.theme, []).append(item)
            return tuple(min(values, key=lambda item: (item.height * item.width, item.full_map_identity_sha256)) for _, values in sorted(groups.items()))
        groups: dict[tuple[str, int, int], list[TopologyRef]] = {}
        for item in available:
            groups.setdefault((item.theme, item.height, item.width), []).append(item)
        if len(groups) != 48 or count not in {6, 48}:
            raise ValueError("Topology calibration validation must use 6 theme cells or all 48 theme/shape cells.")
        selected = tuple(sorted(values, key=lambda item: (item.objective_count, item.full_map_identity_sha256))[0] for _, values in sorted(groups.items()))
        if count == 48:
            return selected
        by_theme: dict[str, list[TopologyRef]] = {}
        for item in selected:
            by_theme.setdefault(item.theme, []).append(item)
        return tuple(min(values, key=lambda item: (item.height * item.width, item.full_map_identity_sha256)) for _, values in sorted(by_theme.items()))

    def load_tensor(self, ref: TopologyRef):
        sample = self.corpus.read_sample(ref.shard_id, ref.sample_index, expected_split=ref.split)
        if (
            sample.full_map_identity_sha256 != ref.full_map_identity_sha256
            or sample.sample_identity_sha256 != ref.sample_identity_sha256
            or sample.theme != ref.theme
            or sample.config.width != ref.width
            or sample.config.height != ref.height
        ):
            raise ValueError("Loaded topology sample drifted from its immutable reference.")
        return encode_topology_tensor(
            terrain=sample.raw.terrain,
            hazard=sample.raw.hazard,
            elevation=sample.raw.elevation,
            theme=sample.theme,
            config=sample.config,
            start=sample.start,
            exit=sample.exit,
            objectives=sample.objectives,
            spawns=sample.spawns,
            conditions=inferred_conditions(sample.raw.terrain, sample.raw.hazard),
        )

    def collate(self, refs: tuple[TopologyRef, ...], device: torch.device) -> dict[str, torch.Tensor]:
        batch = collate_topology_tensors([self.load_tensor(ref) for ref in refs])
        return {name: value.to(device, non_blocking=False) for name, value in batch.items()}

    def training_refs(
        self,
        step: int,
        generator: torch.Generator,
        config: TopologyCodecCalibrationConfig,
    ) -> tuple[TopologyRef, ...]:
        shapes = tuple(self.train_buckets)
        shape = shapes[step % len(shapes)]
        bucket = self.train_buckets[shape]
        batch_size = min(config.maximum_batch_size, max(1, config.cell_budget // (shape[0] * shape[1])), len(bucket))
        order = torch.randperm(len(bucket), generator=generator)[:batch_size].tolist()
        return tuple(bucket[index] for index in order)


def ref_registry_sha256(refs: tuple[TopologyRef, ...]) -> str:
    return json_sha256([item.identity_payload() for item in refs])
