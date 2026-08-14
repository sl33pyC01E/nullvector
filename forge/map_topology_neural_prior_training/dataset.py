from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..map_topology_neural_prior_corpus.shard import _load_arrays
from ..map_topology_neural_prior_corpus.supervisor import ROOT_MANIFEST, validate_corpus
from .contract import (
    FROZEN_LATENT_CORPUS_IDENTITY_SHA256,
    FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256,
    FROZEN_LATENT_CORPUS_MANIFEST_SHA256,
    FROZEN_LATENT_CORPUS_SOURCE_SHA256,
    PriorCalibrationConfig,
    sha256_file,
)


@dataclass(frozen=True, slots=True)
class LatentRef:
    shard_id: str
    local_index: int
    sample_index: int
    split: str
    theme: str
    height: int
    width: int
    full_map_identity_sha256: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width


class PriorTrainingDataset:
    def __init__(self, corpus_root: Path, latent_root: Path) -> None:
        self.corpus_root = Path(corpus_root).resolve()
        self.latent_root = Path(latent_root).resolve()
        manifest_path = self.latent_root / ROOT_MANIFEST
        if sha256_file(manifest_path) != FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256:
            raise ValueError("Prior training latent corpus manifest file drifted.")
        manifest = validate_corpus(self.corpus_root, self.latent_root)
        if (
            manifest["manifest_sha256"] != FROZEN_LATENT_CORPUS_MANIFEST_SHA256
            or manifest["latent_corpus_identity_sha256"] != FROZEN_LATENT_CORPUS_IDENTITY_SHA256
            or manifest["source_sha256"] != FROZEN_LATENT_CORPUS_SOURCE_SHA256
        ):
            raise ValueError("Prior training latent corpus authority drifted.")
        refs: list[LatentRef] = []
        for record in manifest["shards"]:
            shard_manifest = json.loads((self.latent_root / record["manifest"]).read_text(encoding="utf-8"))
            for local_index, source in enumerate(shard_manifest["source_refs"]):
                refs.append(LatentRef(
                    shard_id=record["shard_id"], local_index=local_index,
                    sample_index=int(source["sample_index"]), split=str(source["split"]),
                    theme=str(source["theme"]), height=int(record["shape"][0]),
                    width=int(record["shape"][1]),
                    full_map_identity_sha256=str(source["full_map_identity_sha256"]),
                ))
        refs.sort(key=lambda ref: (ref.split, ref.shape, ref.theme, ref.full_map_identity_sha256))
        self.refs = tuple(refs)
        self.refs_by_split = {split: tuple(ref for ref in refs if ref.split == split) for split in ("train", "validation", "test")}
        if {split: len(values) for split, values in self.refs_by_split.items()} != {"train": 2496, "validation": 576, "test": 24}:
            raise ValueError("Prior training split census drifted.")
        buckets: dict[tuple[int, int], list[LatentRef]] = {}
        for ref in self.refs_by_split["train"]:
            buckets.setdefault(ref.shape, []).append(ref)
        self.train_buckets = {shape: tuple(values) for shape, values in sorted(buckets.items())}
        if len(self.train_buckets) != 8:
            raise ValueError("Prior training shape bucket census drifted.")

    def evaluation_refs(self, split: str, count: int) -> tuple[LatentRef, ...]:
        if split == "test":
            refs = self.refs_by_split["test"]
            if count == 24:
                return refs
            groups: dict[str, list[LatentRef]] = {}
            for ref in refs:
                groups.setdefault(ref.theme, []).append(ref)
            return tuple(min(values, key=lambda ref: (ref.height * ref.width, ref.full_map_identity_sha256)) for _, values in sorted(groups.items()))
        if split != "validation":
            raise ValueError("Prior evaluation split must be validation or test.")
        groups: dict[tuple[str, int, int], list[LatentRef]] = {}
        for ref in self.refs_by_split["validation"]:
            groups.setdefault((ref.theme, ref.height, ref.width), []).append(ref)
        selected = tuple(min(values, key=lambda ref: ref.full_map_identity_sha256) for _, values in sorted(groups.items()))
        if len(selected) != 48:
            raise ValueError("Prior validation theme/shape census drifted.")
        if count == 48:
            return selected
        by_theme: dict[str, list[LatentRef]] = {}
        for ref in selected:
            by_theme.setdefault(ref.theme, []).append(ref)
        return tuple(min(values, key=lambda ref: (ref.height * ref.width, ref.full_map_identity_sha256)) for _, values in sorted(by_theme.items()))

    def training_refs(self, step: int, generator: torch.Generator, config: PriorCalibrationConfig) -> tuple[LatentRef, ...]:
        shapes = tuple(self.train_buckets)
        shape = shapes[step % len(shapes)]
        bucket = self.train_buckets[shape]
        batch_size = min(config.maximum_batch_size, max(1, config.cell_budget // (shape[0] * shape[1])), len(bucket))
        order = torch.randperm(len(bucket), generator=generator)[:batch_size].tolist()
        return tuple(bucket[index] for index in order)

    def collate(self, refs: tuple[LatentRef, ...]) -> dict[str, torch.Tensor]:
        if not refs or len({ref.shape for ref in refs}) != 1:
            raise ValueError("Prior training batch must be nonempty and homogeneous.")
        by_shard: dict[str, list[tuple[int, LatentRef]]] = {}
        for output_index, ref in enumerate(refs):
            by_shard.setdefault(ref.shard_id, []).append((output_index, ref))
        count = len(refs)
        height, width = refs[0].shape
        arrays = {
            "targets": torch.empty((count, height, width), dtype=torch.long),
            "valid_mask": torch.empty((count, 1, height, width), dtype=torch.bool),
            "point_conditions": torch.empty((count, 4, height, width), dtype=torch.float32),
            "global_conditions": torch.empty((count, 14), dtype=torch.float32),
            "theme_index": torch.empty((count,), dtype=torch.long),
        }
        for shard_id, members in by_shard.items():
            stored = _load_arrays(self.latent_root / "shards" / shard_id / "latents.npz")
            for output_index, ref in members:
                index = ref.local_index
                if int(stored["sample_index"][index]) != ref.sample_index:
                    raise ValueError("Prior training shard sample index drifted.")
                arrays["targets"][output_index] = torch.from_numpy(stored["tokens"][index].astype(np.int64))
                arrays["valid_mask"][output_index, 0] = torch.from_numpy(stored["valid_mask"][index].astype(bool))
                arrays["point_conditions"][output_index] = torch.from_numpy(stored["point_conditions"][index].astype(np.float32))
                arrays["global_conditions"][output_index] = torch.from_numpy(stored["global_conditions"][index].copy())
                arrays["theme_index"][output_index] = int(stored["theme_index"][index])
        return arrays

