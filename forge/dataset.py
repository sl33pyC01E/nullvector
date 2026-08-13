from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import ARCHETYPES
from .grammar import genome_from_seed, genome_vector, mix32, render_layers
from .safety import require_disk_floor


def _dataset_seed(base_seed: int, index: int) -> int:
    return mix32(base_seed ^ ((index + 1) * 0x9E3779B1))


def build_corpus(
    path: Path,
    count: int,
    base_seed: int,
    *,
    force: bool = False,
) -> Path:
    path = Path(path)
    if path.exists() and not force:
        with np.load(path) as cached:
            if (
                int(cached["layers"].shape[0]) == count
                and int(cached["base_seed"][0]) == base_seed
            ):
                return path

    estimated_bytes = count * (8 * 32 * 32 + 8 * 4 + 8)
    require_disk_floor(path.parent, planned_bytes=estimated_bytes * 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = np.empty((count, 8, 32, 32), dtype=np.uint8)
    labels = np.empty((count,), dtype=np.uint8)
    seeds = np.empty((count,), dtype=np.uint32)
    genes = np.empty((count, 8), dtype=np.float32)

    for index in tqdm(range(count), desc="constructing sprite corpus", unit="sprite"):
        seed = _dataset_seed(base_seed, index)
        archetype = index % len(ARCHETYPES)
        genome = genome_from_seed(seed, archetype)
        layers[index] = render_layers(genome)
        labels[index] = archetype
        seeds[index] = seed
        genes[index] = genome_vector(genome)

    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        layers=layers,
        labels=labels,
        seeds=seeds,
        genes=genes,
        base_seed=np.asarray([base_seed], dtype=np.uint32),
    )
    temporary.replace(path)
    return path


class CachedSpriteDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(self, path: Path, indices: np.ndarray | None = None) -> None:
        payload = np.load(Path(path))
        self.layers = payload["layers"]
        self.labels = payload["labels"]
        self.seeds = payload["seeds"]
        self.genes = payload["genes"]
        self.indices = (
            np.asarray(indices, dtype=np.int64)
            if indices is not None
            else np.arange(len(self.layers), dtype=np.int64)
        )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_index = int(self.indices[index])
        layers = torch.from_numpy(self.layers[source_index].astype(np.float32, copy=True))
        label = torch.tensor(int(self.labels[source_index]), dtype=torch.long)
        seed = torch.tensor(int(self.seeds[source_index]), dtype=torch.long)
        genes = torch.from_numpy(self.genes[source_index].astype(np.float32, copy=True))
        return layers, label, seed, genes


def split_indices(count: int, validation_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if validation_size <= 0 or validation_size >= count:
        raise ValueError("validation_size must be between zero and corpus size.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    return order[validation_size:], order[:validation_size]


def layer_occupancy(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        layers = payload["layers"]
        return layers.mean(axis=(0, 2, 3), dtype=np.float64)
