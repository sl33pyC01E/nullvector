from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset

from .morphology.constants import (
    EMISSION_LEVEL_NAMES,
    FAMILIES,
    GUIDE_CHANNEL_NAMES,
    MATERIAL_NAMES,
    PART_OWNER_NAMES,
    ROLE_NAMES,
    SUBTYPE_NAMES,
)
from .morphology.corpus import CORPUS_FORMAT, corpus_source_hash, split_indices


FIELD_KEYS = ("part_owner", "material", "emission_level")
CONDITION_KEYS = ("morphologies", "subtypes", "roles")
REQUIRED_ARRAY_KEYS = (
    "guide",
    *FIELD_KEYS,
    "genes",
    *CONDITION_KEYS,
    "seeds",
)
REQUIRED_METADATA_KEYS = (
    "format",
    "split_version",
    "corpus_source_sha256",
    "genome_version",
    "renderer_version",
    "semantic_format",
    "base_seed",
    "guide_storage_dtype",
    "semantic_sha256",
    "training_arrays_sha256",
    "guide_channel_names",
    "part_owner_names",
    "material_names",
    "emission_level_names",
    "family_names",
    "subtype_names",
    "role_names",
)
GUIDE_NAME_TO_INDEX = {
    name: index for index, name in enumerate(GUIDE_CHANNEL_NAMES)
}
SCAFFOLD_GUIDE_INDICES = tuple(
    GUIDE_NAME_TO_INDEX[name] for name in ("skeleton", "joints", "sockets")
)
TARGET_DERIVED_GUIDE_INDICES = tuple(
    GUIDE_NAME_TO_INDEX[name] for name in ("silhouette", "body", "core")
)


@dataclass(frozen=True, slots=True)
class GuidePolicy:
    """Versioned anti-leakage preprocessing for morphology scaffold guides."""

    name: str = "scaffold_only"
    version: str = "scaffold-guide-policy-v1"
    thicken_radius: int = 1
    training_channel_dropout: float = 0.08
    training_jitter_pixels: int = 1

    def __post_init__(self) -> None:
        if self.name not in {"scaffold_only", "full_debug"}:
            raise ValueError("Guide policy must be scaffold_only or full_debug.")
        if self.thicken_radius < 0 or self.thicken_radius > 3:
            raise ValueError("thicken_radius must be between zero and three.")
        if not 0.0 <= self.training_channel_dropout < 1.0:
            raise ValueError("training_channel_dropout must be in [0, 1).")
        if self.training_jitter_pixels < 0 or self.training_jitter_pixels > 4:
            raise ValueError("training_jitter_pixels must be between zero and four.")

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "zeroed_target_channels": (
                ["silhouette", "body", "core"]
                if self.name == "scaffold_only"
                else []
            ),
            "retained_channels": (
                [
                    "skeleton",
                    "joints",
                    "sockets",
                    "horizontal_position",
                    "root_distance",
                ]
                if self.name == "scaffold_only"
                else list(GUIDE_CHANNEL_NAMES)
            ),
            "thicken_radius": self.thicken_radius,
            "training_channel_dropout": self.training_channel_dropout,
            "training_jitter_pixels": self.training_jitter_pixels,
            "evaluation_augmentation": False,
        }


def apply_guide_policy(guide: Tensor, policy: GuidePolicy) -> Tensor:
    """Apply non-random guide sanitation to one [C,H,W] tensor."""
    if guide.ndim != 3 or guide.shape[0] != len(GUIDE_CHANNEL_NAMES):
        raise ValueError("guide must have shape [8, height, width].")
    result = guide.float().clone()
    if policy.name == "scaffold_only":
        result[list(TARGET_DERIVED_GUIDE_INDICES)] = 0.0
    if policy.thicken_radius:
        kernel = policy.thicken_radius * 2 + 1
        scaffold = F.max_pool2d(
            result[list(SCAFFOLD_GUIDE_INDICES)].unsqueeze(0),
            kernel_size=kernel,
            stride=1,
            padding=policy.thicken_radius,
        ).squeeze(0)
        result[list(SCAFFOLD_GUIDE_INDICES)] = scaffold
    return result.clamp_(0.0, 1.0)


def augment_scaffold_guides(
    guides: Tensor,
    policy: GuidePolicy,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Apply checkpoint-replayable training-only scaffold dropout and translation."""
    if guides.ndim != 4 or guides.shape[1] != len(GUIDE_CHANNEL_NAMES):
        raise ValueError("guides must have shape [batch, 8, height, width].")
    if policy.name != "scaffold_only":
        return guides
    result = guides.clone()
    batch = result.shape[0]
    if policy.training_channel_dropout > 0.0:
        keep = (
            torch.rand(
                (batch, 3, 1, 1),
                device=result.device,
                generator=generator,
            )
            >= policy.training_channel_dropout
        )
        # Retain at least one of skeleton/joints/sockets per specimen.
        empty = ~keep.flatten(1).any(dim=1)
        keep[empty, 0] = True
        result[:, list(SCAFFOLD_GUIDE_INDICES)] *= keep
    radius = policy.training_jitter_pixels
    if radius:
        offsets = torch.randint(
            -radius,
            radius + 1,
            (batch, 2),
            device=result.device,
            generator=generator,
        )
        height, width = result.shape[-2:]
        shifted = torch.zeros_like(result[:, list(SCAFFOLD_GUIDE_INDICES)])
        for index in range(batch):
            dx, dy = map(int, offsets[index].tolist())
            source_x0, source_x1 = max(0, -dx), min(width, width - dx)
            source_y0, source_y1 = max(0, -dy), min(height, height - dy)
            target_x0, target_x1 = source_x0 + dx, source_x1 + dx
            target_y0, target_y1 = source_y0 + dy, source_y1 + dy
            shifted[index, :, target_y0:target_y1, target_x0:target_x1] = result[
                index,
                list(SCAFFOLD_GUIDE_INDICES),
                source_y0:source_y1,
                source_x0:source_x1,
            ]
        result[:, list(SCAFFOLD_GUIDE_INDICES)] = shifted
    return result


def sha256_file(path: Path, *, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_bounds_in_chunks(
    values: np.ndarray, *, chunk_size: int = 256
) -> tuple[float, float]:
    minimum = float("inf")
    maximum = float("-inf")
    for start in range(0, len(values), chunk_size):
        chunk = values[start : start + chunk_size]
        if not np.isfinite(chunk).all():
            raise ValueError("Corpus contains non-finite floating-point values.")
        minimum = min(minimum, float(chunk.min(initial=np.inf)))
        maximum = max(maximum, float(chunk.max(initial=-np.inf)))
    return minimum, maximum


def _require_shape(values: np.ndarray, expected: tuple[int, ...], name: str) -> None:
    if values.shape != expected:
        raise ValueError(f"{name} has shape {values.shape}; expected {expected}.")


@dataclass(slots=True)
class MorphologyCorpus:
    """Validated in-memory view of a morphology training archive.

    Compressed NPZ members cannot be memory-mapped. Loading each member once is
    substantially faster and safer than reopening the zip archive in DataLoader
    workers. Keep ``num_workers=0`` for very large corpora on Windows so the
    arrays are not copied into spawned worker processes.
    """

    path: Path
    file_sha256: str
    format: str
    base_seed: int
    split_version: str
    corpus_source_sha256: str
    genome_version: str
    renderer_version: str
    semantic_format: str
    guide: np.ndarray
    part_owner: np.ndarray
    material: np.ndarray
    emission_level: np.ndarray
    genes: np.ndarray
    morphologies: np.ndarray
    subtypes: np.ndarray
    roles: np.ndarray
    seeds: np.ndarray

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        verify_hash: bool = True,
        verify_source: bool = True,
    ) -> "MorphologyCorpus":
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Morphology corpus does not exist: {resolved}")
        digest = sha256_file(resolved) if verify_hash else "not-computed"
        with np.load(resolved, allow_pickle=False) as payload:
            missing = sorted(
                set((*REQUIRED_ARRAY_KEYS, *REQUIRED_METADATA_KEYS)).difference(
                    payload.files
                )
            )
            if missing:
                raise ValueError(f"Corpus is missing required keys: {missing}")
            corpus_format = str(payload["format"][0])
            base_seed = (
                int(payload["base_seed"][0])
                if "base_seed" in payload.files
                else 0
            )
            provenance = {
                name: str(payload[name][0]) if name in payload.files else "unknown"
                for name in (
                    "split_version",
                    "corpus_source_sha256",
                    "genome_version",
                    "renderer_version",
                    "semantic_format",
                )
            }
            arrays = {
                key: np.ascontiguousarray(payload[key]) for key in REQUIRED_ARRAY_KEYS
            }
            names = {
                "guide_channel_names": GUIDE_CHANNEL_NAMES,
                "part_owner_names": PART_OWNER_NAMES,
                "material_names": MATERIAL_NAMES,
                "emission_level_names": EMISSION_LEVEL_NAMES,
                "family_names": FAMILIES,
                "subtype_names": SUBTYPE_NAMES,
                "role_names": ROLE_NAMES,
            }
            for key, expected in names.items():
                if tuple(map(str, payload[key].tolist())) != tuple(expected):
                    raise ValueError(f"Corpus {key} disagrees with the active vocabulary.")
            count = int(arrays["guide"].shape[0])
            for key in ("semantic_sha256", "training_arrays_sha256"):
                if payload[key].shape != (count,):
                    raise ValueError(f"Corpus {key} must have one entry per specimen.")
        corpus = cls(
            path=resolved,
            file_sha256=digest,
            format=corpus_format,
            base_seed=base_seed,
            **provenance,
            **arrays,
        )
        corpus.validate()
        if verify_source:
            active_source_hash = corpus_source_hash()
            if corpus.corpus_source_sha256 != active_source_hash:
                raise ValueError(
                    "Corpus source hash does not match the active morphology renderer; "
                    "rebuild the corpus before training."
                )
        return corpus

    @property
    def count(self) -> int:
        return int(self.guide.shape[0])

    @property
    def image_size(self) -> int:
        return int(self.guide.shape[-1])

    def validate(self) -> None:
        if self.format != CORPUS_FORMAT:
            raise ValueError(
                f"Unsupported corpus format {self.format!r}; expected {CORPUS_FORMAT!r}."
            )
        if self.guide.ndim != 4:
            raise ValueError("guide must have shape [samples, channels, height, width].")
        count, channels, height, width = self.guide.shape
        if count <= 1 or channels != len(GUIDE_CHANNEL_NAMES) or height != width:
            raise ValueError(
                "guide must contain at least two square, eight-channel specimens."
            )
        if self.guide.dtype not in (np.float16, np.float32):
            raise ValueError(
                f"guide storage must be float16 or float32, got {self.guide.dtype}."
            )
        guide_min, guide_max = _array_bounds_in_chunks(self.guide)
        if guide_min < 0.0 or guide_max > 1.0:
            raise ValueError("guide values must stay in [0, 1].")

        spatial = (count, height, width)
        categorical = (
            (self.part_owner, len(PART_OWNER_NAMES), "part_owner"),
            (self.material, len(MATERIAL_NAMES), "material"),
            (self.emission_level, len(EMISSION_LEVEL_NAMES), "emission_level"),
        )
        for values, vocabulary_size, name in categorical:
            _require_shape(values, spatial, name)
            if values.dtype != np.uint8:
                raise ValueError(f"{name} must be uint8, got {values.dtype}.")
            if int(values.max(initial=0)) >= vocabulary_size:
                raise ValueError(f"{name} contains an out-of-vocabulary value.")

        _require_shape(self.genes, (count, 24), "genes")
        if self.genes.dtype != np.float32:
            raise ValueError(f"genes must be float32, got {self.genes.dtype}.")
        gene_min, gene_max = _array_bounds_in_chunks(self.genes)
        if gene_min < 0.0 or gene_max > 1.0:
            raise ValueError("gene values must stay in [0, 1].")

        conditions = (
            (self.morphologies, len(FAMILIES), "morphologies"),
            (self.subtypes, len(SUBTYPE_NAMES), "subtypes"),
            (self.roles, len(ROLE_NAMES), "roles"),
        )
        for values, vocabulary_size, name in conditions:
            _require_shape(values, (count,), name)
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{name} must have an integer dtype.")
            if int(values.min(initial=0)) < 0 or int(values.max(initial=0)) >= vocabulary_size:
                raise ValueError(f"{name} contains an out-of-vocabulary value.")
        _require_shape(self.seeds, (count,), "seeds")
        if not np.issubdtype(self.seeds.dtype, np.unsignedinteger):
            raise ValueError("seeds must have an unsigned integer dtype.")

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "format": self.format,
            "file_sha256": self.file_sha256,
            "file_bytes": self.path.stat().st_size,
            "base_seed": self.base_seed,
            "split_version": self.split_version,
            "corpus_source_sha256": self.corpus_source_sha256,
            "genome_version": self.genome_version,
            "renderer_version": self.renderer_version,
            "semantic_format": self.semantic_format,
            "samples": self.count,
            "image_size": self.image_size,
            "guide_channels": len(GUIDE_CHANNEL_NAMES),
            "gene_dim": int(self.genes.shape[1]),
            "vocabulary": {
                "part_count": len(PART_OWNER_NAMES),
                "material_count": len(MATERIAL_NAMES),
                "emission_count": len(EMISSION_LEVEL_NAMES),
                "morphology_count": len(FAMILIES),
                "subtype_count": len(SUBTYPE_NAMES),
                "role_count": len(ROLE_NAMES),
            },
        }


class MorphologyCorpusDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        corpus: MorphologyCorpus,
        indices: Sequence[int] | np.ndarray,
        *,
        guide_policy: GuidePolicy = GuidePolicy(),
    ):
        self.corpus = corpus
        self.guide_policy = guide_policy
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or len(self.indices) == 0:
            raise ValueError("Dataset indices must be a non-empty one-dimensional array.")
        if int(self.indices.min()) < 0 or int(self.indices.max()) >= corpus.count:
            raise IndexError("Dataset indices contain an out-of-range corpus row.")
        if len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("Dataset indices must be unique.")

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        source = int(self.indices[index])
        return {
            # Corpus v2 stores guides in float16 to keep the 32k archive compact;
            # CPU convolutions do not support half reliably, so expose float32 and
            # let autocast select BF16/FP16 on accelerator transfer.
            "guide": apply_guide_policy(
                torch.from_numpy(self.corpus.guide[source]), self.guide_policy
            ),
            "part": torch.from_numpy(self.corpus.part_owner[source]).long(),
            "material": torch.from_numpy(self.corpus.material[source]).long(),
            "emission": torch.from_numpy(self.corpus.emission_level[source]).long(),
            "genes": torch.from_numpy(self.corpus.genes[source]),
            "morphology": torch.tensor(
                int(self.corpus.morphologies[source]), dtype=torch.long
            ),
            "subtype": torch.tensor(int(self.corpus.subtypes[source]), dtype=torch.long),
            "role": torch.tensor(int(self.corpus.roles[source]), dtype=torch.long),
            "seed": torch.tensor(int(self.corpus.seeds[source]), dtype=torch.int64),
            "source_index": torch.tensor(source, dtype=torch.int64),
        }


@dataclass(frozen=True, slots=True)
class CorpusSplit:
    training: np.ndarray
    validation: np.ndarray
    validation_fraction: float
    seed: int
    fingerprint: str

    def metadata(self) -> dict[str, Any]:
        return {
            "validation_fraction": self.validation_fraction,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "training_samples": int(len(self.training)),
            "validation_samples": int(len(self.validation)),
        }


def stratified_corpus_split(
    corpus: MorphologyCorpus,
    *,
    validation_fraction: float = 0.08,
    seed: int = 0x5A17,
) -> CorpusSplit:
    training, validation = split_indices(
        corpus.morphologies,
        corpus.subtypes,
        corpus.roles,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    if len(training) == 0 or len(validation) == 0:
        raise ValueError("Stratified split produced an empty partition.")
    digest = hashlib.sha256()
    digest.update(b"multifield-stratified-split-v1\0")
    digest.update(np.asarray([validation_fraction], dtype=np.float64).tobytes())
    digest.update(np.asarray([seed], dtype=np.uint64).tobytes())
    digest.update(training.tobytes())
    digest.update(validation.tobytes())
    return CorpusSplit(
        training=training,
        validation=validation,
        validation_fraction=float(validation_fraction),
        seed=int(seed),
        fingerprint=digest.hexdigest(),
    )


def _index_chunks(indices: np.ndarray, chunk_size: int = 256) -> Iterator[np.ndarray]:
    for start in range(0, len(indices), chunk_size):
        yield indices[start : start + chunk_size]


def compute_class_weights(
    corpus: MorphologyCorpus,
    indices: np.ndarray,
    *,
    minimum: float = 0.25,
    maximum: float = 5.0,
) -> dict[str, Tensor]:
    if minimum <= 0.0 or maximum < minimum:
        raise ValueError("Class-weight bounds must be positive and ordered.")
    arrays: Mapping[str, tuple[np.ndarray, int]] = {
        "part": (corpus.part_owner, len(PART_OWNER_NAMES)),
        "material": (corpus.material, len(MATERIAL_NAMES)),
        "emission": (corpus.emission_level, len(EMISSION_LEVEL_NAMES)),
    }
    result: dict[str, Tensor] = {}
    for name, (values, vocabulary_size) in arrays.items():
        counts = np.zeros(vocabulary_size, dtype=np.int64)
        for chunk in _index_chunks(np.asarray(indices, dtype=np.int64)):
            counts += np.bincount(
                values[chunk].reshape(-1), minlength=vocabulary_size
            )[:vocabulary_size]
        present = counts > 0
        if not present.all():
            missing = np.flatnonzero(~present).tolist()
            raise ValueError(f"Training split is missing {name} classes {missing}.")
        frequency = counts.astype(np.float64) / float(counts.sum())
        weights = 1.0 / np.sqrt(np.maximum(frequency, 1.0e-12))
        weights /= weights.mean()
        weights = np.clip(weights, minimum, maximum).astype(np.float32)
        result[name] = torch.from_numpy(weights)
    return result


def compute_legal_tuples(
    corpus: MorphologyCorpus, indices: np.ndarray
) -> np.ndarray:
    """Collect train-only aligned field triples without materializing N*H*W*3."""
    material_count = len(MATERIAL_NAMES)
    emission_count = len(EMISSION_LEVEL_NAMES)
    code_count = len(PART_OWNER_NAMES) * material_count * emission_count
    observed = np.zeros(code_count, dtype=np.bool_)
    for chunk in _index_chunks(np.asarray(indices, dtype=np.int64), chunk_size=128):
        codes = (
            corpus.part_owner[chunk].astype(np.int32) * material_count * emission_count
            + corpus.material[chunk].astype(np.int32) * emission_count
            + corpus.emission_level[chunk].astype(np.int32)
        )
        observed[np.unique(codes)] = True
    codes = np.flatnonzero(observed)
    part = codes // (material_count * emission_count)
    remainder = codes % (material_count * emission_count)
    material = remainder // emission_count
    emission = remainder % emission_count
    return np.stack((part, material, emission), axis=1).astype(np.uint8)


def legal_tuple_fingerprint(legal_tuples: np.ndarray) -> str:
    values = np.ascontiguousarray(legal_tuples, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("legal_tuples must have shape [count, 3].")
    return hashlib.sha256(values.tobytes()).hexdigest()


def select_condition_bank(
    corpus: MorphologyCorpus,
    indices: np.ndarray,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    """Choose a deterministic validation bank that maximizes condition coverage."""
    candidates = np.asarray(indices, dtype=np.int64)
    if candidates.ndim != 1 or len(candidates) == 0:
        raise ValueError("Condition-bank candidates must be a non-empty vector.")
    if count < 0:
        raise ValueError("Condition-bank count cannot be negative.")
    if count == 0:
        return np.empty((0,), dtype=np.int64)
    count = min(count, len(candidates))
    order = np.random.default_rng(seed).permutation(len(candidates))
    remaining = candidates[order].tolist()
    selected: list[int] = []
    morphology_counts = np.zeros(len(FAMILIES), dtype=np.int32)
    subtype_counts = np.zeros(len(SUBTYPE_NAMES), dtype=np.int32)
    role_counts = np.zeros(len(ROLE_NAMES), dtype=np.int32)
    while len(selected) < count:
        best_position = 0
        best_score = float("-inf")
        for position, source in enumerate(remaining):
            morphology = int(corpus.morphologies[source])
            subtype = int(corpus.subtypes[source])
            role = int(corpus.roles[source])
            score = (
                (1000.0 if morphology_counts[morphology] == 0 else 0.0)
                + (1000.0 if role_counts[role] == 0 else 0.0)
                + (160.0 if subtype_counts[subtype] == 0 else 0.0)
                - 8.0 * morphology_counts[morphology]
                - 8.0 * role_counts[role]
                - 2.0 * subtype_counts[subtype]
            )
            if score > best_score:
                best_position = position
                best_score = score
        source = remaining.pop(best_position)
        selected.append(source)
        morphology_counts[int(corpus.morphologies[source])] += 1
        subtype_counts[int(corpus.subtypes[source])] += 1
        role_counts[int(corpus.roles[source])] += 1
    return np.asarray(selected, dtype=np.int64)
