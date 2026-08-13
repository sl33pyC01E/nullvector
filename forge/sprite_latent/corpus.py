from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence
import zipfile

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..morphology.constants import (
    CANVAS_SIZE,
    EMISSION_LEVEL_NAMES,
    FAMILIES,
    GENOME_VERSION,
    GUIDE_CHANNEL_NAMES,
    MATERIAL_NAMES,
    PART_OWNER_NAMES,
    RENDERER_VERSION,
    ROLE_NAMES,
    SEMANTIC_FORMAT,
    SUBTYPE_NAMES,
)
from ..morphology.corpus import (
    CORPUS_FORMAT,
    GUIDE_STORAGE_DTYPE,
    MINIMUM_CORPUS_COUNT,
    SPLIT_VERSION,
    corpus_seed,
    corpus_source_hash,
    corpus_stratum,
    split_indices,
)
from ..morphology.motion import allowed_training_field_tuples


MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 3 * 1024 * 1024 * 1024
FROZEN_PRODUCTION_CORPUS_SHA256 = (
    "77dc7313ca6411295bad883f483a6edf4be75016ebfd7c107d0f286d2cb1cd7b"
)
FROZEN_PRODUCTION_SAMPLE_COUNT = 32_768
FROZEN_PRODUCTION_BASE_SEED = 0x4D4F5250
FROZEN_PRODUCTION_SPLIT_FINGERPRINT = (
    "5e400872460dc527c01a2a301f006e761abd1621773c5f67b45568d68886007b"
)
FROZEN_PRODUCTION_LEGAL_TUPLE_FINGERPRINT = (
    "0b15074b76ca69ea9a93e0b73db7e5df0b242dc0ecc46c5e842342fb0378948d"
)
FROZEN_PRODUCTION_LEGAL_TUPLE_COUNT = 69
EXPECTED_KEYS = frozenset(
    {
        "format",
        "split_version",
        "corpus_source_sha256",
        "genome_version",
        "renderer_version",
        "semantic_format",
        "base_seed",
        "guide_storage_dtype",
        "guide",
        "part_owner",
        "material",
        "emission_level",
        "genes",
        "morphologies",
        "subtypes",
        "roles",
        "seeds",
        "semantic_sha256",
        "training_arrays_sha256",
        "guide_channel_names",
        "part_owner_names",
        "material_names",
        "emission_level_names",
        "family_names",
        "subtype_names",
        "role_names",
    }
)


def sha256_file(path: Path, *, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_npy_header(
    archive: zipfile.ZipFile, member: str
) -> tuple[tuple[int, ...], bool, np.dtype[Any], int]:
    with archive.open(member, "r") as handle:
        version = np.lib.format.read_magic(handle)
        shape, fortran_order, dtype = np.lib.format._read_array_header(  # type: ignore[attr-defined]
            handle,
            version,
            max_header_size=16 * 1024,
        )
        return tuple(map(int, shape)), bool(fortran_order), np.dtype(dtype), int(handle.tell())


def _validate_container(path: Path) -> int:
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Sprite semantic corpus exceeds the compressed size bound")
    expected_members = {f"{key}.npy" for key in EXPECTED_KEYS}
    with zipfile.ZipFile(path, "r") as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("Sprite semantic corpus contains duplicate ZIP members")
        if set(names) != expected_members:
            missing = sorted(expected_members.difference(names))
            extra = sorted(set(names).difference(expected_members))
            raise ValueError(f"Sprite semantic corpus member mismatch missing={missing} extra={extra}")
        total = 0
        for entry in entries:
            if PurePosixPath(entry.filename).name != entry.filename:
                raise ValueError("Sprite semantic corpus contains nested or unsafe members")
            if entry.flag_bits & 0x1:
                raise ValueError("Sprite semantic corpus cannot contain encrypted members")
            if entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("Sprite semantic corpus uses an unsupported compression type")
            if entry.file_size < 0 or entry.file_size > MAX_MEMBER_BYTES:
                raise ValueError("Sprite semantic corpus member exceeds its size bound")
            total += int(entry.file_size)
        if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("Sprite semantic corpus exceeds its total uncompressed size bound")

        headers = {
            name.removesuffix(".npy"): _read_npy_header(archive, name)
            for name in names
        }
        part_shape = headers["part_owner"][0]
        if len(part_shape) != 3 or part_shape[1:] != (CANVAS_SIZE, CANVAS_SIZE):
            raise ValueError("Sprite semantic corpus part_owner header is not N,48,48")
        count = int(part_shape[0])
        if count < MINIMUM_CORPUS_COUNT:
            raise ValueError("Sprite semantic corpus is too small for frozen stratum coverage")
        expected: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
            "guide": (
                (count, len(GUIDE_CHANNEL_NAMES), CANVAS_SIZE, CANVAS_SIZE),
                np.dtype(GUIDE_STORAGE_DTYPE),
            ),
            "part_owner": ((count, CANVAS_SIZE, CANVAS_SIZE), np.dtype(np.uint8)),
            "material": ((count, CANVAS_SIZE, CANVAS_SIZE), np.dtype(np.uint8)),
            "emission_level": ((count, CANVAS_SIZE, CANVAS_SIZE), np.dtype(np.uint8)),
            "genes": ((count, 24), np.dtype(np.float32)),
            "morphologies": ((count,), np.dtype(np.uint8)),
            "subtypes": ((count,), np.dtype(np.uint8)),
            "roles": ((count,), np.dtype(np.uint8)),
            "seeds": ((count,), np.dtype(np.uint32)),
            "semantic_sha256": ((count,), np.dtype("U64")),
            "training_arrays_sha256": ((count,), np.dtype("U64")),
            "base_seed": ((1,), np.dtype(np.uint32)),
        }
        vocabulary_lengths = {
            "guide_channel_names": len(GUIDE_CHANNEL_NAMES),
            "part_owner_names": len(PART_OWNER_NAMES),
            "material_names": len(MATERIAL_NAMES),
            "emission_level_names": len(EMISSION_LEVEL_NAMES),
            "family_names": len(FAMILIES),
            "subtype_names": len(SUBTYPE_NAMES),
            "role_names": len(ROLE_NAMES),
        }
        scalar_strings = {
            "format",
            "split_version",
            "corpus_source_sha256",
            "genome_version",
            "renderer_version",
            "semantic_format",
            "guide_storage_dtype",
        }
        info_by_name = {entry.filename.removesuffix(".npy"): entry for entry in entries}
        for name, (shape, dtype) in expected.items():
            observed_shape, fortran_order, observed_dtype, offset = headers[name]
            if observed_shape != shape or observed_dtype != dtype or fortran_order:
                raise ValueError(
                    f"Sprite semantic corpus {name} header mismatch: "
                    f"shape={observed_shape} dtype={observed_dtype} fortran={fortran_order}"
                )
            expected_member_bytes = offset + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if info_by_name[name].file_size != expected_member_bytes:
                raise ValueError(f"Sprite semantic corpus {name} NPY payload size is inconsistent")
        for name, length in vocabulary_lengths.items():
            shape, fortran_order, dtype, offset = headers[name]
            if shape != (length,) or fortran_order or dtype.kind != "U" or dtype.itemsize > 4096:
                raise ValueError(f"Sprite semantic corpus {name} vocabulary header is invalid")
            expected_member_bytes = offset + length * dtype.itemsize
            if info_by_name[name].file_size != expected_member_bytes:
                raise ValueError(f"Sprite semantic corpus {name} NPY payload size is inconsistent")
        for name in scalar_strings:
            shape, fortran_order, dtype, offset = headers[name]
            if shape != (1,) or fortran_order or dtype.kind != "U" or dtype.itemsize > 4096:
                raise ValueError(f"Sprite semantic corpus {name} scalar header is invalid")
            if info_by_name[name].file_size != offset + dtype.itemsize:
                raise ValueError(f"Sprite semantic corpus {name} NPY payload size is inconsistent")
        return count


def _scalar_string(values: np.ndarray, name: str) -> str:
    if values.shape != (1,) or values.dtype.kind != "U":
        raise ValueError(f"Corpus metadata {name} must be a one-element string")
    return str(values[0])


def _validate_sha_vector(values: np.ndarray, name: str, count: int) -> str:
    if values.shape != (count,) or values.dtype != np.dtype("U64"):
        raise ValueError(f"Corpus {name} must be a U64 vector with one value per sample")
    rendered = values.tolist()
    if any(len(value) != 64 or any(character not in "0123456789abcdef" for character in value) for value in rendered):
        raise ValueError(f"Corpus {name} contains a malformed SHA-256 value")
    if len(set(rendered)) != count:
        raise ValueError(f"Corpus {name} identities must be unique")
    digest = hashlib.sha256()
    digest.update(f"nullvector-{name}-vector-v1\0".encode("ascii"))
    for value in rendered:
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(slots=True)
class SemanticFieldCorpus:
    """Strict field-only view of the 48px corpus.

    The original archive also contains a 1.2 GiB guide tensor. The FSQ codec
    does not consume guides, so this loader verifies the entire NPZ container
    and provenance but decompresses only the aligned categorical fields and
    conditions. This keeps fresh-process segment memory bounded on Windows.
    """

    path: Path
    file_sha256: str
    format: str
    split_version: str
    corpus_source_sha256: str
    genome_version: str
    renderer_version: str
    semantic_format: str
    base_seed: int
    part_owner: np.ndarray
    material: np.ndarray
    emission_level: np.ndarray
    genes: np.ndarray
    morphologies: np.ndarray
    subtypes: np.ndarray
    roles: np.ndarray
    seeds: np.ndarray
    semantic_hash_vector_sha256: str
    training_hash_vector_sha256: str

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_file_sha256: str | None = None,
    ) -> "SemanticFieldCorpus":
        resolved = Path(path).resolve()
        if not resolved.is_file() or resolved.is_symlink():
            raise ValueError("Sprite semantic corpus must be a regular non-symlink file")
        header_count = _validate_container(resolved)
        file_sha256 = sha256_file(resolved)
        if expected_file_sha256 is not None and file_sha256 != expected_file_sha256:
            raise ValueError(
                "Sprite semantic corpus file SHA-256 does not match the frozen provenance"
            )
        with np.load(resolved, allow_pickle=False) as archive:
            if set(archive.files) != EXPECTED_KEYS:
                raise ValueError("Sprite semantic corpus array-key mismatch")
            vocabulary = {
                "guide_channel_names": GUIDE_CHANNEL_NAMES,
                "part_owner_names": PART_OWNER_NAMES,
                "material_names": MATERIAL_NAMES,
                "emission_level_names": EMISSION_LEVEL_NAMES,
                "family_names": FAMILIES,
                "subtype_names": SUBTYPE_NAMES,
                "role_names": ROLE_NAMES,
            }
            for name, expected in vocabulary.items():
                if tuple(map(str, archive[name].tolist())) != tuple(expected):
                    raise ValueError(f"Sprite semantic corpus {name} vocabulary drifted")
            base_seed_values = archive["base_seed"]
            if base_seed_values.shape != (1,) or base_seed_values.dtype != np.uint32:
                raise ValueError("Sprite semantic corpus base_seed must be uint32[1]")
            guide_storage_dtype = _scalar_string(
                archive["guide_storage_dtype"], "guide_storage_dtype"
            )
            if guide_storage_dtype != np.dtype(GUIDE_STORAGE_DTYPE).name:
                raise ValueError("Sprite semantic corpus guide storage dtype drifted")
            semantic_hash_vector_sha256 = _validate_sha_vector(
                archive["semantic_sha256"], "semantic_sha256", header_count
            )
            training_hash_vector_sha256 = _validate_sha_vector(
                archive["training_arrays_sha256"], "training_arrays_sha256", header_count
            )
            fields = {
                name: np.ascontiguousarray(archive[name])
                for name in (
                    "part_owner",
                    "material",
                    "emission_level",
                    "genes",
                    "morphologies",
                    "subtypes",
                    "roles",
                    "seeds",
                )
            }
            corpus = cls(
                path=resolved,
                file_sha256=file_sha256,
                format=_scalar_string(archive["format"], "format"),
                split_version=_scalar_string(archive["split_version"], "split_version"),
                corpus_source_sha256=_scalar_string(
                    archive["corpus_source_sha256"], "corpus_source_sha256"
                ),
                genome_version=_scalar_string(archive["genome_version"], "genome_version"),
                renderer_version=_scalar_string(
                    archive["renderer_version"], "renderer_version"
                ),
                semantic_format=_scalar_string(
                    archive["semantic_format"], "semantic_format"
                ),
                base_seed=int(archive["base_seed"][0]),
                semantic_hash_vector_sha256=semantic_hash_vector_sha256,
                training_hash_vector_sha256=training_hash_vector_sha256,
                **fields,
            )
        corpus.validate()
        return corpus

    @property
    def count(self) -> int:
        return int(self.part_owner.shape[0])

    @property
    def image_size(self) -> int:
        return int(self.part_owner.shape[-1])

    @property
    def loaded_array_bytes(self) -> int:
        return int(
            sum(
                getattr(self, name).nbytes
                for name in (
                    "part_owner",
                    "material",
                    "emission_level",
                    "genes",
                    "morphologies",
                    "subtypes",
                    "roles",
                    "seeds",
                )
            )
        )

    def validate(self) -> None:
        if self.format != CORPUS_FORMAT:
            raise ValueError(f"Unsupported sprite semantic corpus format: {self.format}")
        if self.split_version != SPLIT_VERSION:
            raise ValueError("Sprite semantic corpus split version is stale")
        if self.corpus_source_sha256 != corpus_source_hash():
            raise ValueError("Sprite semantic corpus source hash is stale")
        if self.genome_version != GENOME_VERSION:
            raise ValueError("Sprite semantic corpus genome version is stale")
        if self.renderer_version != RENDERER_VERSION:
            raise ValueError("Sprite semantic corpus renderer version is stale")
        if self.semantic_format != SEMANTIC_FORMAT:
            raise ValueError("Sprite semantic corpus semantic format is stale")
        count = self.count
        if count < MINIMUM_CORPUS_COUNT or self.part_owner.shape[1:] != (48, 48):
            raise ValueError("Sprite semantic corpus must contain native 48px fields")
        for name, values, classes in (
            ("part_owner", self.part_owner, len(PART_OWNER_NAMES)),
            ("material", self.material, len(MATERIAL_NAMES)),
            ("emission_level", self.emission_level, len(EMISSION_LEVEL_NAMES)),
        ):
            if values.shape != (count, 48, 48) or values.dtype != np.uint8:
                raise ValueError(f"Corpus {name} must be uint8 N,48,48")
            if int(values.max(initial=0)) >= classes:
                raise ValueError(f"Corpus {name} contains out-of-vocabulary values")
        background = self.part_owner == 0
        if not np.array_equal(background, self.material == 0):
            raise ValueError("Corpus part/material background masks are not exactly aligned")
        if np.any(self.emission_level[background] != 0):
            raise ValueError("Corpus background emission is not exactly zero")
        allowed_codes = np.zeros(
            len(PART_OWNER_NAMES) * len(MATERIAL_NAMES) * len(EMISSION_LEVEL_NAMES),
            dtype=np.bool_,
        )
        for part, material, emission in allowed_training_field_tuples():
            allowed_codes[
                part * len(MATERIAL_NAMES) * len(EMISSION_LEVEL_NAMES)
                + material * len(EMISSION_LEVEL_NAMES)
                + emission
            ] = True
        for chunk in _chunks(np.arange(count, dtype=np.int64)):
            codes = (
                self.part_owner[chunk].astype(np.int32)
                * len(MATERIAL_NAMES)
                * len(EMISSION_LEVEL_NAMES)
                + self.material[chunk].astype(np.int32) * len(EMISSION_LEVEL_NAMES)
                + self.emission_level[chunk].astype(np.int32)
            )
            if not bool(allowed_codes[codes].all()):
                raise ValueError("Corpus contains a tuple outside the versioned semantic allowlist")
        if self.genes.shape != (count, 24) or self.genes.dtype != np.float32:
            raise ValueError("Corpus genes must be float32 N,24")
        if not np.isfinite(self.genes).all() or float(self.genes.min()) < 0 or float(self.genes.max()) > 1:
            raise ValueError("Corpus genes must be finite and bounded [0,1]")
        for name, values, classes in (
            ("morphologies", self.morphologies, len(FAMILIES)),
            ("subtypes", self.subtypes, len(SUBTYPE_NAMES)),
            ("roles", self.roles, len(ROLE_NAMES)),
        ):
            if values.shape != (count,) or values.dtype != np.uint8:
                raise ValueError(f"Corpus {name} must be a uint8 vector")
            if int(values.min(initial=0)) < 0 or int(values.max(initial=0)) >= classes:
                raise ValueError(f"Corpus {name} contains out-of-vocabulary values")
        expected_labels = np.asarray(
            [corpus_stratum(index) for index in range(count)], dtype=np.uint8
        )
        for column, name in enumerate(("morphologies", "subtypes", "roles")):
            if not np.array_equal(getattr(self, name), expected_labels[:, column]):
                raise ValueError(f"Corpus {name} are not in canonical stratum order")
        if self.seeds.shape != (count,) or self.seeds.dtype != np.uint32:
            raise ValueError("Corpus seeds must be a uint32 vector")
        expected_seeds = np.fromiter(
            (corpus_seed(self.base_seed, index) for index in range(count)),
            dtype=np.uint32,
            count=count,
        )
        if not np.array_equal(self.seeds, expected_seeds):
            raise ValueError("Corpus seeds do not match the canonical counter stream")
        if len(np.unique(self.seeds)) != count:
            raise ValueError("Corpus sample identities are not unique")

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "file_bytes": self.path.stat().st_size,
            "format": self.format,
            "split_version": self.split_version,
            "corpus_source_sha256": self.corpus_source_sha256,
            "genome_version": self.genome_version,
            "renderer_version": self.renderer_version,
            "semantic_format": self.semantic_format,
            "base_seed": self.base_seed,
            "sample_count": self.count,
            "image_size": self.image_size,
            "loaded_array_bytes": self.loaded_array_bytes,
            "guide_loaded": False,
            "semantic_hash_vector_sha256": self.semantic_hash_vector_sha256,
            "training_hash_vector_sha256": self.training_hash_vector_sha256,
        }


@dataclass(frozen=True, slots=True)
class SemanticSplit:
    training: np.ndarray
    validation: np.ndarray
    validation_fraction: float
    seed: int
    fingerprint: str

    def metadata(self) -> dict[str, Any]:
        return {
            "training_samples": int(len(self.training)),
            "validation_samples": int(len(self.validation)),
            "validation_fraction": self.validation_fraction,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
        }


def stratified_split(
    corpus: SemanticFieldCorpus,
    *,
    validation_fraction: float = 0.08,
    seed: int = 0x5A17,
) -> SemanticSplit:
    training, validation = split_indices(
        corpus.morphologies,
        corpus.subtypes,
        corpus.roles,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    digest = hashlib.sha256()
    digest.update(b"multifield-stratified-split-v1\0")
    digest.update(np.asarray([validation_fraction], dtype=np.float64).tobytes())
    digest.update(np.asarray([seed], dtype=np.uint64).tobytes())
    digest.update(training.tobytes())
    digest.update(validation.tobytes())
    return SemanticSplit(
        training=training,
        validation=validation,
        validation_fraction=float(validation_fraction),
        seed=int(seed),
        fingerprint=digest.hexdigest(),
    )


class SemanticFieldDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, corpus: SemanticFieldCorpus, indices: Sequence[int] | np.ndarray):
        self.corpus = corpus
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or len(self.indices) == 0:
            raise ValueError("Sprite field dataset indices must be a non-empty vector")
        if int(self.indices.min()) < 0 or int(self.indices.max()) >= corpus.count:
            raise IndexError("Sprite field dataset indices are out of range")
        if len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("Sprite field dataset indices must be unique")

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        source = int(self.indices[index])
        return {
            "part": torch.from_numpy(self.corpus.part_owner[source]).long(),
            "material": torch.from_numpy(self.corpus.material[source]).long(),
            "emission": torch.from_numpy(self.corpus.emission_level[source]).long(),
            "genes": torch.from_numpy(self.corpus.genes[source]),
            "morphology": torch.tensor(int(self.corpus.morphologies[source]), dtype=torch.long),
            "subtype": torch.tensor(int(self.corpus.subtypes[source]), dtype=torch.long),
            "role": torch.tensor(int(self.corpus.roles[source]), dtype=torch.long),
            "seed": torch.tensor(int(self.corpus.seeds[source]), dtype=torch.int64),
            "source_index": torch.tensor(source, dtype=torch.int64),
        }


def _chunks(indices: np.ndarray, size: int = 128) -> Iterator[np.ndarray]:
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


def compute_legal_tuples(corpus: SemanticFieldCorpus, indices: np.ndarray) -> np.ndarray:
    code_count = len(PART_OWNER_NAMES) * len(MATERIAL_NAMES) * len(EMISSION_LEVEL_NAMES)
    observed = np.zeros(code_count, dtype=np.bool_)
    for chunk in _chunks(np.asarray(indices, dtype=np.int64)):
        codes = (
            corpus.part_owner[chunk].astype(np.int32)
            * len(MATERIAL_NAMES)
            * len(EMISSION_LEVEL_NAMES)
            + corpus.material[chunk].astype(np.int32) * len(EMISSION_LEVEL_NAMES)
            + corpus.emission_level[chunk].astype(np.int32)
        )
        observed[np.unique(codes)] = True
    codes = np.flatnonzero(observed)
    part = codes // (len(MATERIAL_NAMES) * len(EMISSION_LEVEL_NAMES))
    remainder = codes % (len(MATERIAL_NAMES) * len(EMISSION_LEVEL_NAMES))
    material = remainder // len(EMISSION_LEVEL_NAMES)
    emission = remainder % len(EMISSION_LEVEL_NAMES)
    return np.stack((part, material, emission), axis=1).astype(np.uint8)


def legal_tuple_fingerprint(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.uint8)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError("Legal tuples must have shape K,3")
    return hashlib.sha256(array.tobytes()).hexdigest()


def compute_class_weights(
    corpus: SemanticFieldCorpus,
    indices: np.ndarray,
    *,
    minimum: float = 0.25,
    maximum: float = 5.0,
) -> dict[str, Tensor]:
    if minimum <= 0 or maximum < minimum:
        raise ValueError("Class-weight bounds must be positive and ordered")
    result: dict[str, Tensor] = {}
    for name, values, classes in (
        ("part", corpus.part_owner, len(PART_OWNER_NAMES)),
        ("material", corpus.material, len(MATERIAL_NAMES)),
        ("emission", corpus.emission_level, len(EMISSION_LEVEL_NAMES)),
    ):
        counts = np.zeros(classes, dtype=np.int64)
        for chunk in _chunks(np.asarray(indices, dtype=np.int64)):
            counts += np.bincount(values[chunk].reshape(-1), minlength=classes)[:classes]
        if not np.all(counts > 0):
            raise ValueError(f"Training split omits {name} classes {np.flatnonzero(counts == 0).tolist()}")
        frequency = counts.astype(np.float64) / counts.sum()
        weights = 1.0 / np.sqrt(frequency)
        weights /= weights.mean()
        result[name] = torch.from_numpy(np.clip(weights, minimum, maximum).astype(np.float32))
    return result
