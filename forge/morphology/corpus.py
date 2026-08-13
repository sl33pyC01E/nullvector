from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from tqdm import tqdm

from ..config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT
from ..safety import require_disk_floor, write_json_atomic
from .constants import (
    CANVAS_SIZE,
    EMISSION_LEVEL_NAMES,
    FAMILIES,
    GUIDE_CHANNEL_NAMES,
    GENOME_VERSION,
    MATERIAL_NAMES,
    PART_OWNER_NAMES,
    RENDERER_VERSION,
    ROLE_NAMES,
    SEMANTIC_FORMAT,
    SUBTYPE_NAMES,
)
from .contract import assert_valid_specimen
from .genome import genome_from_seed, mix32
from .render import render_specimen


CORPUS_FORMAT = "nullvector-morphology-training-corpus-v2"
SPLIT_VERSION = "exact-subtype-role-stratified-v2"
GUIDE_STORAGE_DTYPE = np.float16
STRATUM_COUNT = len(SUBTYPE_NAMES) * len(ROLE_NAMES)
MINIMUM_CORPUS_COUNT = STRATUM_COUNT * 2
CORPUS_SOURCE_FILES = (
    "forge/morphology/constants.py",
    "forge/morphology/genome.py",
    "forge/morphology/render.py",
    "forge/morphology/fields.py",
    "forge/morphology/contract.py",
)


def corpus_seed(base_seed: int, index: int) -> int:
    return mix32(base_seed ^ mix32((index + 1) * 0x9E3779B1))


def corpus_path(count: int, base_seed: int) -> Path:
    return DATA_DIR / f"morphology_{count}_{base_seed & 0xFFFFFFFF:08x}.npz"


def corpus_source_hash(root: Path = PROJECT_ROOT) -> str:
    digest = hashlib.sha256()
    for relative in CORPUS_SOURCE_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((Path(root) / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _planned_bytes(count: int) -> int:
    per_sample = (
        len(GUIDE_CHANNEL_NAMES)
        * CANVAS_SIZE
        * CANVAS_SIZE
        * np.dtype(GUIDE_STORAGE_DTYPE).itemsize
        + 3 * CANVAS_SIZE * CANVAS_SIZE
        + 24 * 4
        + 32
    )
    # Build arrays plus compressed temporary and final publication headroom.
    return count * per_sample * 2 + 256 * 1024 * 1024


def corpus_stratum(index: int) -> tuple[int, int, int]:
    """Return an exactly balanced ``(family, subtype, role)`` assignment.

    Every contiguous block of 160 samples contains each legal subtype/role
    pair exactly once.  This removes chance coverage gaps while leaving all
    continuous and topological traits driven by the independent sample seed.
    """
    if index < 0:
        raise ValueError("Corpus indices must be nonnegative.")
    stratum = index % STRATUM_COUNT
    subtype = stratum // len(ROLE_NAMES)
    role = stratum % len(ROLE_NAMES)
    family = subtype // 4
    return family, subtype, role


def build_morphology_corpus(
    path: Path,
    count: int,
    base_seed: int,
    *,
    force: bool = False,
) -> Path:
    if count < MINIMUM_CORPUS_COUNT:
        raise ValueError(
            f"Corpus needs at least {MINIMUM_CORPUS_COUNT} samples so every "
            "subtype/role stratum has both training and validation coverage."
        )
    path = Path(path).resolve()
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as payload:
            if (
                str(payload["format"][0]) == CORPUS_FORMAT
                and int(payload["base_seed"][0]) == (base_seed & 0xFFFFFFFF)
                and int(payload["guide"].shape[0]) == count
                and "corpus_source_sha256" in payload.files
                and str(payload["corpus_source_sha256"][0]) == corpus_source_hash()
            ):
                return path
    require_disk_floor(path, planned_bytes=_planned_bytes(count))
    path.parent.mkdir(parents=True, exist_ok=True)

    guide = np.empty(
        (count, len(GUIDE_CHANNEL_NAMES), CANVAS_SIZE, CANVAS_SIZE),
        dtype=GUIDE_STORAGE_DTYPE,
    )
    part_owner = np.empty((count, CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    material = np.empty_like(part_owner)
    emission = np.empty_like(part_owner)
    genes = np.empty((count, 24), dtype=np.float32)
    morphologies = np.empty((count,), dtype=np.uint8)
    subtypes = np.empty((count,), dtype=np.uint8)
    roles = np.empty((count,), dtype=np.uint8)
    seeds = np.empty((count,), dtype=np.uint32)
    semantic_hashes = np.empty((count,), dtype="U64")
    training_hashes = np.empty((count,), dtype="U64")

    for index in tqdm(range(count), desc="constructing morphology corpus"):
        family, subtype, role = corpus_stratum(index)
        seed = corpus_seed(base_seed, index)
        genome = genome_from_seed(seed, family)
        genome = replace(
            genome,
            silhouette_variant=subtype % 4,
            subtype_id=subtype,
            role_id=role,
        )
        genome.validate()
        specimen = render_specimen(genome)
        assert_valid_specimen(specimen)
        fields = specimen.training_fields()
        guide[index] = fields.guide
        part_owner[index] = fields.part_owner
        material[index] = fields.material
        emission[index] = fields.emission_level
        genes[index] = fields.genes
        morphologies[index] = fields.morphology_index
        subtypes[index] = fields.subtype_id
        roles[index] = fields.role_id
        seeds[index] = seed
        semantic_hashes[index] = specimen.manifest["hashes"]["semantic_sha256"]
        training_hashes[index] = fields.arrays_hash()

    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        format=np.asarray([CORPUS_FORMAT]),
        split_version=np.asarray([SPLIT_VERSION]),
        corpus_source_sha256=np.asarray([corpus_source_hash()]),
        genome_version=np.asarray([GENOME_VERSION]),
        renderer_version=np.asarray([RENDERER_VERSION]),
        semantic_format=np.asarray([SEMANTIC_FORMAT]),
        base_seed=np.asarray([base_seed & 0xFFFFFFFF], dtype=np.uint32),
        guide_storage_dtype=np.asarray([np.dtype(GUIDE_STORAGE_DTYPE).name]),
        guide=guide,
        part_owner=part_owner,
        material=material,
        emission_level=emission,
        genes=genes,
        morphologies=morphologies,
        subtypes=subtypes,
        roles=roles,
        seeds=seeds,
        semantic_sha256=semantic_hashes,
        training_arrays_sha256=training_hashes,
        guide_channel_names=np.asarray(GUIDE_CHANNEL_NAMES),
        part_owner_names=np.asarray(PART_OWNER_NAMES),
        material_names=np.asarray(MATERIAL_NAMES),
        emission_level_names=np.asarray(EMISSION_LEVEL_NAMES),
        family_names=np.asarray(FAMILIES),
        subtype_names=np.asarray(SUBTYPE_NAMES),
        role_names=np.asarray(ROLE_NAMES),
    )
    os.replace(temporary, path)
    return path


def split_indices(
    morphologies: np.ndarray,
    subtypes: np.ndarray,
    roles: np.ndarray,
    *,
    validation_fraction: float = 0.08,
    seed: int = 0x5A17,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between zero and one half.")
    if not (morphologies.shape == subtypes.shape == roles.shape):
        raise ValueError("Split labels must have identical shapes.")
    if morphologies.ndim != 1:
        raise ValueError("Split labels must be one-dimensional.")
    if len(morphologies) == 0:
        raise ValueError("Cannot split an empty corpus.")
    strata = np.stack((morphologies, subtypes, roles), axis=1)
    validation: list[int] = []
    training: list[int] = []
    for key in sorted(set(map(tuple, strata.tolist()))):
        indices = np.flatnonzero(np.all(strata == key, axis=1))
        generator = np.random.default_rng(
            mix32(seed ^ int(key[0]) * 0x9E37 ^ int(key[1]) * 0x85EB ^ int(key[2]))
        )
        indices = generator.permutation(indices)
        validation_count = max(1, int(round(len(indices) * validation_fraction)))
        if validation_count >= len(indices):
            validation_count = max(0, len(indices) - 1)
        validation.extend(map(int, indices[:validation_count]))
        training.extend(map(int, indices[validation_count:]))
    return (
        np.asarray(sorted(training), dtype=np.int64),
        np.asarray(sorted(validation), dtype=np.int64),
    )


def validate_corpus(path: Path, *, replay_samples: int = 160) -> list[str]:
    """Validate archive structure and replay a deterministic stratified sample."""
    if replay_samples < 0:
        raise ValueError("replay_samples must be nonnegative.")
    errors: list[str] = []
    required = {
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
    }
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(required - set(payload.files))
        if missing:
            return [f"archive keys are missing: {missing}"]
        if str(payload["format"][0]) != CORPUS_FORMAT:
            errors.append("corpus format is unsupported")
        if str(payload["split_version"][0]) != SPLIT_VERSION:
            errors.append("split version is unsupported")
        if str(payload["genome_version"][0]) != GENOME_VERSION:
            errors.append("genome version disagrees with the active renderer")
        if str(payload["renderer_version"][0]) != RENDERER_VERSION:
            errors.append("renderer version disagrees with the active renderer")
        if str(payload["semantic_format"][0]) != SEMANTIC_FORMAT:
            errors.append("semantic format disagrees with the active renderer")
        if str(payload["corpus_source_sha256"][0]) != corpus_source_hash():
            errors.append("corpus source hash disagrees with the active source tree")

        array_names = (
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
        )
        # NPZ members are compressed streams rather than memory maps. Cache each
        # member exactly once so replay does not re-inflate a multi-gigabyte
        # archive for every audited sample.
        arrays = {name: payload[name] for name in array_names}
        guide = arrays["guide"]
        count = int(guide.shape[0]) if guide.ndim else 0
        expected_spatial = (count, CANVAS_SIZE, CANVAS_SIZE)
        expected_shapes = {
            "guide": (count, len(GUIDE_CHANNEL_NAMES), CANVAS_SIZE, CANVAS_SIZE),
            "part_owner": expected_spatial,
            "material": expected_spatial,
            "emission_level": expected_spatial,
            "genes": (count, 24),
            "morphologies": (count,),
            "subtypes": (count,),
            "roles": (count,),
            "seeds": (count,),
            "semantic_sha256": (count,),
            "training_arrays_sha256": (count,),
        }
        for name, shape in expected_shapes.items():
            if arrays[name].shape != shape:
                errors.append(f"{name} shape is {arrays[name].shape}, expected {shape}")
        if guide.dtype != GUIDE_STORAGE_DTYPE:
            errors.append(f"guide dtype is {guide.dtype}, expected {GUIDE_STORAGE_DTYPE}")
        for name in ("part_owner", "material", "emission_level"):
            if arrays[name].dtype != np.uint8:
                errors.append(f"{name} dtype must be uint8")
        if arrays["genes"].dtype != np.float32:
            errors.append("genes dtype must be float32")
        if any(arrays[name].dtype != np.uint8 for name in ("morphologies", "subtypes", "roles")):
            errors.append("condition labels must be uint8")
        if arrays["seeds"].dtype != np.uint32:
            errors.append("seeds dtype must be uint32")

        if count < MINIMUM_CORPUS_COUNT:
            errors.append("corpus is too small for train/validation stratum coverage")
        if not np.isfinite(guide).all() or not ((guide >= 0).all() and (guide <= 1).all()):
            errors.append("guide values must be finite and stay in [0, 1]")
        genes = arrays["genes"]
        if not np.isfinite(genes).all() or not ((genes >= 0).all() and (genes <= 1).all()):
            errors.append("gene values must be finite and stay in [0, 1]")
        categorical = (
            ("part_owner", len(PART_OWNER_NAMES)),
            ("material", len(MATERIAL_NAMES)),
            ("emission_level", len(EMISSION_LEVEL_NAMES)),
        )
        for name, vocabulary_size in categorical:
            values = arrays[name]
            if values.size and int(values.max()) >= vocabulary_size:
                errors.append(f"{name} contains values outside its vocabulary")

        part = arrays["part_owner"]
        material = arrays["material"]
        emission = arrays["emission_level"]
        if not np.array_equal(part == 0, material == 0):
            errors.append("part/material background masks are not aligned")
        if bool(((part == 0) & (emission > 0)).any()):
            errors.append("emission escapes the owned sprite field")

        expected_labels = np.asarray(
            [corpus_stratum(index) for index in range(count)], dtype=np.uint8
        )
        for column, name in enumerate(("morphologies", "subtypes", "roles")):
            if not np.array_equal(arrays[name], expected_labels[:, column]):
                errors.append(f"{name} are not in canonical corpus order")
        base_seed = int(payload["base_seed"][0])
        expected_seeds = np.fromiter(
            (corpus_seed(base_seed, index) for index in range(count)),
            dtype=np.uint32,
            count=count,
        )
        if not np.array_equal(arrays["seeds"], expected_seeds):
            errors.append("sample seeds do not match the counter-based corpus stream")
        if len(np.unique(arrays["seeds"])) != count:
            errors.append("sample seeds are not unique")

        if replay_samples and count:
            replay_indices = np.unique(
                np.linspace(0, count - 1, min(replay_samples, count), dtype=np.int64)
            )
            for index_value in replay_indices:
                index = int(index_value)
                family, subtype, role = corpus_stratum(index)
                genome = genome_from_seed(int(arrays["seeds"][index]), family)
                genome = replace(
                    genome,
                    silhouette_variant=subtype % 4,
                    subtype_id=subtype,
                    role_id=role,
                )
                specimen = render_specimen(genome)
                fields = specimen.training_fields()
                checks = (
                    np.array_equal(guide[index], fields.guide.astype(GUIDE_STORAGE_DTYPE)),
                    np.array_equal(part[index], fields.part_owner),
                    np.array_equal(material[index], fields.material),
                    np.array_equal(emission[index], fields.emission_level),
                    np.array_equal(genes[index], fields.genes),
                    str(arrays["semantic_sha256"][index])
                    == specimen.manifest["hashes"]["semantic_sha256"],
                    str(arrays["training_arrays_sha256"][index]) == fields.arrays_hash(),
                )
                if not all(checks):
                    errors.append(f"sample {index} does not replay exactly")
                    break
    return errors


def inspect_corpus(path: Path) -> dict[str, object]:
    path = Path(path)
    digest_state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest_state.update(block)
    digest = digest_state.hexdigest()
    with np.load(path, allow_pickle=False) as payload:
        training, validation = split_indices(
            payload["morphologies"], payload["subtypes"], payload["roles"]
        )
        report = {
            "format": str(payload["format"][0]),
            "path": str(path.resolve()),
            "file_sha256": digest,
            "bytes": path.stat().st_size,
            "samples": int(payload["guide"].shape[0]),
            "training_samples": int(len(training)),
            "validation_samples": int(len(validation)),
            "guide_shape": list(payload["guide"].shape),
            "guide_storage_dtype": str(payload["guide"].dtype),
            "part_owner_shape": list(payload["part_owner"].shape),
            "family_counts": {
                name: int((payload["morphologies"] == index).sum())
                for index, name in enumerate(FAMILIES)
            },
            "subtype_coverage": int(len(np.unique(payload["subtypes"]))),
            "role_coverage": int(len(np.unique(payload["roles"]))),
            "stratum_count_min": int(
                min(
                    np.count_nonzero(
                        (payload["subtypes"] == subtype) & (payload["roles"] == role)
                    )
                    for subtype in range(len(SUBTYPE_NAMES))
                    for role in range(len(ROLE_NAMES))
                )
            ),
            "stratum_count_max": int(
                max(
                    np.count_nonzero(
                        (payload["subtypes"] == subtype) & (payload["roles"] == role)
                    )
                    for subtype in range(len(SUBTYPE_NAMES))
                    for role in range(len(ROLE_NAMES))
                )
            ),
            "part_owner_coverage": sorted(
                map(int, np.unique(payload["part_owner"]).tolist())
            ),
            "material_coverage": sorted(
                map(int, np.unique(payload["material"]).tolist())
            ),
            "emission_coverage": sorted(
                map(int, np.unique(payload["emission_level"]).tolist())
            ),
        }
    validation_errors = validate_corpus(path)
    report["valid"] = not validation_errors
    report["validation_errors"] = validation_errors
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a stratified 48px multi-family neural morphology corpus."
    )
    parser.add_argument("--count", type=int, default=32_768)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x4D4F5250)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=OUTPUT_DIR / "morphology_corpus_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.output or corpus_path(args.count, args.seed)
    started = time.perf_counter()
    path = build_morphology_corpus(
        destination, args.count, args.seed, force=args.force
    )
    report = inspect_corpus(path)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
