from __future__ import annotations

import gc
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from ..morphology import FAMILIES
from ..multifield_data import (
    MorphologyCorpus,
    compute_legal_tuples,
    legal_tuple_fingerprint,
    stratified_corpus_split,
)
from ..multifield_style.source import PROJECT_ROOT, load_generation_bank
from ..neural_rig_bridge import BindingRejected, bind_raw_sample_archive
from ..neural_rig_bridge.hashing import aligned_fields_hash
from .model import NeuralMotionCandidate, NeuralMotionSource, SelectedNeuralIdentity


EXPECTED_SAMPLE_COUNT = 80
EXPECTED_PER_FAMILY = 16
CENSUS_FORMAT = "nullvector-neural-rig-binding-census-v1"
CENSUS_CATEGORY_ORDER = (
    "anchor_on_background",
    "plant_topology",
    "required_owner_absence",
    "safety_margin",
)
CENSUS_EXPECTED_REJECTIONS = {
    "anchor_on_background": 3,
    "plant_topology": 1,
    "required_owner_absence": 3,
    "safety_margin": 3,
}


def _sha256_file(path: Path, *, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_record_path(root: Path, record: Mapping[str, Any], label: str) -> Path:
    relative = record.get("path")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{label} path must be a nonempty POSIX relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{label} path is unsafe: {relative!r}")
    path = (Path(root).resolve() / Path(*pure.parts)).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"{label} escapes its generation root")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if type(record.get("bytes")) is not int or path.stat().st_size != record["bytes"]:
        raise ValueError(f"{label} byte count mismatch")
    if _sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label} SHA-256 mismatch")
    return path


def _load_legal_tuple_contract(bank: Any) -> tuple[Path, Mapping[str, Any], np.ndarray]:
    provenance = bank.manifest["provenance"]
    corpus_record = provenance["corpus"]
    corpus_path = Path(corpus_record["path"]).resolve()
    if not corpus_path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Neural motion corpus provenance must stay under project root")
    if not corpus_path.is_file() or corpus_path.is_symlink():
        raise ValueError("Neural motion corpus must be a regular non-symlink file")
    if corpus_path.stat().st_size != corpus_record["file_bytes"]:
        raise ValueError("Neural motion corpus byte count mismatch")
    if _sha256_file(corpus_path) != corpus_record["file_sha256"]:
        raise ValueError("Neural motion corpus SHA-256 mismatch")
    corpus = MorphologyCorpus.load(corpus_path, verify_hash=True, verify_source=True)
    metadata = corpus.metadata()
    for name in (
        "format",
        "file_sha256",
        "file_bytes",
        "base_seed",
        "split_version",
        "corpus_source_sha256",
        "genome_version",
        "renderer_version",
        "semantic_format",
        "samples",
        "image_size",
        "guide_channels",
        "gene_dim",
        "vocabulary",
    ):
        if metadata[name] != corpus_record[name]:
            raise ValueError(f"Neural motion corpus provenance mismatch: {name}")
    split_record = provenance["split"]
    split = stratified_corpus_split(
        corpus,
        validation_fraction=float(split_record["validation_fraction"]),
        seed=int(split_record["seed"]),
    )
    if split.metadata() != split_record:
        raise ValueError("Neural motion corpus split provenance mismatch")
    legal = compute_legal_tuples(corpus, split.training)
    fingerprint = legal_tuple_fingerprint(legal)
    if fingerprint != provenance["legal_tuple_fingerprint"]:
        raise ValueError("Neural motion legal tuple provenance mismatch")
    contract = {
        "corpus_sha256": corpus.file_sha256,
        "corpus_bytes": corpus.path.stat().st_size,
        "corpus_source_sha256": corpus.corpus_source_sha256,
        "split_fingerprint": split.fingerprint,
        "split_seed": split.seed,
        "validation_fraction": split.validation_fraction,
        "legal_tuple_fingerprint": fingerprint,
    }
    del corpus
    gc.collect()
    return corpus_path, contract, np.ascontiguousarray(legal, dtype=np.uint8)


def load_neural_motion_source(generation_manifest: Path) -> NeuralMotionSource:
    bank = load_generation_bank(generation_manifest)
    if len(bank.samples) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("Neural motion source must be the final 80-sample production bank")
    grid = bank.manifest["grid"]
    if (
        grid.get("mode") != "stratified"
        or grid.get("covers_all_families") is not True
        or grid.get("covers_all_subtypes") is not True
        or grid.get("covers_all_roles") is not True
        or grid.get("covers_all_family_role_pairs") is not True
    ):
        raise ValueError("Neural motion source bank lacks complete condition coverage")
    corpus_path, legal_contract, legal_tuples = _load_legal_tuple_contract(bank)
    entries = list(bank.manifest["samples"])
    if len(entries) != len(bank.samples):
        raise ValueError("Neural motion source sample accounting mismatch")
    candidates: dict[str, list[NeuralMotionCandidate]] = {family: [] for family in FAMILIES}
    for sample, entry in zip(bank.samples, entries, strict=True):
        loaded_condition = sample.condition.as_dict()
        if any(entry["condition"].get(name) != value for name, value in loaded_condition.items()):
            raise ValueError("Neural motion condition changed after generation-bank load")
        if (
            entry["raw_fields_sha256"] != entry["compiled_fields_sha256"]
            or entry["postprocess"]["changed_pixels"] != 0
            or entry["postprocess"]["changed_fraction"] != 0.0
            or entry["raw_validation"]["tuples"]["legal_tuple_count"] != len(legal_tuples)
        ):
            raise ValueError("Neural motion requires byte-identical raw and compiled fields")
        raw_record = entry["raw_manifest"]
        raw_manifest_path = _safe_record_path(bank.root, raw_record, "raw manifest")
        raw_archive_path = bank.root / "raw" / f"{sample.condition.sample_id}.npz"
        if not raw_archive_path.is_file() or raw_archive_path.is_symlink():
            raise ValueError("Neural motion raw archive is missing or unsafe")
        candidate = NeuralMotionCandidate(
            sample=sample,
            source_entry=entry,
            raw_archive_path=raw_archive_path.resolve(),
            raw_manifest_path=raw_manifest_path,
        )
        candidates[sample.condition.morphology_name].append(candidate)
    if any(len(values) != EXPECTED_PER_FAMILY for values in candidates.values()):
        raise ValueError("Neural motion production source must have 16 candidates per family")
    return NeuralMotionSource(
        bank=bank,
        corpus_path=corpus_path,
        corpus_sha256=str(legal_contract["corpus_sha256"]),
        corpus_bytes=int(legal_contract["corpus_bytes"]),
        corpus_source_sha256=str(legal_contract["corpus_source_sha256"]),
        split_fingerprint=str(legal_contract["split_fingerprint"]),
        split_seed=int(legal_contract["split_seed"]),
        validation_fraction=float(legal_contract["validation_fraction"]),
        legal_tuples=legal_tuples,
        legal_tuple_fingerprint=str(legal_contract["legal_tuple_fingerprint"]),
        candidates_by_family={family: tuple(values) for family, values in candidates.items()},
    )


def bind_candidate(
    source: NeuralMotionSource,
    candidate: NeuralMotionCandidate,
) -> SelectedNeuralIdentity:
    binding = bind_raw_sample_archive(
        candidate.raw_archive_path,
        raw_manifest_path=candidate.raw_manifest_path,
        legal_tuples=source.legal_tuples,
    )
    sample = candidate.sample
    if (
        binding.sample_id != sample.condition.sample_id
        or binding.family != sample.condition.morphology_name
        or binding.family_id != sample.condition.morphology_id
        or binding.subtype_id != sample.condition.subtype_id
        or binding.role_id != sample.condition.role_id
        or binding.raw_fields_sha256 != sample.raw_fields_sha256
        or aligned_fields_hash(
            binding.part_owner,
            binding.material,
            binding.emission_level,
        )
        != sample.fields.aligned_sha256
        or not np.array_equal(binding.part_owner, sample.fields.part)
        or not np.array_equal(binding.material, sample.fields.material)
        or not np.array_equal(binding.emission_level, sample.fields.emission)
    ):
        raise BindingRejected(["neural binding disagrees with immutable generation sample"])
    return SelectedNeuralIdentity(candidate=candidate, binding=binding)


def _rejection_category(reason: str) -> str:
    if "lands on neural background" in reason:
        return "anchor_on_background"
    if "plantlike physical rig" in reason and "components" in reason:
        return "plant_topology"
    if "required " in reason and " owner is absent" in reason:
        return "required_owner_absence"
    if "foreground pixels violate the 3-pixel margin" in reason:
        return "safety_margin"
    raise ValueError(f"Unclassified neural binding rejection: {reason}")


def compute_binding_census(source: NeuralMotionSource) -> dict[str, Any]:
    """Bind every immutable production sample and record every rejection."""

    rejections: list[dict[str, Any]] = []
    family_counts: list[dict[str, Any]] = []
    bindable_total = 0
    for family in FAMILIES:
        bindable = 0
        family_rejections = 0
        for ordinal, candidate in enumerate(source.candidates_by_family[family]):
            try:
                bind_candidate(source, candidate)
                bindable += 1
                bindable_total += 1
            except BindingRejected as error:
                reason = str(error)
                rejections.append(
                    {
                        "family": family,
                        "candidate_ordinal_within_family": ordinal,
                        "sample_id": candidate.sample.condition.sample_id,
                        "category": _rejection_category(reason),
                        "reason": reason,
                    }
                )
                family_rejections += 1
        family_counts.append(
            {
                "family": family,
                "sample_count": EXPECTED_PER_FAMILY,
                "bindable_count": bindable,
                "rejected_count": family_rejections,
            }
        )
    category_counts = {
        category: sum(record["category"] == category for record in rejections)
        for category in CENSUS_CATEGORY_ORDER
    }
    if (
        bindable_total != 70
        or len(rejections) != 10
        or category_counts != CENSUS_EXPECTED_REJECTIONS
    ):
        raise ValueError(
            "Neural production binding census drifted: "
            f"bindable={bindable_total} rejected={len(rejections)} categories={category_counts}"
        )
    return {
        "format": CENSUS_FORMAT,
        "scope": "all-80-immutable-production-samples",
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "bindable_count": bindable_total,
        "rejected_count": len(rejections),
        "family_counts": family_counts,
        "rejection_categories": [
            {"category": category, "count": category_counts[category]}
            for category in CENSUS_CATEGORY_ORDER
        ],
        "rejections": rejections,
        "animation_bank_scope": {
            "selected_identity_count": len(FAMILIES),
            "all_80_animated": False,
            "policy": "first-bank-ordered-full-matrix-valid-identity-per-family-v1",
            "binding_census_does_not_imply_animation": True,
        },
    }
