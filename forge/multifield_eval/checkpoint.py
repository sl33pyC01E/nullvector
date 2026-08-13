from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from ..morphology.constants import (
    EMISSION_LEVEL_NAMES,
    FAMILIES,
    GUIDE_CHANNEL_NAMES,
    MATERIAL_NAMES,
    PART_OWNER_NAMES,
    ROLE_NAMES,
    SUBTYPE_NAMES,
)
from ..multifield_data import (
    GuidePolicy,
    MorphologyCorpus,
    compute_legal_tuples,
    legal_tuple_fingerprint,
    stratified_corpus_split,
)
from ..multifield_diffusion import MultiFieldSpriteDiffusion, MultiFieldVocabulary
from ..provenance import canonical_state_dict_hash
from ..safety import require_disk_floor
from ..train_multifield import CHECKPOINT_FORMAT, training_source_hash


EVALUATION_CHECKPOINT_CONTRACT = "nullvector-multifield-evaluation-checkpoint-v1"
REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "format",
        "ema_model",
        "architecture",
        "config",
        "corpus",
        "split",
        "legal_tuples",
        "legal_tuple_fingerprint",
        "guide_policy",
        "canonical_ema_hash",
        "training_source_hash",
        "fixed_validation",
        "class_weights",
        "next_epoch",
        "global_step",
    }
)


class CheckpointNotReady(RuntimeError):
    """A checkpoint has not yet been atomically published or is incomplete."""

    def __init__(self, checkpoint: Path, reason: str):
        self.checkpoint = Path(checkpoint).resolve()
        self.reason = reason
        super().__init__(f"Checkpoint incomplete: {reason} ({self.checkpoint})")

    def report(self) -> dict[str, Any]:
        return {
            "status": "checkpoint_incomplete",
            "checkpoint": str(self.checkpoint),
            "reason": self.reason,
        }


class CheckpointProvenanceError(ValueError):
    """A published checkpoint disagrees with its active inference contract."""


@dataclass(slots=True)
class LoadedMultiFieldCheckpoint:
    path: Path
    checkpoint_sha256: str
    payload: Mapping[str, Any]
    corpus: MorphologyCorpus
    training_indices: np.ndarray
    validation_indices: np.ndarray
    guide_policy: GuidePolicy
    legal_tuples: np.ndarray
    model: MultiFieldSpriteDiffusion
    device: torch.device
    precision: str

    @property
    def training_complete(self) -> bool:
        return int(self.payload["next_epoch"]) >= int(self.payload["config"]["epochs"])

    def provenance(self) -> dict[str, Any]:
        evaluation_environment = {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_type": self.device.type,
            "device": str(self.device),
            "precision": self.precision,
            "gpu_name": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else None
            ),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
        return {
            "contract": EVALUATION_CHECKPOINT_CONTRACT,
            "checkpoint": str(self.path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "canonical_ema_hash": str(self.payload["canonical_ema_hash"]),
            "training_source_hash": str(self.payload["training_source_hash"]),
            "corpus": self.corpus.metadata(),
            "split": dict(self.payload["split"]),
            "guide_policy": self.guide_policy.metadata(),
            "legal_tuple_fingerprint": str(
                self.payload["legal_tuple_fingerprint"]
            ),
            "architecture": dict(self.payload["architecture"]),
            "published_next_epoch": int(self.payload["next_epoch"]),
            "global_step": int(self.payload["global_step"]),
            "training_complete": self.training_complete,
            "device": str(self.device),
            "precision": self.precision,
            "checkpoint_environment": dict(self.payload.get("environment", {})),
            "evaluation_environment": evaluation_environment,
        }


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _publication_signature(path: Path) -> tuple[int, int, int]:
    status = path.stat()
    return int(status.st_size), int(status.st_mtime_ns), int(status.st_ino)


def _load_stable_payload(
    path: Path, *, attempts: int = 3
) -> tuple[Mapping[str, Any], str]:
    """Load only an atomically published file whose identity remains stable.

    The trainer writes hidden ``.tmp`` files and publishes with ``os.replace``.
    This loader never follows those temporary files.  A before/after identity
    check also prevents pairing a payload with the hash of the next epoch when
    publication occurs while evaluation starts.
    """

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if path.name.startswith(".") or path.suffix.lower() == ".tmp" or ".tmp" in path.name:
        raise CheckpointNotReady(path, "temporary checkpoint paths are never readable")
    if not path.is_file():
        temporary = list(path.parent.glob(f".{path.name}.*.tmp")) if path.parent.exists() else []
        reason = "no published checkpoint exists yet"
        if temporary:
            reason += "; the trainer currently has an unpublished temporary file"
        raise CheckpointNotReady(path, reason)
    if path.stat().st_size <= 0:
        raise CheckpointNotReady(path, "published path is empty")

    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            before = _publication_signature(path)
            digest = _sha256_file(path)
            after_hash = _publication_signature(path)
            if before != after_hash:
                raise OSError("checkpoint changed while hashing")
            with path.open("rb") as handle:
                opened = os.fstat(handle.fileno())
                opened_signature = (
                    int(opened.st_size),
                    int(opened.st_mtime_ns),
                    int(opened.st_ino),
                )
                if opened_signature != before:
                    raise OSError("checkpoint changed before opening")
                payload = torch.load(handle, map_location="cpu", weights_only=False)
            after_load = _publication_signature(path)
            if after_load != before:
                raise OSError("checkpoint changed while loading")
            if not isinstance(payload, Mapping):
                raise CheckpointNotReady(path, "published payload is not a mapping")
            return payload, digest
        except CheckpointNotReady:
            raise
        except Exception as error:
            if isinstance(error, MemoryError):
                raise
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.05 * (attempt + 1))
    raise CheckpointNotReady(
        path,
        "published file could not be read as a stable complete checkpoint: "
        f"{type(last_error).__name__}: {last_error}",
    )


def snapshot_published_checkpoint(
    source: Path, destination: Path, *, attempts: int = 3
) -> dict[str, Any]:
    """Copy one immutable published checkpoint without following a live rewrite."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if destination.exists():
        raise FileExistsError(
            f"Checkpoint snapshots are immutable and will not be overwritten: {destination}"
        )
    if source.name.startswith(".") or source.suffix.lower() == ".tmp" or ".tmp" in source.name:
        raise CheckpointNotReady(source, "temporary checkpoint paths cannot be snapshotted")
    if not source.is_file() or source.stat().st_size <= 0:
        raise CheckpointNotReady(source, "no complete published checkpoint exists to snapshot")
    destination.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(destination.parent, planned_bytes=source.stat().st_size * 2)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    last_reason = "checkpoint changed during snapshot"
    try:
        for attempt in range(attempts):
            temporary.unlink(missing_ok=True)
            before = _publication_signature(source)
            digest = hashlib.sha256()
            copied = 0
            try:
                with source.open("rb") as reader, temporary.open("xb") as writer:
                    opened = os.fstat(reader.fileno())
                    opened_signature = (
                        int(opened.st_size),
                        int(opened.st_mtime_ns),
                        int(opened.st_ino),
                    )
                    if opened_signature != before:
                        raise OSError("source changed before snapshot opened")
                    for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                        writer.write(block)
                        digest.update(block)
                        copied += len(block)
                    writer.flush()
                    os.fsync(writer.fileno())
                if _publication_signature(source) != before:
                    raise OSError("source changed while snapshotting")
                if copied != before[0] or temporary.stat().st_size != before[0]:
                    raise OSError("snapshot byte count differs from source")
                os.replace(temporary, destination)
                return {
                    "status": "snapshotted",
                    "source": str(source),
                    "destination": str(destination),
                    "bytes": copied,
                    "sha256": digest.hexdigest(),
                }
            except OSError as error:
                last_reason = str(error)
                temporary.unlink(missing_ok=True)
                if attempt + 1 < attempts:
                    time.sleep(0.05 * (attempt + 1))
        raise CheckpointNotReady(
            source, f"could not capture a stable published snapshot: {last_reason}"
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _resolve_precision(requested: str, device: torch.device) -> str:
    if requested == "auto":
        return (
            "bf16"
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else "fp32"
        )
    if requested not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be auto, fp32, bf16, or fp16")
    if requested == "fp16" and device.type != "cuda":
        raise ValueError("fp16 evaluation is only supported on CUDA")
    return requested


def model_from_multifield_architecture(
    architecture: Mapping[str, Any],
) -> MultiFieldSpriteDiffusion:
    expected_name = "graph-guided-multifield-categorical-diffusion-unet"
    if architecture.get("name") != expected_name:
        raise CheckpointProvenanceError(
            f"Unsupported architecture {architecture.get('name')!r}"
        )
    vocabulary = architecture.get("vocabulary")
    if not isinstance(vocabulary, Mapping):
        raise CheckpointProvenanceError("architecture vocabulary is missing")
    return MultiFieldSpriteDiffusion(
        vocabulary=MultiFieldVocabulary(
            part_count=int(vocabulary["part_count"]),
            material_count=int(vocabulary["material_count"]),
            emission_count=int(vocabulary["emission_count"]),
        ),
        morphology_count=int(architecture["morphology_count"]),
        subtype_count=int(architecture["subtype_count"]),
        role_count=int(architecture["role_count"]),
        gene_dim=int(architecture["gene_dim"]),
        guide_channels=int(architecture["guide_channels"]),
        steps=int(architecture["steps"]),
        width=int(architecture["width"]),
        image_size=int(architecture["image_size"]),
    )


def _verify_architecture(architecture: Mapping[str, Any]) -> None:
    expected = {
        "part_count": len(PART_OWNER_NAMES),
        "material_count": len(MATERIAL_NAMES),
        "emission_count": len(EMISSION_LEVEL_NAMES),
    }
    if dict(architecture.get("vocabulary", {})) != expected:
        raise CheckpointProvenanceError(
            "checkpoint vocabulary disagrees with active morphology constants"
        )
    counts = {
        "morphology_count": len(FAMILIES),
        "subtype_count": len(SUBTYPE_NAMES),
        "role_count": len(ROLE_NAMES),
        "guide_channels": len(GUIDE_CHANNEL_NAMES),
    }
    for name, expected_value in counts.items():
        if int(architecture.get(name, -1)) != expected_value:
            raise CheckpointProvenanceError(
                f"architecture {name} disagrees with the active contract"
            )


def _guide_policy_from_checkpoint(payload: Mapping[str, Any]) -> GuidePolicy:
    config = payload["config"]
    policy = GuidePolicy(
        name=str(config["guide_policy"]),
        thicken_radius=int(config["guide_thicken_radius"]),
        training_channel_dropout=float(config["guide_channel_dropout"]),
        training_jitter_pixels=int(config["guide_jitter_pixels"]),
    )
    if policy.metadata() != dict(payload["guide_policy"]):
        raise CheckpointProvenanceError(
            "checkpoint guide policy metadata disagrees with its training config"
        )
    return policy


def load_multifield_checkpoint(
    checkpoint_path: Path,
    *,
    corpus_path: Path | None = None,
    device: str = "auto",
    precision: str = "auto",
) -> LoadedMultiFieldCheckpoint:
    """Strictly load a published EMA checkpoint and all inference provenance."""

    path = Path(checkpoint_path).resolve()
    payload, checkpoint_digest = _load_stable_payload(path)
    missing = sorted(REQUIRED_CHECKPOINT_KEYS.difference(payload))
    if missing:
        raise CheckpointNotReady(path, f"payload is missing required keys: {missing}")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointProvenanceError(
            f"unsupported checkpoint format {payload.get('format')!r}"
        )
    if int(payload["next_epoch"]) < 1 or int(payload["global_step"]) < 1:
        raise CheckpointNotReady(path, "checkpoint has no completed training epoch")

    architecture = payload["architecture"]
    if not isinstance(architecture, Mapping):
        raise CheckpointProvenanceError("checkpoint architecture is missing")
    _verify_architecture(architecture)

    active_source_hash = training_source_hash()
    if str(payload["training_source_hash"]) != active_source_hash:
        raise CheckpointProvenanceError(
            "training source hash differs from the active source tree"
        )

    corpus_location = (
        Path(corpus_path).resolve()
        if corpus_path is not None
        else Path(str(payload["corpus"]["path"])).resolve()
    )
    corpus = MorphologyCorpus.load(
        corpus_location, verify_hash=True, verify_source=True
    )
    if corpus.file_sha256 != str(payload["corpus"]["file_sha256"]):
        raise CheckpointProvenanceError(
            "checkpoint corpus hash differs from the supplied corpus"
        )
    for name in (
        "format",
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
        if corpus.metadata()[name] != payload["corpus"][name]:
            raise CheckpointProvenanceError(
                f"checkpoint corpus metadata field {name!r} disagrees"
            )

    config = payload["config"]
    split = stratified_corpus_split(
        corpus,
        validation_fraction=float(config["validation_fraction"]),
        seed=int(config["split_seed"]),
    )
    if split.metadata() != dict(payload["split"]):
        raise CheckpointProvenanceError(
            "checkpoint split fingerprint or population disagrees"
        )
    if int(architecture["image_size"]) != corpus.image_size:
        raise CheckpointProvenanceError(
            "checkpoint architecture image size differs from the corpus"
        )
    if int(architecture["gene_dim"]) != int(corpus.genes.shape[1]):
        raise CheckpointProvenanceError(
            "checkpoint architecture gene dimension differs from the corpus"
        )
    legal_tuples = np.asarray(payload["legal_tuples"], dtype=np.uint8)
    active_legal_tuples = compute_legal_tuples(corpus, split.training)
    if not np.array_equal(legal_tuples, active_legal_tuples):
        raise CheckpointProvenanceError(
            "checkpoint legal tuple table differs from the train split"
        )
    if legal_tuple_fingerprint(legal_tuples) != str(
        payload["legal_tuple_fingerprint"]
    ):
        raise CheckpointProvenanceError("checkpoint legal tuple fingerprint is invalid")

    expected_weight_counts = {
        "part": len(PART_OWNER_NAMES),
        "material": len(MATERIAL_NAMES),
        "emission": len(EMISSION_LEVEL_NAMES),
    }
    if set(payload["class_weights"]) != set(expected_weight_counts):
        raise CheckpointProvenanceError("checkpoint class-weight fields are incomplete")
    for name, count in expected_weight_counts.items():
        values = torch.as_tensor(payload["class_weights"][name]).detach().cpu()
        if values.shape != (count,) or not torch.isfinite(values).all() or bool((values <= 0).any()):
            raise CheckpointProvenanceError(
                f"checkpoint {name} class weights are invalid"
            )

    fixed = payload["fixed_validation"]
    if not isinstance(fixed, Mapping):
        raise CheckpointProvenanceError("checkpoint fixed validation contract is missing")
    for seed_name in ("full_mask_seed", "generation_seed"):
        seed = int(fixed[seed_name])
        if not 0 <= seed <= 0x7FFFFFFFFFFFFFFF:
            raise CheckpointProvenanceError(
                f"checkpoint {seed_name} is outside signed int64"
            )
    validation_lookup = set(map(int, split.validation))
    fixed_indices = list(map(int, fixed["generation_source_indices"]))
    if any(index not in validation_lookup for index in fixed_indices):
        raise CheckpointProvenanceError(
            "checkpoint fixed generation bank escapes the validation split"
        )

    guide_policy = _guide_policy_from_checkpoint(payload)
    model = model_from_multifield_architecture(architecture)
    ema_state = payload["ema_model"]
    if canonical_state_dict_hash(ema_state) != str(payload["canonical_ema_hash"]):
        raise CheckpointProvenanceError("canonical EMA hash is invalid")
    try:
        model.load_state_dict(ema_state, strict=True)
    except RuntimeError as error:
        raise CheckpointProvenanceError(
            f"EMA tensors do not match the recorded architecture: {error}"
        ) from error

    target_device = _resolve_device(device)
    target_precision = _resolve_precision(precision, target_device)
    if bool(config.get("deterministic", True)):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    model.to(target_device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Inference does not retain raw/EMA duplicate tensors, optimizer moments,
    # scheduler/scaler state, or RNG snapshots after strict verification.  A
    # production checkpoint can otherwise pin several hundred MiB of avoidable
    # CPU memory alongside the instantiated EMA model.
    retained_payload = dict(payload)
    for heavy_key in (
        "model",
        "ema_model",
        "optimizer",
        "scheduler",
        "scaler",
        "rng_state",
    ):
        retained_payload.pop(heavy_key, None)

    return LoadedMultiFieldCheckpoint(
        path=path,
        checkpoint_sha256=checkpoint_digest,
        payload=retained_payload,
        corpus=corpus,
        training_indices=split.training,
        validation_indices=split.validation,
        guide_policy=guide_policy,
        legal_tuples=legal_tuples,
        model=model,
        device=target_device,
        precision=target_precision,
    )
