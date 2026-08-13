from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import uuid
import zipfile

import numpy as np

from ..map_decorator.catalog import CATALOG_SHA256, validate_decoration_fields
from ..map_decorator.contract import FEATURE_CONTRACT_SHA256
from ..map_decorator.features import EncodedFeatures, validate_encoded_features
from ..map_decorator.hashing import named_arrays_sha256
from ..maps.io import CANONICAL_ARRAY_HASH_ALGORITHM, array_digest
from ..maps.model import MapData
from ..safety import require_disk_floor
from .checkpoint import file_sha256, inspect_checkpoint_provenance
from .contract import HEAD_CLASS_COUNTS, HEAD_NAMES, MODEL_CONTRACT_SHA256
from .sampling import DecorationPrediction, SamplerConfig


PREDICTION_FORMAT_VERSION = "1.0.0"
FIELDS_FILE = "raw_fields.npz"
MANIFEST_FILE = "prediction_manifest.json"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_FIELDS_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_FIELDS_MEMBER_BYTES = 2 * 1024 * 1024


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, planned_bytes=len(payload) + 1024 * 1024)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(buffer, **{name: arrays[name] for name in HEAD_NAMES})
    return buffer.getvalue()


def _validate_npz_container(path: Path, expected_shape: tuple[int, int]) -> None:
    if path.stat().st_size > MAX_FIELDS_ARCHIVE_BYTES:
        raise ValueError("Raw field archive exceeds the bounded foundation contract.")
    expected = {f"{name}.npy" for name in HEAD_NAMES}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            if len(members) != len(expected) or {member.filename for member in members} != expected:
                raise ValueError("Raw field NPZ container members are incomplete or unexpected.")
            for member in members:
                if (
                    member.file_size > MAX_FIELDS_MEMBER_BYTES
                    or member.compress_size > MAX_FIELDS_ARCHIVE_BYTES
                    or member.is_dir()
                ):
                    raise ValueError("Raw field NPZ member exceeds the bounded foundation contract.")
                with archive.open(member, "r") as handle:
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                    else:
                        raise ValueError(
                            f"Raw field writer contract requires NPY v1.0, observed {version!r}."
                        )
                    expected_bytes = int(np.prod(expected_shape, dtype=np.int64))
                    if dtype != np.dtype(np.uint8) or tuple(shape) != expected_shape or fortran:
                        raise ValueError("Raw field NPY header violates dtype/shape/layout bounds.")
                    if member.file_size != handle.tell() + expected_bytes:
                        raise ValueError("Raw field NPY member size disagrees with its bounded header.")
    except zipfile.BadZipFile as error:
        raise ValueError("Raw field artifact is not a valid NPZ container.") from error


def write_prediction_pack(
    output_root: Path,
    prediction: DecorationPrediction,
    data: MapData,
    encoded: EncodedFeatures,
    *,
    checkpoint_path: Path,
    sampler_config: SamplerConfig,
    source_sha256: str,
    corpus_sha256: str,
    ema_tensor_sha256: str,
) -> Path:
    if not prediction.report.get("passed"):
        raise ValueError("Only a validated raw prediction can be published.")
    arrays = prediction.arrays()
    field_hash = named_arrays_sha256(arrays)
    if prediction.report.get("field_sha256") != field_hash:
        raise ValueError("Prediction report field hash does not match the supplied arrays.")
    step_hashes = prediction.report.get("step_field_sha256")
    if (
        not isinstance(step_hashes, list)
        or len(step_hashes) != sampler_config.steps
        or not all(_is_sha256(value) for value in step_hashes)
    ):
        raise ValueError("Prediction report refinement-step hashes are incomplete or malformed.")
    generation_seed = prediction.report.get("generation_seed")
    if (
        isinstance(generation_seed, bool)
        or not isinstance(generation_seed, int)
        or not 0 <= generation_seed < (1 << 63)
    ):
        raise ValueError("Prediction report generation seed violates the sampler contract.")
    checkpoint_sidecar_path = Path(checkpoint_path).with_suffix(
        Path(checkpoint_path).suffix + ".json"
    )
    checkpoint_sidecar = json.loads(checkpoint_sidecar_path.read_text(encoding="utf-8"))
    if (
        not isinstance(checkpoint_sidecar, dict)
        or checkpoint_sidecar.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        or checkpoint_sidecar.get("source_sha256") != source_sha256
        or checkpoint_sidecar.get("corpus_sha256") != corpus_sha256
        or checkpoint_sidecar.get("ema_tensor_sha256") != ema_tensor_sha256
        or not all(
            _is_sha256(value) for value in (source_sha256, corpus_sha256, ema_tensor_sha256)
        )
    ):
        raise ValueError("Checkpoint/EMA provenance does not match its verified sidecar.")
    feature_report = validate_encoded_features(
        data,
        encoded,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
    )
    if not feature_report["passed"]:
        raise ValueError(f"Encoded features failed pre-publication validation: {feature_report}")
    validation = validate_decoration_fields(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
        **arrays,
    )
    if not validation["passed"]:
        raise ValueError(f"Prediction failed pre-publication legality: {validation}")
    output_root = Path(output_root)
    final = output_root / f"{data.map_id}-seed{generation_seed:016x}"
    if final.exists():
        raise FileExistsError(f"Prediction pack already exists and will not be overwritten: {final}")
    payload = _npz_bytes(arrays)
    require_disk_floor(output_root, planned_bytes=len(payload) + 4 * 1024 * 1024)
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{final.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        fields_path = staging / FIELDS_FILE
        _atomic_bytes(fields_path, payload)
        topology_masks = {
            "protected_backbone": data.protected_backbone,
            "required_clearance": data.required_clearance,
            "decoration_forbidden": data.decoration_forbidden,
        }
        manifest: dict[str, object] = {
            "format_version": PREDICTION_FORMAT_VERSION,
            "source_map_id": data.map_id,
            "source_semantic_hash_algorithm": CANONICAL_ARRAY_HASH_ALGORITHM,
            "source_semantic_sha256": array_digest(data.arrays()),
            "topology_masks_sha256": named_arrays_sha256(topology_masks),
            "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
            "feature_tensor_sha256": encoded.tensor_sha256,
            "catalog_sha256": CATALOG_SHA256,
            "model_contract_sha256": MODEL_CONTRACT_SHA256,
            "checkpoint": {
                "file": Path(checkpoint_path).name,
                "sha256": file_sha256(checkpoint_path),
                "source_sha256": source_sha256,
                "corpus_sha256": corpus_sha256,
                "weight_set": "ema",
                "ema_tensor_sha256": ema_tensor_sha256,
            },
            "sampler": {
                "name": "parallel-masked-categorical-refinement-v1",
                "steps": sampler_config.steps,
                "temperature": sampler_config.temperature,
                "generation_seed": generation_seed,
            },
            "artifacts": {
                "raw_fields": {
                    "file": FIELDS_FILE,
                    "sha256": file_sha256(fields_path),
                    "field_sha256": field_hash,
                    "arrays": {
                        name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                        for name, value in arrays.items()
                    },
                }
            },
            "validation": validation,
            "refinement_step_field_sha256": step_hashes,
        }
        _atomic_bytes(staging / MANIFEST_FILE, _json_bytes(manifest))
        os.replace(staging, final)
    except BaseException:
        # Leave the unique staging directory as a diagnostic; never overwrite a candidate.
        raise
    validate_prediction_pack(final, data=data, encoded=encoded, checkpoint_path=checkpoint_path)
    return final


def validate_prediction_pack(
    path: Path,
    *,
    data: MapData,
    encoded: EncodedFeatures,
    checkpoint_path: Path,
) -> dict[str, object]:
    path = Path(path)
    manifest_path = path / MANIFEST_FILE
    if not manifest_path.is_file() or manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Prediction manifest is missing or exceeds its size bound.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"Prediction manifest cannot be read: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("Prediction manifest root must be an object.")
    failures: list[str] = []
    expected_manifest_keys = {
        "format_version",
        "source_map_id",
        "source_semantic_hash_algorithm",
        "source_semantic_sha256",
        "topology_masks_sha256",
        "feature_contract_sha256",
        "feature_tensor_sha256",
        "catalog_sha256",
        "model_contract_sha256",
        "checkpoint",
        "sampler",
        "artifacts",
        "validation",
        "refinement_step_field_sha256",
    }
    if set(manifest) != expected_manifest_keys:
        failures.append("manifest_members")
    if manifest.get("format_version") != PREDICTION_FORMAT_VERSION:
        failures.append("format_version")
    if manifest.get("source_map_id") != data.map_id:
        failures.append("source_map_id")
    if manifest.get("source_semantic_sha256") != array_digest(data.arrays()):
        failures.append("source_semantic_sha256")
    if manifest.get("source_semantic_hash_algorithm") != CANONICAL_ARRAY_HASH_ALGORITHM:
        failures.append("source_semantic_hash_algorithm")
    topology_masks = {
        "protected_backbone": data.protected_backbone,
        "required_clearance": data.required_clearance,
        "decoration_forbidden": data.decoration_forbidden,
    }
    if manifest.get("topology_masks_sha256") != named_arrays_sha256(topology_masks):
        failures.append("topology_masks_sha256")
    if manifest.get("feature_contract_sha256") != FEATURE_CONTRACT_SHA256:
        failures.append("feature_contract_sha256")
    if manifest.get("feature_tensor_sha256") != encoded.tensor_sha256:
        failures.append("feature_tensor_sha256")
    if manifest.get("catalog_sha256") != CATALOG_SHA256:
        failures.append("catalog_sha256")
    if manifest.get("model_contract_sha256") != MODEL_CONTRACT_SHA256:
        failures.append("model_contract_sha256")
    feature_report = validate_encoded_features(
        data,
        encoded,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
    )
    if not feature_report["passed"]:
        failures.append("encoded_feature_validation")
    checkpoint = manifest.get("checkpoint", {})
    try:
        checkpoint_provenance = inspect_checkpoint_provenance(checkpoint_path)
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError):
        checkpoint_provenance = {}
        failures.append("checkpoint_provenance")
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {
            "file",
            "sha256",
            "source_sha256",
            "corpus_sha256",
            "weight_set",
            "ema_tensor_sha256",
        }
        or checkpoint.get("file") != Path(checkpoint_path).name
        or checkpoint.get("sha256") != file_sha256(checkpoint_path)
        or checkpoint.get("sha256") != checkpoint_provenance.get("checkpoint_sha256")
        or checkpoint.get("source_sha256") != checkpoint_provenance.get("source_sha256")
        or checkpoint.get("corpus_sha256") != checkpoint_provenance.get("corpus_sha256")
        or checkpoint.get("weight_set") != "ema"
        or checkpoint.get("ema_tensor_sha256") != checkpoint_provenance.get("ema_tensor_sha256")
    ):
        failures.append("checkpoint_sha256")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        failures.append("artifacts")
        artifacts = {}
    artifact = artifacts.get("raw_fields")
    if not isinstance(artifact, dict):
        failures.append("raw_fields_artifact")
        artifact = {}
    fields_path = path / FIELDS_FILE
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {"file", "sha256", "field_sha256", "arrays"}
        or artifact.get("file") != FIELDS_FILE
        or not fields_path.is_file()
        or artifact.get("sha256") != file_sha256(fields_path)
    ):
        failures.append("raw_fields_file_sha256")
        arrays: dict[str, np.ndarray] = {}
    else:
        _validate_npz_container(fields_path, data.shape)
        with np.load(fields_path, allow_pickle=False) as archive:
            if set(archive.files) != set(HEAD_NAMES):
                failures.append("raw_fields_members")
                arrays = {}
            else:
                arrays = {name: np.ascontiguousarray(archive[name]) for name in HEAD_NAMES}
        for name, classes in HEAD_CLASS_COUNTS.items():
            value = arrays.get(name)
            if (
                value is None
                or value.dtype != np.uint8
                or value.shape != data.shape
                or not bool((value < classes).all())
            ):
                failures.append(f"raw_fields.{name}")
            elif (
                not isinstance(artifact.get("arrays"), dict)
                or set(artifact["arrays"]) != set(HEAD_NAMES)
                or artifact["arrays"].get(name)
                != {"dtype": str(value.dtype), "shape": list(value.shape)}
            ):
                failures.append(f"raw_fields_descriptor.{name}")
        if arrays and artifact.get("field_sha256") != named_arrays_sha256(arrays):
            failures.append("raw_fields_field_sha256")
    legality: dict[str, object] | None = None
    if arrays and not any(item.startswith("raw_fields.") for item in failures):
        legality = validate_decoration_fields(
            data,
            protected_backbone=data.protected_backbone,
            required_clearance=data.required_clearance,
            decoration_forbidden=data.decoration_forbidden,
            **arrays,
        )
        if not legality["passed"]:
            failures.append("legality")
        if manifest.get("validation") != legality:
            failures.append("recorded_validation")
    sampler = manifest.get("sampler")
    if (
        not isinstance(sampler, dict)
        or set(sampler) != {"name", "steps", "temperature", "generation_seed"}
        or sampler.get("name") != "parallel-masked-categorical-refinement-v1"
        or isinstance(sampler.get("steps"), bool)
        or not isinstance(sampler.get("steps"), int)
        or not 1 <= sampler.get("steps", 0) <= 32
        or not isinstance(sampler.get("temperature"), (int, float))
        or isinstance(sampler.get("temperature"), bool)
        or not 0.05 <= sampler.get("temperature", 0.0) <= 4.0
        or isinstance(sampler.get("generation_seed"), bool)
        or not isinstance(sampler.get("generation_seed"), int)
        or not 0 <= sampler.get("generation_seed", -1) < (1 << 63)
    ):
        failures.append("sampler")
    step_hashes = manifest.get("refinement_step_field_sha256")
    if (
        not isinstance(step_hashes, list)
        or not isinstance(sampler, dict)
        or len(step_hashes) != sampler.get("steps")
        or not all(_is_sha256(value) for value in step_hashes)
    ):
        failures.append("refinement_step_hashes")
    report = {"passed": not failures, "failures": failures, "legality": legality}
    if failures:
        raise ValueError(f"Prediction pack validation failed closed: {failures}")
    return report
