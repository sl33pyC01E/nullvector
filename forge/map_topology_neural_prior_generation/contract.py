from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

import jsonschema

from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-neural-map-topology-prior-generation-bank/1.0.0"
CASE_FORMAT: Final[str] = "nullvector-neural-map-topology-prior-generation-case/1.0.0"
REPLAY_FORMAT: Final[str] = "nullvector-neural-map-topology-prior-generation-replay/1.0.0"
PROPOSAL_SOURCE: Final[str] = "masked_prior_ema_fully_masked_seeded_parallel"
PRIOR_CHECKPOINT_RELATIVE: Final[str] = "outputs/map_topology_neural_prior_training/calibration_500step_v1/checkpoint_final.pt"
PRIOR_CHECKPOINT_SHA256: Final[str] = "bb18b1c98474abe51e852a46ebe6c47de773814657d27302ec177ff495cd0475"
PRIOR_EMA_SHA256: Final[str] = "2987741e4d8d2b3fde339f7513566b9dd2d64283cfdfd237c9da30712c3f58e4"
PRIOR_TRAINING_SOURCE_SHA256: Final[str] = "b54ca3e2a863b9abba8731b8286f86c688e600c26f99e84d6ffa0829e42dc8b1"
CODEC_CHECKPOINT_RELATIVE: Final[str] = "outputs/map_topology_neural_production/calibration_500step_v2_hardened/checkpoint_final.pt"
CODEC_CHECKPOINT_SHA256: Final[str] = "536d7e54e1da9f35ca9200353774121a59da69d9ea12853a5271b89fe06bce64"
CODEC_EMA_SHA256: Final[str] = "0d90d210505fbda8fa3a319cc3a6d55ca1094252781159c4447f51bac6121d72"
CODEC_SOURCE_SHA256: Final[str] = "1fe97d977aaf0a21e2caa6c75a52ee9a0e519087b8c2c2c1dca7e86806253a50"
LATENT_CORPUS_MANIFEST_SHA256: Final[str] = "12ae282fe1d89f4b8f5c87d0d5acf1a8eddf7ab15cd2d32031a6bf7ba1cc3b96"
LATENT_CORPUS_IDENTITY_SHA256: Final[str] = "bbcce0606f12d04d53e15e50c16852a8ee3d0e7262146e4c85c5965cf10f4d56"
SOURCE_CORPUS_SHA256: Final[str] = "16ed5f3b1a661e2bfc2abe9e16c39e9b8caaecba81f50fed6658cc4f73cffab8"
SOURCE_CORPUS_MANIFEST_SHA256: Final[str] = "fd5ee2e88725262f23ef1943e34aad7f19c1b0886100f43298f93226de2ccbaf"
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/map_topology_neural_prior_generation/__init__.py",
    "forge/map_topology_neural_prior_generation/__main__.py",
    "forge/map_topology_neural_prior_generation/bank.py",
    "forge/map_topology_neural_prior_generation/contract.py",
    "forge/map_topology_neural_prior_generation/render.py",
    "forge/map_topology_neural_prior_generation/sampling.py",
    "shared/schema/map_topology_neural_prior_generation_bank.schema.json",
    "shared/schema/map_topology_neural_prior_generation_case.schema.json",
    "shared/schema/map_topology_neural_prior_generation_replay.schema.json",
)


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(root: Path = PROJECT_ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def generation_source_sha256(root: Path = PROJECT_ROOT) -> str:
    return hashlib.sha256(canonical_json_bytes(source_manifest(root))).hexdigest()


def stable_seed(*parts: object, bits: int = 63) -> int:
    if bits not in (63, 64):
        raise ValueError("Stable generation seeds support only 63 or 64 bits.")
    digest = hashlib.sha256(canonical_json_bytes(list(parts))).digest()
    value = int.from_bytes(digest[:8], "big")
    return value & ((1 << bits) - 1)


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} is not canonical SHA-256.")
    return value


def validate_schema(payload: dict[str, object], filename: str) -> None:
    schema_path = PROJECT_ROOT / "shared" / "schema" / filename
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    variants_per_condition: int = 2
    sampling_steps: int = 8
    temperature: float = 0.8
    top_k: int = 16
    base_seed: int = 0x465245454D415053
    maximum_workers: int = 2
    maximum_attempts: int = 3
    worker_timeout_seconds: int = 900
    contact_scale: int = 4

    def __post_init__(self) -> None:
        if type(self.variants_per_condition) is not int or not 1 <= self.variants_per_condition <= 8:
            raise ValueError("variants_per_condition must be in [1,8].")
        if type(self.sampling_steps) is not int or not 2 <= self.sampling_steps <= 32:
            raise ValueError("sampling_steps must be in [2,32].")
        if not math.isfinite(self.temperature) or not 0.05 <= self.temperature <= 4.0:
            raise ValueError("temperature must be finite and in [0.05,4].")
        if type(self.top_k) is not int or not 1 <= self.top_k <= 512:
            raise ValueError("top_k must be in [1,512].")
        if type(self.base_seed) is not int or not 0 <= self.base_seed < 1 << 63:
            raise ValueError("base_seed must be unsigned 63-bit.")
        if type(self.maximum_workers) is not int or not 1 <= self.maximum_workers <= 2:
            raise ValueError("maximum_workers must be one or two.")
        if type(self.maximum_attempts) is not int or not 1 <= self.maximum_attempts <= 3:
            raise ValueError("maximum_attempts must be in [1,3].")
        if type(self.worker_timeout_seconds) is not int or not 60 <= self.worker_timeout_seconds <= 3600:
            raise ValueError("worker_timeout_seconds must be in [60,3600].")
        if type(self.contact_scale) is not int or not 2 <= self.contact_scale <= 8:
            raise ValueError("contact_scale must be in [2,8].")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GenerationConfig":
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("Generation configuration member census drifted.")
        return cls(**payload)  # type: ignore[arg-type]


def authority_payload() -> dict[str, object]:
    return {
        "prior": {
            "checkpoint": PRIOR_CHECKPOINT_RELATIVE,
            "checkpoint_sha256": PRIOR_CHECKPOINT_SHA256,
            "ema_sha256": PRIOR_EMA_SHA256,
            "training_source_sha256": PRIOR_TRAINING_SOURCE_SHA256,
        },
        "codec": {
            "checkpoint": CODEC_CHECKPOINT_RELATIVE,
            "checkpoint_sha256": CODEC_CHECKPOINT_SHA256,
            "ema_sha256": CODEC_EMA_SHA256,
            "source_sha256": CODEC_SOURCE_SHA256,
        },
        "latent_corpus": {
            "manifest_file_sha256": LATENT_CORPUS_MANIFEST_SHA256,
            "identity_sha256": LATENT_CORPUS_IDENTITY_SHA256,
            "used_for_generation_targets": False,
        },
        "condition_source_corpus": {
            "corpus_sha256": SOURCE_CORPUS_SHA256,
            "manifest_file_sha256": SOURCE_CORPUS_MANIFEST_SHA256,
            "split": "test",
        },
    }
