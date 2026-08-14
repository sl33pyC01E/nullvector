from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-organism-hierarchical-rectified-flow/1.0.0"
CHECKPOINT_FORMAT: Final[str] = "nullvector-organism-hierarchical-rectified-flow-checkpoint/1.0.0"
CORPUS_FORMAT: Final[str] = "nullvector-organism-hierarchical-latent-corpus/1.0.0"
VAE_SOURCE_SHA256: Final[str] = "79f7a2c1f30cc11a85f5fd3f57d8f40a7820373b64cfb9a0c2725c27f4f5bff8"
VAE_MANIFEST_SHA256: Final[str] = "e162b5e68e22101411622ea5a57c2fc28aad32ac08bb90dd9cfa52a0d3736c13"
VAE_CHECKPOINT_SHA256: Final[str] = "3a9673d95a7744e51c18a71249192ce15e840fc57996f1c0c3d67fdb663f69be"
VAE_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/organism_raster_vae_v2/fit_v2"
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/organism_latent_flow/__init__.py",
    "forge/organism_latent_flow/__main__.py",
    "forge/organism_latent_flow/artifacts.py",
    "forge/organism_latent_flow/contract.py",
    "forge/organism_latent_flow/corpus.py",
    "forge/organism_latent_flow/model.py",
    "forge/organism_latent_flow/training.py",
)


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def source_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(source_manifest())).hexdigest()


@dataclass(frozen=True, slots=True)
class OrganismFlowConfig:
    coarse_channels: int = 32
    fine_channels: int = 16
    coarse_width: int = 256
    fine_width: int = 192
    condition_dim: int = 256
    time_dim: int = 128
    depth: int = 6
    condition_dropout: float = 0.12
    posterior_noise: float = 0.18
    ema_decay: float = 0.999

    def __post_init__(self) -> None:
        integers = (self.coarse_channels, self.fine_channels, self.coarse_width, self.fine_width, self.condition_dim, self.time_dim, self.depth)
        if any(type(value) is not int for value in integers):
            raise ValueError("Organism flow integer contract drifted.")
        if self.coarse_channels != 32 or self.fine_channels != 16:
            raise ValueError("Organism flow must match the frozen VAE latent pyramid.")
        if not 64 <= self.coarse_width <= 512 or not 48 <= self.fine_width <= 384 or not 64 <= self.condition_dim <= 512 or self.time_dim < 32 or not 1 <= self.depth <= 12:
            raise ValueError("Organism flow dimensions are outside bounds.")
        for value in (self.condition_dropout, self.posterior_noise, self.ema_decay):
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("Organism flow scalar contract drifted.")
        if not 0 <= self.condition_dropout < .5 or not 0 <= self.posterior_noise <= 1 or not .9 <= self.ema_decay < 1:
            raise ValueError("Organism flow scalar bounds drifted.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
