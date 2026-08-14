from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from ..config import PROJECT_ROOT
from ..organism_raster_vae.contract import FROZEN_UPSTREAM, organism_vae_source_sha256


FORMAT: Final[str] = "nullvector-hierarchical-organism-raster-vae-v2/1.0.0"
CHECKPOINT_FORMAT: Final[str] = "nullvector-hierarchical-organism-raster-vae-v2-checkpoint/1.0.0"
V1_DATA_CONTRACT_SHA256: Final[str] = organism_vae_source_sha256()
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/organism_raster_vae_v2/__init__.py", "forge/organism_raster_vae_v2/__main__.py",
    "forge/organism_raster_vae_v2/contract.py", "forge/organism_raster_vae_v2/dataset.py",
    "forge/organism_raster_vae_v2/model.py", "forge/organism_raster_vae_v2/smoke.py",
)


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    result = {}
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file(): raise FileNotFoundError(relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def source_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(source_manifest())).hexdigest()


def authority() -> dict[str, Any]:
    return {"v1_data_contract_sha256": V1_DATA_CONTRACT_SHA256, "frozen_upstream": FROZEN_UPSTREAM}


@dataclass(frozen=True, slots=True)
class OrganismVAEV2Config:
    width: int = 128
    coarse_width: int = 384
    coarse_latent_channels: int = 32
    fine_latent_channels: int = 16
    residual_depth: int = 3
    condition_dim: int = 192
    input_channels: int = 74
    style_dim: int = 8
    beta_coarse: float = 1.5e-3
    beta_fine: float = 8e-4
    free_bits: float = 0.02

    def __post_init__(self) -> None:
        ints = (self.width, self.coarse_width, self.coarse_latent_channels, self.fine_latent_channels, self.residual_depth, self.condition_dim, self.input_channels, self.style_dim)
        if any(type(value) is not int for value in ints) or self.input_channels != 74 or self.style_dim != 8: raise ValueError("Organism VAE v2 integer contract drifted.")
        if self.width < 32 or self.width % 16 or self.coarse_width < self.width * 2 or self.coarse_width % 16 or not 8 <= self.coarse_latent_channels <= 64 or not 4 <= self.fine_latent_channels <= 32 or not 1 <= self.residual_depth <= 6 or self.condition_dim < 64: raise ValueError("Organism VAE v2 dimensions are outside bounds.")
        for value in (self.beta_coarse, self.beta_fine, self.free_bits):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0: raise ValueError("Organism VAE v2 regularization drifted.")

    def to_dict(self) -> dict[str, Any]: return asdict(self)
