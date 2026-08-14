from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from ..config import PROJECT_ROOT


FORMAT: Final[str] = "nullvector-continuous-organism-raster-vae-smoke/1.0.0"
CHECKPOINT_FORMAT: Final[str] = "nullvector-continuous-organism-raster-vae-checkpoint/1.0.0"
ANATOMY_MANIFEST: Final[Path] = PROJECT_ROOT / "outputs/cellular_breeding_symmetry_v1/cellular_symmetry_manifest.json"
PHYSIOLOGY_MANIFEST: Final[Path] = PROJECT_ROOT / "outputs/cellular_physiology_v4/cellular_physiology_manifest.json"
TRAUMA_MANIFEST: Final[Path] = PROJECT_ROOT / "outputs/cellular_trauma_v4/cellular_trauma_manifest.json"
FROZEN_UPSTREAM: Final[dict[str, str]] = {
    "anatomy_manifest_sha256": "7be6610ce2de3ba507608ed5ad96cf20aed7faec13885b1fbed45a9887f19452",
    "physiology_manifest_sha256": "c532542bbcb42ab5257a1fd3db6ce085b3b4c9915400960db2889c1dd173687c",
    "trauma_manifest_sha256": "daf6aa1ec42f12da0f9ac8bbcae0786c918de98701715d533d14dbeb7458b476",
}
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/organism_raster_vae/__init__.py",
    "forge/organism_raster_vae/__main__.py",
    "forge/organism_raster_vae/contract.py",
    "forge/organism_raster_vae/dataset.py",
    "forge/organism_raster_vae/model.py",
    "forge/organism_raster_vae/smoke.py",
)


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def organism_vae_source_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(source_manifest())).hexdigest()


@dataclass(frozen=True, slots=True)
class OrganismVAEConfig:
    image_size: int = 48
    input_channels: int = 74
    width: int = 48
    latent_channels: int = 12
    residual_depth: int = 2
    condition_dim: int = 64
    gene_dim: int = 16
    beta: float = 2.0e-3
    free_bits: float = 0.02
    occupancy_weight: float = 1.0
    rgba_weight: float = 2.0
    categorical_weight: float = 0.7
    physiology_weight: float = 0.8
    cell_state_weight: float = 0.5
    symmetry_weight: float = 0.08
    alpha_consistency_weight: float = 0.2

    def __post_init__(self) -> None:
        if self.image_size != 48 or self.input_channels != 74:
            raise ValueError("Organism VAE v1 requires the native 48px/74-channel living field.")
        if self.width < 16 or self.width % 8 or not 4 <= self.latent_channels <= 64 or not 1 <= self.residual_depth <= 6:
            raise ValueError("Organism VAE dimensions are outside the bounded contract.")
        if self.condition_dim < 32 or self.gene_dim != 16:
            raise ValueError("Organism VAE conditioning dimensions drifted.")
        for value in (self.beta, self.free_bits, self.occupancy_weight, self.rgba_weight, self.categorical_weight, self.physiology_weight, self.cell_state_weight, self.symmetry_weight, self.alpha_consistency_weight):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("Organism VAE loss weights must be finite and non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
