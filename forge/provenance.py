from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .config import PROJECT_ROOT


INFERENCE_SOURCE_FILES = (
    "forge/config.py",
    "forge/determinism.py",
    "forge/diffusion.py",
    "forge/grammar.py",
    "forge/provenance.py",
    "forge/rig.py",
    "forge/safety.py",
    "forge/sample.py",
)


def checkpoint_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_state_dict_hash(state: Mapping[str, Tensor]) -> str:
    """Hash tensor names, shapes, dtypes and bytes independent of .pt packaging."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(map(str, tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def inference_source_hash(root: Path = PROJECT_ROOT) -> str:
    digest = hashlib.sha256()
    root = Path(root)
    for relative in INFERENCE_SOURCE_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def architecture_from_state_dict(state: Mapping[str, Tensor]) -> dict[str, int | str]:
    """Recover architecture metadata from legacy checkpoints that predate it."""
    token_count = int(state["output.2.weight"].shape[0])
    return {
        "name": "categorical-absorbing-diffusion-unet",
        "token_count": token_count,
        "mask_token": token_count,
        "archetype_count": int(state["archetype_embedding.weight"].shape[0]),
        "gene_dim": int(state["gene_embedding.0.weight"].shape[1]),
        "steps": int(state["time_embedding.weight"].shape[0] - 1),
        "width": int(state["token_embedding.weight"].shape[1]),
        "image_size": 32,
    }


def model_from_architecture(architecture: Mapping[str, object]):
    # Local import avoids making checkpoint utilities depend on model import order.
    from .diffusion import CategoricalSpriteDiffusion

    return CategoricalSpriteDiffusion(
        token_count=int(architecture["token_count"]),
        archetype_count=int(architecture["archetype_count"]),
        gene_dim=int(architecture["gene_dim"]),
        steps=int(architecture["steps"]),
        width=int(architecture["width"]),
        image_size=int(architecture.get("image_size", 32)),
    )
