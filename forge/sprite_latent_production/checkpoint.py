from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from math import isfinite

import torch

from ..sprite_latent.corpus import (
    FROZEN_PRODUCTION_CORPUS_SHA256,
    FROZEN_PRODUCTION_LEGAL_TUPLE_FINGERPRINT,
    FROZEN_PRODUCTION_SPLIT_FINGERPRINT,
)
from ..sprite_latent.training import canonical_state_hash
from .contract import CHECKPOINT_FORMAT, ProductionConfig, production_source_hash, sha256_file


def _state_hash(state: dict[str, torch.Tensor], config: ProductionConfig) -> str:
    from ..sprite_latent.codec import SemanticSpriteFSQ

    model = SemanticSpriteFSQ(config.codec_config())
    model.load_state_dict(state, strict=True)
    return canonical_state_hash(model)


def _assert_finite_tensors(value: Any, *, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"production checkpoint {label} contains non-finite tensors")
        return
    if isinstance(value, dict):
        for nested in value.values():
            _assert_finite_tensors(nested, label=label)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _assert_finite_tensors(nested, label=label)
        return
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"production checkpoint {label} contains non-finite values")


def validate_checkpoint(payload: dict[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    required = {
        "format", "source_sha256", "corpus_sha256", "split_fingerprint",
        "legal_tuple_fingerprint", "config", "epoch", "global_step",
        "model_state", "ema_state", "optimizer_state", "model_state_sha256",
        "ema_state_sha256", "history", "partial_epoch", "previous_checkpoint_sha256", "rng",
    }
    if set(payload) != required:
        raise ValueError(f"production checkpoint key mismatch missing={sorted(required-set(payload))} extra={sorted(set(payload)-required)}")
    if payload["format"] != CHECKPOINT_FORMAT or payload["source_sha256"] != production_source_hash():
        raise ValueError("production checkpoint format/source mismatch")
    config = ProductionConfig.from_metadata(dict(payload["config"]))
    epoch = int(payload["epoch"]); global_step = int(payload["global_step"])
    if epoch < 0 or epoch > config.epochs or global_step < 0 or len(payload["history"]) != epoch:
        raise ValueError("production checkpoint progress/history mismatch")
    if _state_hash(payload["model_state"], config) != payload["model_state_sha256"]:
        raise ValueError("production checkpoint model state hash mismatch")
    if _state_hash(payload["ema_state"], config) != payload["ema_state_sha256"]:
        raise ValueError("production checkpoint EMA state hash mismatch")
    _assert_finite_tensors(payload["model_state"], label="model state")
    _assert_finite_tensors(payload["ema_state"], label="EMA state")
    frozen = {
        "corpus_sha256": FROZEN_PRODUCTION_CORPUS_SHA256,
        "split_fingerprint": FROZEN_PRODUCTION_SPLIT_FINGERPRINT,
        "legal_tuple_fingerprint": FROZEN_PRODUCTION_LEGAL_TUPLE_FINGERPRINT,
    }
    for key, expected in frozen.items():
        value = str(payload[key])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"production checkpoint {key} is malformed")
        if value != expected:
            raise ValueError(f"production checkpoint {key} is not the frozen production authority")
    if not isinstance(payload["history"], list):
        raise TypeError("production checkpoint history must be a list")
    total_steps = 0
    for expected_epoch, record in enumerate(payload["history"], start=1):
        if not isinstance(record, dict) or int(record.get("epoch", -1)) != expected_epoch or record.get("complete") is not True:
            raise ValueError("production checkpoint history is not contiguous and complete")
        steps = int(record.get("steps", -1))
        if steps <= 0:
            raise ValueError("production checkpoint history contains an invalid step count")
        total_steps += steps
    partial = payload["partial_epoch"]
    if partial is not None:
        if (
            not isinstance(partial, dict)
            or int(partial.get("epoch", -1)) != epoch + 1
            or partial.get("complete") is not False
            or int(partial.get("steps", -1)) <= 0
        ):
            raise ValueError("production checkpoint partial epoch is malformed")
        total_steps += int(partial["steps"])
    if total_steps != global_step:
        raise ValueError("production checkpoint global step does not match history")
    previous = payload["previous_checkpoint_sha256"]
    if previous is not None and (
        not isinstance(previous, str)
        or len(previous) != 64
        or any(character not in "0123456789abcdef" for character in previous)
    ):
        raise ValueError("production checkpoint predecessor hash is malformed")
    if not isinstance(payload["optimizer_state"], dict):
        raise TypeError("production checkpoint optimizer state must be a mapping")
    _assert_finite_tensors(payload["optimizer_state"], label="optimizer state")
    _assert_finite_tensors(payload["history"], label="history")
    _assert_finite_tensors(payload["partial_epoch"], label="partial epoch")
    rng = payload["rng"]
    if not isinstance(rng, dict) or set(rng) != {"torch", "cuda"}:
        raise ValueError("production checkpoint RNG contract is malformed")
    if not isinstance(rng["torch"], torch.Tensor) or rng["torch"].dtype != torch.uint8 or rng["torch"].ndim != 1:
        raise ValueError("production checkpoint CPU RNG state is malformed")
    if not isinstance(rng["cuda"], list) or len(rng["cuda"]) > 16 or any(
        not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1
        for value in rng["cuda"]
    ):
        raise ValueError("production checkpoint CUDA RNG state is malformed")
    if path is not None and not Path(path).is_file():
        raise ValueError("production checkpoint path does not exist")
    return payload


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size > 512 * 1024**2:
        raise ValueError("production checkpoint must be a bounded regular file")
    payload = torch.load(resolved, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("production checkpoint must contain a mapping")
    return validate_checkpoint(payload, path=resolved)


def save_checkpoint_new(path: Path, payload: dict[str, Any]) -> str:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    load_checkpoint(target)
    return sha256_file(target)
