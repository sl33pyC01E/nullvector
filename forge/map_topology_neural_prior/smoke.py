from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Final

import torch

from ..map_topology_neural.corpus import FROZEN_CORPUS_MANIFEST_FILE_SHA256, FROZEN_CORPUS_SHA256
from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..safety import require_disk_floor
from .checkpoint import load_checkpoint, save_checkpoint
from .contract import (
    FROZEN_CODEC_CHECKPOINT_SHA256,
    FROZEN_CODEC_EMA_SHA256,
    FROZEN_CODEC_SOURCE_SHA256,
    PRIOR_FORMAT,
    RAW_BANK_FORMAT,
    MaskedPriorConfig,
    canonical_json_bytes,
    prior_source_sha256,
    sha256_file,
    source_manifest,
)
from .dataset import FrozenLatentDataset
from .masking import mask_tokens, sample_parallel, tensor_sha256
from .model import build_prior, masked_token_loss


MANIFEST_NAME: Final[str] = "smoke_manifest.json"
RAW_BANK_NAME: Final[str] = "raw_latent_bank.json"
CHECKPOINT_NAME: Final[str] = "checkpoint.pt"
MAX_JSON_BYTES: Final[int] = 16 * 1024 * 1024
GATE_KEYS: Final[set[str]] = {
    "cpu_only_cuda_uninitialized", "codec_frozen", "six_theme_conditions_exact",
    "finite_training_history", "model_updated", "all_raw_tokens_revealed",
    "raw_proposals_not_compiled", "deterministic_compiler_not_invoked",
    "runtime_integration_disabled",
}


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_canonical(path: Path, *, self_key: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_JSON_BYTES:
        raise ValueError(f"Masked-prior {path.name} is missing or oversized.")
    encoded = path.read_bytes()
    payload = json.loads(encoded)
    if not isinstance(payload, dict) or encoded != canonical_json_bytes(payload):
        raise ValueError(f"Masked-prior {path.name} is not canonical JSON.")
    stored = payload.pop(self_key, None)
    if stored != _hash_json(payload):
        raise ValueError(f"Masked-prior {path.name} self-hash failed.")
    payload[self_key] = stored
    return payload


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    return value


def _ema_update(ema: dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                ema[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema[name].copy_(value.detach())


def _fixed_mask_accuracy(model: torch.nn.Module, latent: dict[str, torch.Tensor], config: MaskedPriorConfig) -> float:
    generator = torch.Generator(device="cpu").manual_seed(config.seed ^ 0x4556414C)
    masked = mask_tokens(latent["targets"], latent["valid_mask"], generator=generator, config=config, step=0)
    with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
        logits = model({
            **{name: latent[name] for name in ("valid_mask", "point_conditions", "global_conditions", "theme_index")},
            "tokens": masked["tokens"],
            "mask_fraction": masked["mask_fraction"],
        })
    prediction = logits.argmax(dim=1)
    return float((prediction[masked["mask"]] == latent["targets"][masked["mask"]]).float().mean())


def _train(latent: dict[str, torch.Tensor], config: MaskedPriorConfig) -> dict[str, Any]:
    model = build_prior(config)
    initial_sha256 = tensor_state_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(config.seed ^ 0x545241494E)
    ema = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, Any]] = []
    model.train()
    with torch.backends.mkldnn.flags(enabled=False):
        for step in range(config.steps):
            masked = mask_tokens(latent["targets"], latent["valid_mask"], generator=generator, config=config, step=step)
            optimizer.zero_grad(set_to_none=True)
            logits = model({
                **{name: latent[name] for name in ("valid_mask", "point_conditions", "global_conditions", "theme_index")},
                "tokens": masked["tokens"],
                "mask_fraction": masked["mask_fraction"],
            })
            loss = masked_token_loss(logits, latent["targets"], masked["mask"])
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            _ema_update(ema, model, config.model_ema_decay)
            history.append({
                "step": step + 1,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                "masked_cells": int(masked["mask"].sum()),
                "mask_fraction_mean": float(masked["mask_fraction"].mean()),
                "modes": list(masked["modes"]),
                "masked_tokens_sha256": tensor_sha256(masked["tokens"]),
                "mask_sha256": tensor_sha256(masked["mask"]),
            })
    raw_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    ema_model = build_prior(config)
    ema_model.load_state_dict(ema, strict=True)
    ema_model.eval()
    model.eval()
    return {
        "model": model,
        "ema_model": ema_model,
        "raw_state": raw_state,
        "ema_state": ema,
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "generator_state": generator.get_state().cpu(),
        "history": history,
        "initial_sha256": initial_sha256,
        "model_sha256": tensor_state_sha256(raw_state),
        "ema_sha256": tensor_state_sha256(ema),
        "raw_accuracy": _fixed_mask_accuracy(model, latent, config),
        "ema_accuracy": _fixed_mask_accuracy(ema_model, latent, config),
    }


def _raw_bank(
    latent: dict[str, torch.Tensor],
    refs: tuple[Any, ...],
    identity: Any,
    ema_model: torch.nn.Module,
    config: MaskedPriorConfig,
) -> dict[str, Any]:
    conditions = {name: latent[name] for name in ("valid_mask", "point_conditions", "global_conditions", "theme_index")}
    sampled = sample_parallel(ema_model, conditions, sampling_steps=config.sampling_steps)
    samples: list[dict[str, Any]] = []
    for index, ref in enumerate(refs):
        height = int(latent["valid_mask"][index, 0].any(dim=1).sum())
        width = int(latent["valid_mask"][index, 0].any(dim=0).sum())
        tokens = sampled["tokens"][index, :height, :width].contiguous()
        uncertainty = sampled["uncertainty"][index, :height, :width].contiguous()
        samples.append({
            "theme": ref.theme,
            "source_full_map_identity_sha256": ref.full_map_identity_sha256,
            "source_latent_target_sha256": tensor_sha256(latent["targets"][index, :height, :width]),
            "shape": [height, width],
            "tokens": tokens.tolist(),
            "tokens_sha256": tensor_sha256(tokens),
            "uncertainty_q16": torch.round(uncertainty.clamp(0, 1) * 65535).to(torch.int32).tolist(),
            "trace_sha256": list(sampled["trace_sha256"]),
        })
    bank: dict[str, Any] = {
        "format": RAW_BANK_FORMAT,
        "proposal_status": "raw_neural_latents_uncompiled_unvalidated",
        "source_sha256": prior_source_sha256(),
        "codec_checkpoint_sha256": FROZEN_CODEC_CHECKPOINT_SHA256,
        "codec_ema_sha256": FROZEN_CODEC_EMA_SHA256,
        "source_batch": {
            "sample_ids": list(identity.sample_ids),
            "target_sha256": identity.target_sha256,
            "valid_sha256": identity.valid_sha256,
            "point_sha256": identity.point_sha256,
        },
        "sampling_steps": config.sampling_steps,
        "samples": samples,
    }
    bank["bank_sha256"] = _hash_json(bank)
    return bank


def _execute(corpus_root: Path, config: MaskedPriorConfig) -> dict[str, Any]:
    if torch.cuda.is_initialized():
        raise RuntimeError("Masked-prior foundation must start before CUDA initialization.")
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        dataset = FrozenLatentDataset(corpus_root)
        refs = dataset.smoke_refs()
        latent, identity = dataset.encode(refs)
        trained = _train(latent, config)
        bank = _raw_bank(latent, refs, identity, trained["ema_model"], config)
        return {"dataset": dataset, "refs": refs, "latent": latent, "identity": identity, "trained": trained, "bank": bank}
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        torch.set_num_threads(previous_threads)


def build_smoke(output: Path, *, corpus_root: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Masked-prior smoke publication is immutable.")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=256 * 1024 * 1024)
    config = MaskedPriorConfig()
    executed = _execute(Path(corpus_root), config)
    trained = executed["trained"]
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        checkpoint_payload = {
            "format": "nullvector-neural-map-topology-masked-prior-checkpoint/1.0.0",
            "source_sha256": prior_source_sha256(),
            "source_manifest": source_manifest(),
            "codec_checkpoint_sha256": FROZEN_CODEC_CHECKPOINT_SHA256,
            "codec_source_sha256": FROZEN_CODEC_SOURCE_SHA256,
            "codec_ema_sha256": FROZEN_CODEC_EMA_SHA256,
            "corpus_sha256": FROZEN_CORPUS_SHA256,
            "corpus_manifest_file_sha256": FROZEN_CORPUS_MANIFEST_FILE_SHA256,
            "config": config.to_dict(),
            "step": config.steps,
            "model_state": trained["raw_state"],
            "ema_state": trained["ema_state"],
            "optimizer_state": trained["optimizer_state"],
            "generator_state": trained["generator_state"],
            "torch_rng_state": torch.Generator(device="cpu").manual_seed(config.seed ^ 0x524E47).get_state(),
            "history": trained["history"],
            "model_state_sha256": trained["model_sha256"],
            "ema_state_sha256": trained["ema_sha256"],
        }
        sidecar = save_checkpoint(staging / CHECKPOINT_NAME, checkpoint_payload)
        _atomic_json(staging / RAW_BANK_NAME, executed["bank"])
        gates = {
            "cpu_only_cuda_uninitialized": not torch.cuda.is_initialized(),
            "codec_frozen": all(not parameter.requires_grad for parameter in executed["dataset"].codec.parameters()),
            "six_theme_conditions_exact": {ref.theme for ref in executed["refs"]} == {"arena", "rooms", "caves", "archipelago", "garden", "anomaly"},
            "finite_training_history": all(math.isfinite(item["loss"]) for item in trained["history"]),
            "model_updated": trained["model_sha256"] != trained["initial_sha256"],
            "all_raw_tokens_revealed": all(
                all(0 <= token < 512 for row in sample["tokens"] for token in row)
                for sample in executed["bank"]["samples"]
            ),
            "raw_proposals_not_compiled": True,
            "deterministic_compiler_not_invoked": True,
            "runtime_integration_disabled": True,
        }
        require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=0)
        manifest: dict[str, Any] = {
            "format": PRIOR_FORMAT,
            "status": "passed" if all(gates.values()) else "failed",
            "source_sha256": prior_source_sha256(),
            "source_manifest": source_manifest(),
            "corpus_sha256": FROZEN_CORPUS_SHA256,
            "corpus_manifest_file_sha256": FROZEN_CORPUS_MANIFEST_FILE_SHA256,
            "codec": {
                "checkpoint_sha256": FROZEN_CODEC_CHECKPOINT_SHA256,
                "source_sha256": FROZEN_CODEC_SOURCE_SHA256,
                "ema_sha256": FROZEN_CODEC_EMA_SHA256,
                "frozen": True,
            },
            "config": config.to_dict(),
            "source_batch": {
                "sample_ids": list(executed["identity"].sample_ids),
                "target_sha256": executed["identity"].target_sha256,
                "valid_sha256": executed["identity"].valid_sha256,
                "point_sha256": executed["identity"].point_sha256,
            },
            "history": trained["history"],
            "metrics": {
                "fixed_mask_accuracy_raw": trained["raw_accuracy"],
                "fixed_mask_accuracy_ema": trained["ema_accuracy"],
                "unique_raw_samples": len({sample["tokens_sha256"] for sample in executed["bank"]["samples"]}),
                "raw_sample_count": len(executed["bank"]["samples"]),
            },
            "model": {
                "initial_sha256": trained["initial_sha256"],
                "raw_sha256": trained["model_sha256"],
                "ema_sha256": trained["ema_sha256"],
            },
            "checkpoint": {**sidecar, "path": CHECKPOINT_NAME},
            "raw_bank": {
                "path": RAW_BANK_NAME,
                "file_sha256": sha256_file(staging / RAW_BANK_NAME),
                "bank_sha256": executed["bank"]["bank_sha256"],
            },
            "gates": gates,
            "claim_boundary": {
                "cpu_foundation_only": True,
                "generative_quality_claim": False,
                "raw_latents_are_runtime_maps": False,
                "compiler_invoked": False,
                "cuda_training_started": False,
                "godot_integration": False,
            },
        }
        manifest["manifest_sha256"] = _hash_json(manifest)
        _atomic_json(staging / MANIFEST_NAME, manifest)
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            diagnostic = output.parent / f"{staging.name}.failed"
            if diagnostic.exists():
                diagnostic = output.parent / f"{staging.name}.failed-{time.time_ns()}"
            os.replace(staging, diagnostic)
        raise
    return validate_smoke(output, corpus_root=corpus_root)


def validate_smoke(output: Path, *, corpus_root: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest = _read_canonical(output / MANIFEST_NAME, self_key="manifest_sha256")
    expected_keys = {
        "format", "status", "source_sha256", "source_manifest", "corpus_sha256",
        "corpus_manifest_file_sha256", "codec", "config", "source_batch", "history",
        "metrics", "model", "checkpoint", "raw_bank", "gates", "claim_boundary",
        "manifest_sha256",
    }
    if set(manifest) != expected_keys or manifest["format"] != PRIOR_FORMAT or manifest["status"] != "passed":
        raise ValueError("Masked-prior smoke manifest format/status/census failed.")
    if manifest["source_sha256"] != prior_source_sha256() or manifest["source_manifest"] != source_manifest():
        raise ValueError("Masked-prior smoke source drifted.")
    if manifest["corpus_sha256"] != FROZEN_CORPUS_SHA256 or manifest["corpus_manifest_file_sha256"] != FROZEN_CORPUS_MANIFEST_FILE_SHA256:
        raise ValueError("Masked-prior smoke corpus authority drifted.")
    if manifest["codec"] != {
        "checkpoint_sha256": FROZEN_CODEC_CHECKPOINT_SHA256,
        "source_sha256": FROZEN_CODEC_SOURCE_SHA256,
        "ema_sha256": FROZEN_CODEC_EMA_SHA256,
        "frozen": True,
    }:
        raise ValueError("Masked-prior smoke codec authority drifted.")
    if (
        not isinstance(manifest["gates"], dict)
        or set(manifest["gates"]) != GATE_KEYS
        or any(type(value) is not bool or not value for value in manifest["gates"].values())
    ):
        raise ValueError("Masked-prior smoke safety gates failed.")
    expected_claim = {
        "cpu_foundation_only": True,
        "generative_quality_claim": False,
        "raw_latents_are_runtime_maps": False,
        "compiler_invoked": False,
        "cuda_training_started": False,
        "godot_integration": False,
    }
    if manifest["claim_boundary"] != expected_claim:
        raise ValueError("Masked-prior smoke claim boundary drifted.")
    config = MaskedPriorConfig.from_dict(manifest["config"])
    checkpoint_path = output / manifest["checkpoint"]["path"]
    payload = load_checkpoint(checkpoint_path)
    sidecar = json.loads(checkpoint_path.with_suffix(checkpoint_path.suffix + ".json").read_text(encoding="utf-8"))
    if manifest["checkpoint"] != {**sidecar, "path": CHECKPOINT_NAME}:
        raise ValueError("Masked-prior smoke checkpoint descriptor drifted.")
    if payload["history"] != manifest["history"] or payload["model_state_sha256"] != manifest["model"]["raw_sha256"] or payload["ema_state_sha256"] != manifest["model"]["ema_sha256"]:
        raise ValueError("Masked-prior smoke checkpoint semantics drifted.")
    raw_path = output / manifest["raw_bank"]["path"]
    if sha256_file(raw_path) != manifest["raw_bank"]["file_sha256"]:
        raise ValueError("Masked-prior raw bank file identity failed.")
    raw_bank = _read_canonical(raw_path, self_key="bank_sha256")
    if raw_bank["bank_sha256"] != manifest["raw_bank"]["bank_sha256"]:
        raise ValueError("Masked-prior raw bank semantic identity failed.")
    replay = _execute(Path(corpus_root), config)
    trained = replay["trained"]
    if (
        trained["history"] != manifest["history"]
        or trained["initial_sha256"] != manifest["model"]["initial_sha256"]
        or trained["model_sha256"] != manifest["model"]["raw_sha256"]
        or trained["ema_sha256"] != manifest["model"]["ema_sha256"]
        or replay["bank"] != raw_bank
    ):
        raise ValueError("Masked-prior smoke exact replay failed.")
    if manifest["source_batch"] != {
        "sample_ids": list(replay["identity"].sample_ids),
        "target_sha256": replay["identity"].target_sha256,
        "valid_sha256": replay["identity"].valid_sha256,
        "point_sha256": replay["identity"].point_sha256,
    }:
        raise ValueError("Masked-prior smoke source latent batch drifted.")
    expected_metrics = {
        "fixed_mask_accuracy_raw": trained["raw_accuracy"],
        "fixed_mask_accuracy_ema": trained["ema_accuracy"],
        "unique_raw_samples": len({sample["tokens_sha256"] for sample in raw_bank["samples"]}),
        "raw_sample_count": len(raw_bank["samples"]),
    }
    if manifest["metrics"] != expected_metrics:
        raise ValueError("Masked-prior smoke metrics drifted.")
    return manifest
