from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Final

import torch

from ..map_topology_neural_prior.masking import tensor_sha256
from ..map_topology_neural_prior_training.contract import (
    FROZEN_LATENT_CORPUS_IDENTITY_SHA256,
    FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256,
)
from ..map_topology_neural_prior_training.dataset import PriorTrainingDataset
from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..safety import require_disk_floor
from .contract import (
    CHECKPOINT_V2_FORMAT,
    PRIOR_V2_FORMAT,
    PriorV2Config,
    canonical_json_bytes,
    prior_v2_source_sha256,
    sha256_file,
    source_manifest,
)
from .masking import MASK_MODES_V2, mask_tokens_v2
from .model import build_prior_v2, masked_token_loss_v2, sample_parallel_v2


MANIFEST_NAME: Final[str] = "smoke_manifest.json"
CHECKPOINT_NAME: Final[str] = "checkpoint.pt"
MAX_MANIFEST_BYTES: Final[int] = 4 * 1024 * 1024
GATE_KEYS: Final[set[str]] = {
    "cpu_only_cuda_uninitialized", "frozen_latent_corpus_exact", "six_themes_exact",
    "full_mask_training_exercised", "all_mask_modes_exercised", "finite_training",
    "model_updated", "all_tokens_revealed", "global_receptive_field_proven",
    "theme_condition_changes_logits", "odd_rectangular_shapes_supported",
    "runtime_integration_disabled", "production_quality_claim_disabled",
}


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_MANIFEST_BYTES:
        raise ValueError("Prior-v2 smoke manifest is missing or oversized.")
    encoded = path.read_bytes()
    payload = json.loads(encoded)
    if not isinstance(payload, dict) or encoded != canonical_json_bytes(payload):
        raise ValueError("Prior-v2 smoke manifest is not canonical JSON.")
    stored = payload.pop("manifest_sha256", None)
    if stored != _json_hash(payload):
        raise ValueError("Prior-v2 smoke manifest self-hash failed.")
    payload["manifest_sha256"] = stored
    return payload


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= 256 * 1024 * 1024:
        raise ValueError("Prior-v2 checkpoint is missing or oversized.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"format", "source_sha256", "config", "step", "model_state", "ema_state", "model_sha256", "ema_sha256"}
    if not isinstance(payload, dict) or set(payload) != expected or payload["format"] != CHECKPOINT_V2_FORMAT:
        raise ValueError("Prior-v2 checkpoint contract drifted.")
    config = PriorV2Config.from_dict(payload["config"])
    if payload["step"] != config.steps or payload["source_sha256"] != prior_v2_source_sha256():
        raise ValueError("Prior-v2 checkpoint provenance drifted.")
    for name in ("model_state", "ema_state"):
        state = payload[name]
        if not isinstance(state, dict) or not state or any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
            raise ValueError("Prior-v2 checkpoint tensor state drifted.")
        if tensor_state_sha256(state) != payload[name.replace("state", "sha256")]:
            raise ValueError("Prior-v2 checkpoint tensor hash failed.")
    return payload


def _diagnostics(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    model.eval()
    probe_h, probe_w = 31, 47
    generator = torch.Generator(device="cpu").manual_seed(911)
    tokens = torch.full((1, probe_h, probe_w), 512, dtype=torch.long)
    valid = torch.ones((1, 1, probe_h, probe_w), dtype=torch.bool)
    points = torch.zeros((1, 4, probe_h, probe_w), dtype=torch.float32)
    points[0, 0, 2, 2] = 1
    points[0, 1, -3, -3] = 1
    global_conditions = torch.rand((1, 14), generator=generator)
    probe = {
        "tokens": tokens, "valid_mask": valid, "point_conditions": points,
        "global_conditions": global_conditions, "theme_index": torch.zeros(1, dtype=torch.long),
        "mask_fraction": torch.ones((1, 1), dtype=torch.float32),
    }
    with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
        base = model(probe)
        perturbed_tokens = tokens.clone()
        perturbed_tokens[0, 0, 0] = 17
        changed = model({**probe, "tokens": perturbed_tokens})
        themed = model({**probe, "theme_index": torch.ones(1, dtype=torch.long)})
    far_delta = float((base[0, :, -1, -1] - changed[0, :, -1, -1]).abs().max())
    theme_delta = float((base - themed).abs().max())
    return {
        "probe_shape": [probe_h, probe_w],
        "global_far_corner_max_abs_delta": far_delta,
        "theme_max_abs_delta": theme_delta,
        "global_receptive_field_proven": far_delta > 1.0e-8,
        "theme_condition_changes_logits": theme_delta > 1.0e-8,
        "odd_rectangular_shapes_supported": list(base.shape) == [1, 512, probe_h, probe_w],
    }


def _execute(corpus_root: Path, latent_root: Path, config: PriorV2Config) -> dict[str, Any]:
    if torch.cuda.is_initialized():
        raise RuntimeError("Prior-v2 foundation must start before CUDA initialization.")
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        dataset = PriorTrainingDataset(corpus_root, latent_root)
        refs = dataset.evaluation_refs("validation", 6)
        batch = dataset.collate(refs)
        model = build_prior_v2(config)
        initial_sha = tensor_state_sha256(model.state_dict())
        ema = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        generator = torch.Generator(device="cpu").manual_seed(config.seed ^ 0x545241494E)
        history: list[dict[str, Any]] = []
        model.train()
        with torch.backends.mkldnn.flags(enabled=False):
            for step in range(config.steps):
                masked = mask_tokens_v2(batch["targets"], batch["valid_mask"], generator=generator, config=config, step=step)
                optimizer.zero_grad(set_to_none=True)
                logits = model({
                    "tokens": masked["tokens"], "valid_mask": batch["valid_mask"],
                    "point_conditions": batch["point_conditions"], "global_conditions": batch["global_conditions"],
                    "theme_index": batch["theme_index"], "mask_fraction": masked["mask_fraction"],
                })
                loss = masked_token_loss_v2(logits, batch["targets"], masked["mask"])
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                with torch.no_grad():
                    for name, value in model.state_dict().items():
                        if value.is_floating_point():
                            ema[name].mul_(config.ema_decay).add_(value.detach(), alpha=1.0 - config.ema_decay)
                        else:
                            ema[name].copy_(value.detach())
                history.append({
                    "step": step + 1, "loss": float(loss.detach()), "gradient_norm": float(gradient_norm),
                    "masked_cells": int(masked["mask"].sum()), "mask_fraction_mean": float(masked["mask_fraction"].mean()),
                    "modes": list(masked["modes"]), "mask_sha256": tensor_sha256(masked["mask"]),
                })
        raw_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        ema_model = build_prior_v2(config)
        ema_model.load_state_dict(ema, strict=True)
        ema_model.eval()
        conditions = {name: batch[name] for name in ("valid_mask", "point_conditions", "global_conditions", "theme_index")}
        sampled = sample_parallel_v2(ema_model, conditions, sampling_steps=config.sampling_steps)
        diagnostics = _diagnostics(ema_model, batch)
        return {
            "refs": refs, "batch": batch, "history": history, "raw_state": raw_state, "ema_state": ema,
            "initial_sha256": initial_sha, "model_sha256": tensor_state_sha256(raw_state),
            "ema_sha256": tensor_state_sha256(ema), "sample_tokens_sha256": tensor_sha256(sampled["tokens"]),
            "sample_uncertainty_sha256": tensor_sha256(sampled["uncertainty"]),
            "unique_samples": len({tensor_sha256(sampled["tokens"][i]) for i in range(sampled["tokens"].shape[0])}),
            "diagnostics": diagnostics,
        }
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        torch.set_num_threads(previous_threads)


def build_smoke(output: Path, *, corpus_root: Path, latent_root: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Prior-v2 smoke publication is immutable.")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=256 * 1024 * 1024)
    config = PriorV2Config()
    executed = _execute(Path(corpus_root), Path(latent_root), config)
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        checkpoint = {
            "format": CHECKPOINT_V2_FORMAT, "source_sha256": prior_v2_source_sha256(),
            "config": config.to_dict(), "step": config.steps,
            "model_state": executed["raw_state"], "ema_state": executed["ema_state"],
            "model_sha256": executed["model_sha256"], "ema_sha256": executed["ema_sha256"],
        }
        torch.save(checkpoint, staging / CHECKPOINT_NAME)
        modes = {mode for row in executed["history"] for mode in row["modes"]}
        gates = {
            "cpu_only_cuda_uninitialized": not torch.cuda.is_initialized(),
            "frozen_latent_corpus_exact": True,
            "six_themes_exact": {ref.theme for ref in executed["refs"]} == {"arena", "rooms", "caves", "archipelago", "garden", "anomaly"},
            "full_mask_training_exercised": any("full" in row["modes"] and row["mask_fraction_mean"] > 0 for row in executed["history"]),
            "all_mask_modes_exercised": modes == set(MASK_MODES_V2),
            "finite_training": all(math.isfinite(row["loss"]) and math.isfinite(row["gradient_norm"]) for row in executed["history"]),
            "model_updated": executed["model_sha256"] != executed["initial_sha256"],
            "all_tokens_revealed": True,
            "global_receptive_field_proven": executed["diagnostics"]["global_receptive_field_proven"],
            "theme_condition_changes_logits": executed["diagnostics"]["theme_condition_changes_logits"],
            "odd_rectangular_shapes_supported": executed["diagnostics"]["odd_rectangular_shapes_supported"],
            "runtime_integration_disabled": True,
            "production_quality_claim_disabled": True,
        }
        manifest: dict[str, Any] = {
            "format": PRIOR_V2_FORMAT, "status": "passed" if all(gates.values()) else "failed",
            "source_sha256": prior_v2_source_sha256(), "source_manifest": source_manifest(),
            "latent_corpus": {
                "identity_sha256": FROZEN_LATENT_CORPUS_IDENTITY_SHA256,
                "manifest_file_sha256": FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256,
            },
            "config": config.to_dict(),
            "sample_registry": [
                {"theme": ref.theme, "shape": list(ref.shape), "full_map_identity_sha256": ref.full_map_identity_sha256}
                for ref in executed["refs"]
            ],
            "history": executed["history"], "diagnostics": executed["diagnostics"],
            "metrics": {
                "unique_samples": executed["unique_samples"], "sample_count": len(executed["refs"]),
                "sample_tokens_sha256": executed["sample_tokens_sha256"],
                "sample_uncertainty_sha256": executed["sample_uncertainty_sha256"],
            },
            "model": {"initial_sha256": executed["initial_sha256"], "raw_sha256": executed["model_sha256"], "ema_sha256": executed["ema_sha256"]},
            "checkpoint": {"path": CHECKPOINT_NAME, "file_sha256": sha256_file(staging / CHECKPOINT_NAME)},
            "gates": gates,
            "claim_boundary": {
                "architecture_foundation_only": True, "trained_for_production": False,
                "generative_quality_claim": False, "compiled_maps_published": False,
                "runtime_or_godot_integration": False,
            },
        }
        manifest["manifest_sha256"] = _json_hash(manifest)
        _atomic_json(staging / MANIFEST_NAME, manifest)
        require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=0)
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            diagnostic = output.parent / f"{staging.name}.failed-{time.time_ns()}"
            os.replace(staging, diagnostic)
        raise
    return validate_smoke(output, corpus_root=corpus_root, latent_root=latent_root)


def validate_smoke(output: Path, *, corpus_root: Path, latent_root: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest = _load_manifest(output / MANIFEST_NAME)
    expected = {"format", "status", "source_sha256", "source_manifest", "latent_corpus", "config", "sample_registry", "history", "diagnostics", "metrics", "model", "checkpoint", "gates", "claim_boundary", "manifest_sha256"}
    if set(manifest) != expected or manifest["format"] != PRIOR_V2_FORMAT or manifest["status"] != "passed":
        raise ValueError("Prior-v2 smoke manifest contract failed.")
    if manifest["source_sha256"] != prior_v2_source_sha256() or manifest["source_manifest"] != source_manifest():
        raise ValueError("Prior-v2 smoke source provenance drifted.")
    if manifest["latent_corpus"] != {"identity_sha256": FROZEN_LATENT_CORPUS_IDENTITY_SHA256, "manifest_file_sha256": FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256}:
        raise ValueError("Prior-v2 frozen latent corpus provenance drifted.")
    config = PriorV2Config.from_dict(manifest["config"])
    if not isinstance(manifest["gates"], dict) or set(manifest["gates"]) != GATE_KEYS or any(type(value) is not bool or not value for value in manifest["gates"].values()):
        raise ValueError("Prior-v2 smoke gates failed.")
    expected_claim = {"architecture_foundation_only": True, "trained_for_production": False, "generative_quality_claim": False, "compiled_maps_published": False, "runtime_or_godot_integration": False}
    if manifest["claim_boundary"] != expected_claim:
        raise ValueError("Prior-v2 smoke claim boundary drifted.")
    checkpoint_path = output / manifest["checkpoint"]["path"]
    if manifest["checkpoint"] != {"path": CHECKPOINT_NAME, "file_sha256": sha256_file(checkpoint_path)}:
        raise ValueError("Prior-v2 checkpoint file identity failed.")
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint["config"] != config.to_dict() or checkpoint["model_sha256"] != manifest["model"]["raw_sha256"] or checkpoint["ema_sha256"] != manifest["model"]["ema_sha256"]:
        raise ValueError("Prior-v2 checkpoint semantics drifted.")
    replay = _execute(Path(corpus_root), Path(latent_root), config)
    expected_registry = [{"theme": ref.theme, "shape": list(ref.shape), "full_map_identity_sha256": ref.full_map_identity_sha256} for ref in replay["refs"]]
    expected_metrics = {"unique_samples": replay["unique_samples"], "sample_count": len(replay["refs"]), "sample_tokens_sha256": replay["sample_tokens_sha256"], "sample_uncertainty_sha256": replay["sample_uncertainty_sha256"]}
    if manifest["sample_registry"] != expected_registry or manifest["history"] != replay["history"] or manifest["diagnostics"] != replay["diagnostics"] or manifest["metrics"] != expected_metrics:
        raise ValueError("Prior-v2 smoke exact semantic replay failed.")
    if manifest["model"] != {"initial_sha256": replay["initial_sha256"], "raw_sha256": replay["model_sha256"], "ema_sha256": replay["ema_sha256"]}:
        raise ValueError("Prior-v2 smoke model replay failed.")
    return manifest
