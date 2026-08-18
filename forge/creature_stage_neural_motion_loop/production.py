from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Final

from jsonschema import Draft202012Validator
import numpy as np
import torch

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_neural_motion.dataset import NativeMotionTeacher
from ..creature_stage_neural_motion.model import CellularMotionTransformer
from ..creature_stage_neural_motion.training import (
    _atomic_bytes,
    _atomic_torch,
    _canonical,
    _config,
    _sha256_file,
    _state_sha256,
)
from ..creature_stage_neural_motion_rollout.contract import source_sha256 as rollout_source_sha256
from ..creature_stage_neural_motion_rollout.training import (
    PRODUCTION_FORMAT as ROLLOUT_PRODUCTION_FORMAT,
    _load_rollout_checkpoint,
    _rollout,
)
from .contract import DEFAULT_PARENT, LoopTrainingConfig, source_sha256 as loop_source_sha256
from .sampler import LoopAwareRolloutBatchSampler
from .smoke import _base_config


PRODUCTION_FORMAT: Final[str] = "nullvector-creature-stage-neural-motion-loop-production-v1"
CHECKPOINT_FORMAT: Final[str] = "nullvector-creature-stage-neural-motion-loop-checkpoint-v1"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/creature_stage_neural_motion_loop/production_v1"
SCHEMA: Final[Path] = PROJECT_ROOT / "shared/schema/creature_stage_neural_motion_loop_production.schema.json"
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/creature_stage_neural_motion_loop/production.py",
    "shared/schema/creature_stage_neural_motion_loop_production.schema.json",
)
SEED: Final[int] = 0x4C4F4F5050524F31
MAX_CHECKPOINT_BYTES: Final[int] = 2 * 1024**3
HISTORY_KEYS: Final[set[str]] = {
    "step", "loss", "position", "velocity", "graph", "appendage", "delta",
    "energy", "outside", "gradient_norm", "lr", "seam_sequences",
}


def production_source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-motion-loop-production-source-v1\0")
    digest.update(loop_source_sha256().encode("ascii") + b"\0")
    digest.update(rollout_source_sha256().encode("ascii") + b"\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("loop motion production path must remain inside the project") from error


def _semantic(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _read_json(path: Path, *, maximum_bytes: int = 2 * 1024**2) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum_bytes:
        raise ValueError("loop motion production JSON is missing, linked, or oversized")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if raw != _canonical(payload) or not isinstance(payload, dict):
        raise ValueError("loop motion production JSON is not canonical")
    return payload


def _load_parent(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    contract = _read_json(path.parent / "production_contract.json")
    if (
        contract.get("format") != ROLLOUT_PRODUCTION_FORMAT
        or contract.get("source_sha256") != rollout_source_sha256()
        or contract.get("semantic_sha256")
        != _semantic({key: value for key, value in contract.items() if key != "semantic_sha256"})
    ):
        raise ValueError("loop motion parent rollout contract drifted")
    prefix = "cell_motion_rollout_"
    if not path.stem.startswith(prefix) or not path.stem[len(prefix):].isdigit():
        raise ValueError("loop motion parent rollout checkpoint name drifted")
    update = int(path.stem[len(prefix):])
    checkpoint = _load_rollout_checkpoint(path, contract, update)
    if update != 1_000 or path != DEFAULT_PARENT.resolve():
        raise ValueError("loop motion production must initialize from sealed rollout update 1000")
    return contract, checkpoint


def _validate_contract(contract: dict[str, Any]) -> None:
    errors = sorted(
        Draft202012Validator(json.loads(SCHEMA.read_bytes())).iter_errors(contract),
        key=lambda error: list(error.path),
    )
    if errors:
        raise ValueError(f"loop motion production schema drifted: {errors[0].message}")
    if (
        contract.get("format") != PRODUCTION_FORMAT
        or contract.get("source_sha256") != production_source_sha256()
        or contract.get("loop_source_sha256") != loop_source_sha256()
        or contract.get("rollout_source_sha256") != rollout_source_sha256()
        or contract.get("semantic_sha256")
        != _semantic({key: value for key, value in contract.items() if key != "semantic_sha256"})
        or contract.get("training") != LoopTrainingConfig().to_dict()
        or contract.get("total_updates") != contract.get("segment_updates")
    ):
        raise ValueError("loop motion production authority drifted")
    parent_contract, parent = _load_parent(PROJECT_ROOT / contract["parent"]["path"])
    expected_parent = {
        "path": _relative(DEFAULT_PARENT),
        "sha256": _sha256_file(DEFAULT_PARENT),
        "update": int(parent["update"]),
        "model_state_sha256": parent["model_state_sha256"],
        "ema_state_sha256": parent["ema_state_sha256"],
        "contract_semantic_sha256": parent_contract["semantic_sha256"],
    }
    if (
        contract["parent"] != expected_parent
        or contract["teacher"] != parent_contract["teacher"]
        or contract["model"] != parent_contract["model"]
    ):
        raise ValueError("loop motion production parent provenance drifted")


def prepare_production(
    output: Path = DEFAULT_OUTPUT,
    *,
    parent_checkpoint: Path = DEFAULT_PARENT,
    total_updates: int = 500,
    batch_size: int = 5,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if (
        type(total_updates) is not int or not 100 <= total_updates <= 1_000
        or total_updates % 100
        or type(batch_size) is not int or not 5 <= batch_size <= 20 or batch_size % 5
    ):
        raise ValueError("loop motion bounded production schedule drifted")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    parent_contract, parent = _load_parent(parent_checkpoint)
    contract: dict[str, Any] = {
        "format": PRODUCTION_FORMAT,
        "source_sha256": production_source_sha256(),
        "loop_source_sha256": loop_source_sha256(),
        "rollout_source_sha256": rollout_source_sha256(),
        "parent": {
            "path": _relative(Path(parent_checkpoint)),
            "sha256": _sha256_file(Path(parent_checkpoint)),
            "update": int(parent["update"]),
            "model_state_sha256": parent["model_state_sha256"],
            "ema_state_sha256": parent["ema_state_sha256"],
            "contract_semantic_sha256": parent_contract["semantic_sha256"],
        },
        "teacher": dict(parent_contract["teacher"]),
        "model": dict(parent_contract["model"]),
        "training": LoopTrainingConfig().to_dict(),
        "seed": SEED,
        "total_updates": total_updates,
        "segment_updates": total_updates,
        "batch_size": batch_size,
        "optimizer": {
            "name": "AdamW", "lr": 4e-5, "weight_decay": 1e-5,
            "warmup_updates": min(100, total_updates // 4), "gradient_clip": 1.0,
        },
        "ema_decay": 0.999,
        "precision": "bf16-autocast-float32-loss",
        "minimum_free_vram_bytes": 6 * 1024**3,
        "minimum_free_disk_bytes": 100 * 1024**3,
    }
    contract["semantic_sha256"] = _semantic(contract)
    _validate_contract(contract)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "production_contract.json"
    encoded = _canonical(contract)
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise ValueError("loop motion production contract changed during resume")
    else:
        _atomic_bytes(destination, encoded)
    return contract


def checkpoint_name(update: int) -> str:
    return f"cell_motion_loop_{update:07d}.pt"


def load_checkpoint(path: Path, contract: dict[str, Any], expected_update: int) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("loop motion checkpoint is missing, linked, or oversized")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "format", "source_sha256", "contract_semantic_sha256", "update", "model_state",
        "ema_state", "optimizer_state", "model_state_sha256", "ema_state_sha256",
        "cpu_rng_state", "cuda_rng_state", "history", "runtime",
    }
    if (
        not isinstance(payload, dict) or set(payload) != required
        or payload["format"] != CHECKPOINT_FORMAT
        or payload["source_sha256"] != production_source_sha256()
        or payload["contract_semantic_sha256"] != contract["semantic_sha256"]
        or payload["update"] != expected_update or len(payload["history"]) != expected_update
    ):
        raise ValueError("loop motion checkpoint contract drifted")
    model = CellularMotionTransformer(_config(contract["model"]))
    model.load_state_dict(payload["model_state"], strict=True)
    if _state_sha256(model.state_dict()) != payload["model_state_sha256"]:
        raise ValueError("loop motion model state hash drifted")
    model.load_state_dict(payload["ema_state"], strict=True)
    if _state_sha256(model.state_dict()) != payload["ema_state_sha256"]:
        raise ValueError("loop motion EMA state hash drifted")
    for index, row in enumerate(payload["history"], 1):
        if (
            set(row) != HISTORY_KEYS or row["step"] != index
            or any(not math.isfinite(float(value)) for key, value in row.items() if key != "step")
        ):
            raise ValueError("loop motion checkpoint history drifted")
    return payload


def train_segment(output: Path = DEFAULT_OUTPUT, *, end_update: int | None = None) -> dict[str, Any]:
    output = Path(output).resolve()
    contract = _read_json(output / "production_contract.json")
    _validate_contract(contract)
    target = int(contract["total_updates"] if end_update is None else end_update)
    if target != contract["total_updates"]:
        raise ValueError("loop motion v1 permits exactly one bounded segment")
    destination = output / checkpoint_name(target)
    if destination.exists():
        checked = load_checkpoint(destination, contract, target)
        return {"passed": True, "update": target, "reused": True, "model_state_sha256": checked["model_state_sha256"], "ema_state_sha256": checked["ema_state_sha256"]}
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
        or torch.cuda.mem_get_info(0)[0] < contract["minimum_free_vram_bytes"]
    ):
        raise RuntimeError("loop motion training requires deterministic CUDA BF16 and 6 GiB free VRAM")
    require_disk_floor(output, floor_gb=100, planned_bytes=1024**3)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(contract["seed"])
    torch.cuda.manual_seed_all(contract["seed"])
    np.random.seed(contract["seed"] & 0xFFFFFFFF)
    device = torch.device("cuda", 0)
    torch.cuda.reset_peak_memory_stats(device)
    model = CellularMotionTransformer(_config(contract["model"])).to(device)
    parent_contract, parent = _load_parent(PROJECT_ROOT / contract["parent"]["path"])
    if parent_contract["semantic_sha256"] != contract["parent"]["contract_semantic_sha256"]:
        raise ValueError("loop motion parent changed during training")
    model.load_state_dict(parent["ema_state"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=contract["optimizer"]["lr"], weight_decay=contract["optimizer"]["weight_decay"])
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    teacher = NativeMotionTeacher(PROJECT_ROOT / contract["teacher"]["path"])
    if teacher.semantic_sha256 != contract["teacher"]["semantic_sha256"]:
        raise ValueError("loop motion teacher changed during training")
    training = LoopTrainingConfig(**contract["training"])
    sampler = LoopAwareRolloutBatchSampler(teacher, batch_size=contract["batch_size"], config=training, seed=contract["seed"])
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    model.train()
    for update in range(target):
        frames, coordinates = sampler.sequence(update, device)
        lr = contract["optimizer"]["lr"] * min(1.0, (update + 1) / max(1, contract["optimizer"]["warmup_updates"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        pieces, _ = _rollout(model, frames, _base_config(training), backward=True)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), contract["optimizer"]["gradient_clip"])
        if not math.isfinite(float(gradient)):
            raise FloatingPointError("loop motion production became non-finite")
        optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if value.dtype.is_floating_point:
                    ema[name].lerp_(value.detach(), 1.0 - contract["ema_decay"])
                else:
                    ema[name].copy_(value)
        history.append({"step": update + 1, **{name: round(float(value), 9) for name, value in pieces.items()}, "gradient_norm": round(float(gradient), 9), "lr": round(float(lr), 12), "seam_sequences": sum(int(row.forced_seam) for row in coordinates)})
    elapsed = time.perf_counter() - started
    model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    ema_state = {name: value.detach().cpu().clone() for name, value in ema.items()}
    payload = {
        "format": CHECKPOINT_FORMAT, "source_sha256": production_source_sha256(),
        "contract_semantic_sha256": contract["semantic_sha256"], "update": target,
        "model_state": model_state, "ema_state": ema_state, "optimizer_state": optimizer.state_dict(),
        "model_state_sha256": _state_sha256(model_state), "ema_state_sha256": _state_sha256(ema_state),
        "cpu_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state(device),
        "history": history,
        "runtime": {"segment_seconds": round(elapsed, 6), "updates_per_second": round(target / elapsed, 6), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "device": torch.cuda.get_device_name(device), "torch": str(torch.__version__)},
    }
    _atomic_torch(destination, payload)
    checked = load_checkpoint(destination, contract, target)
    return {"passed": True, "update": target, "reused": False, "model_state_sha256": checked["model_state_sha256"], "ema_state_sha256": checked["ema_state_sha256"], "checkpoint_sha256": _sha256_file(destination), "runtime": checked["runtime"]}
