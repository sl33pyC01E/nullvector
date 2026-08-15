from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Final

from jsonschema import Draft202012Validator
import numpy as np
import torch
from torch import Tensor

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_neural_motion.training import _state_sha256
from ..creature_stage_developmental_motion.dataset import DevelopmentalMotionTeacher, DevelopmentalSequenceSampler, project_path
from ..creature_stage_developmental_motion.training import atomic_bytes, canonical, read_json, semantic, sha256_file
from ..creature_stage_developmental_actuator_v2.contract import CausalActuatorConfig, CausalTrainingConfig
from ..creature_stage_developmental_actuator_v2.training import (
    _validate_contract as validate_v2_contract,
    load_checkpoint as load_v2_checkpoint,
    rollout_sequence,
)
from .contract import (
    CHECKPOINT_FORMAT,
    DEFAULT_OUTPUT,
    DEFAULT_V2_OUTPUT,
    DEFAULT_V2_SEED,
    PRODUCTION_FORMAT,
    PRODUCTION_SCHEMA,
    SEED,
    BoneProjectionConfig,
    source_sha256,
)
from .model import LengthProjectedCellularActuator


MAX_CHECKPOINT_BYTES: Final[int] = 2 * 1024**3
HISTORY_KEYS: Final[set[str]] = {
    "step", "loss", "cell_position", "cell_velocity", "node_position", "node_velocity",
    "muscle", "bone_length", "appendage", "anti_copy", "acceleration", "parent_prior",
    "outside", "muscle_l1", "muscle_velocity", "muscle_force", "seam", "gradient_norm",
    "lr", "teacher_forcing", "seam_sequences", "previous_muscle_gate", "force_gate",
}


def checkpoint_name(update: int) -> str:
    return f"length_projected_actuator_{update:07d}.pt"


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_bytes(path, buffer.getvalue())


def make_model(
    config: CausalActuatorConfig | None = None,
    projection: BoneProjectionConfig | None = None,
) -> LengthProjectedCellularActuator:
    return LengthProjectedCellularActuator(config or CausalActuatorConfig(), projection or BoneProjectionConfig())


def model_state(model: LengthProjectedCellularActuator) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def load_model_state(model: LengthProjectedCellularActuator, state: dict[str, Tensor]) -> None:
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("length-projected model state registry drifted")


def _v2_seed_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_json(DEFAULT_V2_OUTPUT / "production_contract.json")
    validate_v2_contract(contract)
    checkpoint = load_v2_checkpoint(DEFAULT_V2_SEED, contract, 1_200)
    return contract, checkpoint


def _validate_contract(contract: dict[str, Any]) -> None:
    errors = sorted(
        Draft202012Validator(json.loads(PRODUCTION_SCHEMA.read_text(encoding="utf-8"))).iter_errors(contract),
        key=lambda error: list(error.path),
    )
    if errors:
        raise ValueError(f"length-projected production schema drifted: {errors[0].message}")
    if (
        contract.get("format") != PRODUCTION_FORMAT or contract.get("source_sha256") != source_sha256()
        or contract.get("semantic_sha256") != semantic({key: value for key, value in contract.items() if key != "semantic_sha256"})
        or contract.get("model") != CausalActuatorConfig().to_dict()
        or contract.get("projection") != BoneProjectionConfig().to_dict()
        or contract.get("training") != CausalTrainingConfig().to_dict()
        or contract.get("finetune_teacher_forcing") != {"start": .05, "end": 0.0}
        or contract.get("total_updates", 0) % contract.get("segment_updates", 1)
    ):
        raise ValueError("length-projected production authority drifted")
    teacher = DevelopmentalMotionTeacher(
        PROJECT_ROOT / contract["teacher"]["path"], prior=PROJECT_ROOT / contract["prior"]["path"], replay=False,
    )
    if teacher.semantic_sha256 != contract["teacher"]["semantic_sha256"] or teacher.prior_semantic_sha256 != contract["prior"]["semantic_sha256"]:
        raise ValueError("length-projected teacher authority drifted")
    v2_contract, v2_checkpoint = _v2_seed_authority()
    expected = {
        "path": project_path(DEFAULT_V2_SEED), "sha256": sha256_file(DEFAULT_V2_SEED), "update": 1_200,
        "ema_state_sha256": v2_checkpoint["ema_state_sha256"],
        "contract_semantic_sha256": v2_contract["semantic_sha256"],
    }
    if contract["v2_seed"] != expected:
        raise ValueError("length-projected v2 seed drifted")


def prepare_production(
    output: Path = DEFAULT_OUTPUT,
    *,
    total_updates: int = 400,
    segment_updates: int = 50,
    batch_size: int = 5,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if (
        type(total_updates) is not int or not 200 <= total_updates <= 3_000
        or type(segment_updates) is not int or not 25 <= segment_updates <= 100
        or total_updates % segment_updates
        or type(batch_size) is not int or not 5 <= batch_size <= 20 or batch_size % 5
    ):
        raise ValueError("length-projected bounded schedule drifted")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    v2_contract, v2_checkpoint = _v2_seed_authority()
    teacher = DevelopmentalMotionTeacher(
        PROJECT_ROOT / v2_contract["teacher"]["path"], prior=PROJECT_ROOT / v2_contract["prior"]["path"], replay=True,
    )
    contract: dict[str, Any] = {
        "format": PRODUCTION_FORMAT, "source_sha256": source_sha256(),
        "teacher": {"path": project_path(teacher.root), "semantic_sha256": teacher.semantic_sha256},
        "prior": {"path": v2_contract["prior"]["path"], "semantic_sha256": teacher.prior_semantic_sha256},
        "v2_seed": {
            "path": project_path(DEFAULT_V2_SEED), "sha256": sha256_file(DEFAULT_V2_SEED), "update": 1_200,
            "ema_state_sha256": v2_checkpoint["ema_state_sha256"], "contract_semantic_sha256": v2_contract["semantic_sha256"],
        },
        "model": CausalActuatorConfig().to_dict(), "projection": BoneProjectionConfig().to_dict(),
        "training": CausalTrainingConfig().to_dict(), "finetune_teacher_forcing": {"start": .05, "end": 0.0},
        "seed": SEED, "total_updates": total_updates, "segment_updates": segment_updates, "batch_size": batch_size,
        "optimizer": {"name": "AdamW", "lr": 4e-5, "weight_decay": 1e-5, "warmup_updates": 25, "gradient_clip": 1.0},
        "ema_decay": .997, "precision": "bf16-autocast-float32-loss",
        "minimum_free_vram_bytes": 8 * 1024**3, "minimum_free_disk_bytes": 100 * 1024**3,
    }
    contract["semantic_sha256"] = semantic(contract)
    _validate_contract(contract)
    output.mkdir(parents=True, exist_ok=True)
    encoded = canonical(contract)
    path = output / "production_contract.json"
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("length-projected production contract changed during resume")
    else:
        atomic_bytes(path, encoded)
    return contract


def load_checkpoint(path: Path, contract: dict[str, Any], expected_update: int) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("length-projected checkpoint is missing, linked, or oversized")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format", "source_sha256", "contract_semantic_sha256", "update", "model_state", "ema_state", "optimizer_state", "model_state_sha256", "ema_state_sha256", "cpu_rng_state", "cuda_rng_state", "history", "runtime"}
    if (
        not isinstance(payload, dict) or set(payload) != required or payload["format"] != CHECKPOINT_FORMAT
        or payload["source_sha256"] != source_sha256() or payload["contract_semantic_sha256"] != contract["semantic_sha256"]
        or payload["update"] != expected_update or len(payload["history"]) != expected_update
    ):
        raise ValueError("length-projected checkpoint contract drifted")
    probe = make_model(CausalActuatorConfig(**contract["model"]), BoneProjectionConfig(**contract["projection"]))
    load_model_state(probe, payload["model_state"])
    if _state_sha256(model_state(probe)) != payload["model_state_sha256"]:
        raise ValueError("length-projected model hash drifted")
    load_model_state(probe, payload["ema_state"])
    if _state_sha256(model_state(probe)) != payload["ema_state_sha256"]:
        raise ValueError("length-projected EMA hash drifted")
    for index, row in enumerate(payload["history"], 1):
        if set(row) != HISTORY_KEYS or row["step"] != index or any(not math.isfinite(float(value)) for key, value in row.items() if key != "step"):
            raise ValueError("length-projected checkpoint history drifted")
    return payload


def _latest_update(output: Path, contract: dict[str, Any]) -> int:
    updates = [int(path.stem.removeprefix("length_projected_actuator_")) for path in output.glob("length_projected_actuator_*.pt") if path.stem.removeprefix("length_projected_actuator_").isdigit()]
    if not updates:
        return 0
    latest = max(updates)
    if latest % int(contract["segment_updates"]) or latest > int(contract["total_updates"]):
        raise ValueError("length-projected schedule drifted")
    load_checkpoint(output / checkpoint_name(latest), contract, latest)
    return latest


def train_next_segment(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output).resolve()
    contract = read_json(output / "production_contract.json")
    _validate_contract(contract)
    start = _latest_update(output, contract)
    if start >= int(contract["total_updates"]):
        checkpoint = load_checkpoint(output / checkpoint_name(start), contract, start)
        return {"passed": True, "complete": True, "update": start, "reused": True, "ema_state_sha256": checkpoint["ema_state_sha256"]}
    end = min(start + int(contract["segment_updates"]), int(contract["total_updates"]))
    destination = output / checkpoint_name(end)
    if destination.exists():
        raise FileExistsError(destination)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(0)[0] < int(contract["minimum_free_vram_bytes"]):
        raise RuntimeError("length-projected training requires deterministic CUDA BF16 and 8 GiB free VRAM")
    require_disk_floor(output, floor_gb=100, planned_bytes=1024**3)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(contract["seed"])); torch.cuda.manual_seed_all(int(contract["seed"])); np.random.seed(int(contract["seed"]) & 0xFFFFFFFF)
    device = torch.device("cuda", 0); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    model = make_model(CausalActuatorConfig(**contract["model"]), BoneProjectionConfig(**contract["projection"])).to(device)
    _, v2_checkpoint = _v2_seed_authority()
    load_model_state(model, v2_checkpoint["ema_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(contract["optimizer"]["lr"]), weight_decay=float(contract["optimizer"]["weight_decay"]))
    history: list[dict[str, float | int]] = []
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if start:
        previous = load_checkpoint(output / checkpoint_name(start), contract, start)
        load_model_state(model, previous["model_state"]); optimizer.load_state_dict(previous["optimizer_state"])
        ema = {name: value.to(device=device).clone() for name, value in previous["ema_state"].items()}
        for state in optimizer.state.values():
            for name, value in tuple(state.items()):
                if isinstance(value, Tensor): state[name] = value.to(device=device)
        torch.set_rng_state(previous["cpu_rng_state"]); torch.cuda.set_rng_state_all(previous["cuda_rng_state"]); history = list(previous["history"])
    teacher = DevelopmentalMotionTeacher(PROJECT_ROOT / contract["teacher"]["path"], prior=PROJECT_ROOT / contract["prior"]["path"], replay=False)
    training = CausalTrainingConfig(**contract["training"])
    sampler = DevelopmentalSequenceSampler(teacher, batch_size=int(contract["batch_size"]), sequence_frames=training.sequence_frames, seed=int(contract["seed"]), seam_numerator=training.seam_quota_numerator, seam_denominator=training.seam_quota_denominator)
    started = time.perf_counter(); model.train()
    for update in range(start, end):
        progress = update / max(1, int(contract["total_updates"]) - 1)
        teacher_forcing = float(contract["finetune_teacher_forcing"]["start"]) * (1.0 - progress)
        frames, coordinates = sampler.sequence(update, device)
        lr = float(contract["optimizer"]["lr"]) * min(1.0, (update + 1) / int(contract["optimizer"]["warmup_updates"]))
        for group in optimizer.param_groups: group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        pieces, _ = rollout_sequence(model, frames, training, teacher_forcing=teacher_forcing, backward=True)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), float(contract["optimizer"]["gradient_clip"]))
        if not math.isfinite(float(gradient)) or float(gradient) <= 0.0: raise FloatingPointError("length-projected gradient became non-finite or zero")
        optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if value.dtype.is_floating_point: ema[name].lerp_(value, 1.0 - float(contract["ema_decay"]))
                else: ema[name].copy_(value)
        history.append({"step": update + 1, **{name: round(float(value), 9) for name, value in pieces.items()}, "gradient_norm": round(float(gradient), 9), "lr": round(float(lr), 12), "teacher_forcing": round(float(teacher_forcing), 9), "seam_sequences": sum(int(item.forced_seam) for item in coordinates), "previous_muscle_gate": round(float(torch.sigmoid(model.actuator.previous_muscle_gate)), 9), "force_gate": round(float(torch.sigmoid(model.actuator.force_gate)), 9)})
    elapsed = time.perf_counter() - started; state = model_state(model)
    payload = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "contract_semantic_sha256": contract["semantic_sha256"], "update": end, "model_state": state, "ema_state": {name: value.detach().cpu().clone() for name, value in ema.items()}, "optimizer_state": optimizer.state_dict(), "model_state_sha256": _state_sha256(state), "ema_state_sha256": _state_sha256(ema), "cpu_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all(), "history": history, "runtime": {"segment_start": start, "segment_end": end, "elapsed_seconds": round(elapsed, 6), "updates_per_second": round((end-start)/max(elapsed,1e-9),6), "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "torch": str(torch.__version__), "cuda": str(torch.version.cuda or "none"), "device": torch.cuda.get_device_name(device)}}
    atomic_torch(destination, payload); checked = load_checkpoint(destination, contract, end)
    return {"passed": True, "complete": end == int(contract["total_updates"]), "update": end, "reused": False, "checkpoint": project_path(destination), "checkpoint_sha256": sha256_file(destination), "model_state_sha256": checked["model_state_sha256"], "ema_state_sha256": checked["ema_state_sha256"], "runtime": checked["runtime"]}
