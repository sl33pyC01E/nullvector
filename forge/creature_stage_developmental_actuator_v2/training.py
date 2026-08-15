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
from ..creature_stage_developmental_motion.dataset import (
    DevelopmentalMotionTeacher,
    DevelopmentalSequenceSampler,
    project_path,
)
from ..creature_stage_developmental_motion.training import (
    _validate_contract as validate_v1_contract,
    atomic_bytes,
    canonical,
    load_checkpoint as load_v1_checkpoint,
    read_json,
    semantic,
    sha256_file,
)
from .contract import (
    CHECKPOINT_FORMAT,
    DEFAULT_OUTPUT,
    DEFAULT_PRIOR,
    DEFAULT_TEACHER,
    DEFAULT_V1_OUTPUT,
    DEFAULT_V1_SEED,
    PRODUCTION_FORMAT,
    PRODUCTION_SCHEMA,
    SEED,
    CausalActuatorConfig,
    CausalTrainingConfig,
    source_sha256,
)
from .model import MuscleCausalCellularActuator, causal_actuator_loss


MAX_CHECKPOINT_BYTES: Final[int] = 2 * 1024**3
HISTORY_KEYS: Final[set[str]] = {
    "step", "loss", "cell_position", "cell_velocity", "node_position", "node_velocity",
    "muscle", "bone_length", "appendage", "anti_copy", "acceleration", "parent_prior",
    "outside", "muscle_l1", "muscle_velocity", "muscle_force", "seam", "gradient_norm",
    "lr", "teacher_forcing", "seam_sequences", "previous_muscle_gate", "force_gate",
}


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_bytes(path, buffer.getvalue())


def checkpoint_name(update: int) -> str:
    return f"muscle_causal_actuator_{update:07d}.pt"


def _v1_seed_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_json(DEFAULT_V1_OUTPUT / "production_contract.json")
    validate_v1_contract(contract)
    checkpoint = load_v1_checkpoint(DEFAULT_V1_SEED, contract, 2_000)
    return contract, checkpoint


def make_model(config: CausalActuatorConfig | None = None) -> MuscleCausalCellularActuator:
    return MuscleCausalCellularActuator(config or CausalActuatorConfig())


def model_state(model: MuscleCausalCellularActuator) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def load_model_state(model: MuscleCausalCellularActuator, state: dict[str, Tensor]) -> None:
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("muscle-causal model state registry drifted")


def warm_start_from_v1(model: MuscleCausalCellularActuator, state: dict[str, Tensor]) -> dict[str, Any]:
    current = model.state_dict()
    compatible = {
        name: value for name, value in state.items()
        if name in current and current[name].shape == value.shape and current[name].dtype == value.dtype
    }
    result = model.load_state_dict(compatible, strict=False)
    expected_new = {
        "actuator.previous_muscle_gate", "actuator.force_gate",
        "actuator.force_to_node.0.weight", "actuator.force_to_node.0.bias",
        "actuator.force_to_node.2.weight", "actuator.force_to_node.2.bias",
    }
    if set(result.missing_keys) != expected_new or result.unexpected_keys:
        raise ValueError("muscle-causal v1 transfer registry drifted")
    return {
        "transferred_tensors": len(compatible),
        "new_tensors": sorted(expected_new),
        "transferred_elements": sum(int(value.numel()) for value in compatible.values()),
    }


def _validate_contract(contract: dict[str, Any]) -> None:
    errors = sorted(
        Draft202012Validator(json.loads(PRODUCTION_SCHEMA.read_text(encoding="utf-8"))).iter_errors(contract),
        key=lambda error: list(error.path),
    )
    if errors:
        raise ValueError(f"muscle-causal production schema drifted: {errors[0].message}")
    if (
        contract.get("format") != PRODUCTION_FORMAT
        or contract.get("source_sha256") != source_sha256()
        or contract.get("semantic_sha256")
        != semantic({key: value for key, value in contract.items() if key != "semantic_sha256"})
        or contract.get("model") != CausalActuatorConfig().to_dict()
        or contract.get("training") != CausalTrainingConfig().to_dict()
        or contract.get("total_updates", 0) % contract.get("segment_updates", 1)
    ):
        raise ValueError("muscle-causal production authority drifted")
    teacher = DevelopmentalMotionTeacher(
        PROJECT_ROOT / contract["teacher"]["path"],
        prior=PROJECT_ROOT / contract["prior"]["path"], replay=False,
    )
    if teacher.semantic_sha256 != contract["teacher"]["semantic_sha256"]:
        raise ValueError("muscle-causal teacher drifted")
    if teacher.prior_semantic_sha256 != contract["prior"]["semantic_sha256"]:
        raise ValueError("muscle-causal prior drifted")
    v1_contract, v1_checkpoint = _v1_seed_authority()
    probe = make_model()
    transfer = warm_start_from_v1(probe, v1_checkpoint["ema_state"])
    expected_seed = {
        "path": project_path(DEFAULT_V1_SEED), "sha256": sha256_file(DEFAULT_V1_SEED),
        "update": 2_000, "ema_state_sha256": v1_checkpoint["ema_state_sha256"],
        "contract_semantic_sha256": v1_contract["semantic_sha256"],
        "transferred_tensors": transfer["transferred_tensors"],
    }
    if contract["v1_seed"] != expected_seed:
        raise ValueError("muscle-causal v1 seed provenance drifted")


def prepare_production(
    output: Path = DEFAULT_OUTPUT,
    *,
    teacher_path: Path = DEFAULT_TEACHER,
    prior_path: Path = DEFAULT_PRIOR,
    total_updates: int = 1_200,
    segment_updates: int = 50,
    batch_size: int = 5,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if (
        type(total_updates) is not int or not 500 <= total_updates <= 10_000
        or type(segment_updates) is not int or not 50 <= segment_updates <= 250
        or total_updates % segment_updates
        or type(batch_size) is not int or not 5 <= batch_size <= 20 or batch_size % 5
    ):
        raise ValueError("muscle-causal bounded schedule drifted")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=4 * 1024**3)
    teacher = DevelopmentalMotionTeacher(teacher_path, prior=prior_path, replay=True)
    v1_contract, v1_checkpoint = _v1_seed_authority()
    probe = make_model()
    transfer = warm_start_from_v1(probe, v1_checkpoint["ema_state"])
    contract: dict[str, Any] = {
        "format": PRODUCTION_FORMAT,
        "source_sha256": source_sha256(),
        "teacher": {"path": project_path(teacher.root), "semantic_sha256": teacher.semantic_sha256},
        "prior": {"path": project_path(Path(prior_path)), "semantic_sha256": teacher.prior_semantic_sha256},
        "v1_seed": {
            "path": project_path(DEFAULT_V1_SEED), "sha256": sha256_file(DEFAULT_V1_SEED),
            "update": 2_000, "ema_state_sha256": v1_checkpoint["ema_state_sha256"],
            "contract_semantic_sha256": v1_contract["semantic_sha256"],
            "transferred_tensors": transfer["transferred_tensors"],
        },
        "model": CausalActuatorConfig().to_dict(),
        "training": CausalTrainingConfig().to_dict(),
        "seed": SEED,
        "total_updates": total_updates,
        "segment_updates": segment_updates,
        "batch_size": batch_size,
        "optimizer": {
            "name": "AdamW", "lr": 8e-5, "weight_decay": 1e-5,
            "warmup_updates": min(100, total_updates // 5), "gradient_clip": 1.0,
        },
        "ema_decay": .9985,
        "precision": "bf16-autocast-float32-loss",
        "minimum_free_vram_bytes": 8 * 1024**3,
        "minimum_free_disk_bytes": 100 * 1024**3,
    }
    contract["semantic_sha256"] = semantic(contract)
    _validate_contract(contract)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "production_contract.json"
    encoded = canonical(contract)
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise ValueError("muscle-causal production contract changed during resume")
    else:
        atomic_bytes(destination, encoded)
    return contract


def load_checkpoint(path: Path, contract: dict[str, Any], expected_update: int) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("muscle-causal checkpoint is missing, linked, or oversized")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "format", "source_sha256", "contract_semantic_sha256", "update", "model_state",
        "ema_state", "optimizer_state", "model_state_sha256", "ema_state_sha256",
        "cpu_rng_state", "cuda_rng_state", "history", "runtime",
    }
    if (
        not isinstance(payload, dict) or set(payload) != required
        or payload["format"] != CHECKPOINT_FORMAT or payload["source_sha256"] != source_sha256()
        or payload["contract_semantic_sha256"] != contract["semantic_sha256"]
        or payload["update"] != expected_update or len(payload["history"]) != expected_update
    ):
        raise ValueError("muscle-causal checkpoint contract drifted")
    probe = make_model(CausalActuatorConfig(**contract["model"]))
    load_model_state(probe, payload["model_state"])
    if _state_sha256(model_state(probe)) != payload["model_state_sha256"]:
        raise ValueError("muscle-causal model state hash drifted")
    load_model_state(probe, payload["ema_state"])
    if _state_sha256(model_state(probe)) != payload["ema_state_sha256"]:
        raise ValueError("muscle-causal EMA state hash drifted")
    for index, row in enumerate(payload["history"], 1):
        if (
            set(row) != HISTORY_KEYS or row["step"] != index
            or any(not math.isfinite(float(value)) for key, value in row.items() if key != "step")
        ):
            raise ValueError("muscle-causal checkpoint history drifted")
    return payload


def _latest_update(output: Path, contract: dict[str, Any]) -> int:
    updates = []
    for path in output.glob("muscle_causal_actuator_*.pt"):
        suffix = path.stem.removeprefix("muscle_causal_actuator_")
        if suffix.isdigit():
            updates.append(int(suffix))
    if not updates:
        return 0
    latest = max(updates)
    if latest % int(contract["segment_updates"]) or latest > int(contract["total_updates"]):
        raise ValueError("muscle-causal checkpoint schedule drifted")
    load_checkpoint(output / checkpoint_name(latest), contract, latest)
    return latest


def rollout_sequence(
    model: MuscleCausalCellularActuator,
    frames: list[dict[str, Tensor]],
    training: CausalTrainingConfig,
    *,
    teacher_forcing: float,
    backward: bool,
) -> tuple[dict[str, float], list[dict[str, Tensor]]]:
    if not 0.0 <= teacher_forcing <= .25:
        raise ValueError("muscle-causal teacher forcing drifted")
    cell_state = frames[0]["state"].float()
    node_state = frames[0]["node_state"].float()
    muscle_state = frames[0]["muscle_state"].float()
    names = (
        "loss", "cell_position", "cell_velocity", "node_position", "node_velocity",
        "muscle", "bone_length", "appendage", "anti_copy", "acceleration",
        "parent_prior", "outside", "muscle_l1", "muscle_velocity", "muscle_force",
    )
    totals = {name: 0.0 for name in names}
    outputs = []
    seam_total = 0.0
    for frame in frames:
        with torch.autocast(
            device_type=cell_state.device.type, dtype=torch.bfloat16,
            enabled=cell_state.device.type == "cuda",
        ):
            output = model(
                frame["static"], cell_state, frame["mask"], frame["adjacency"],
                frame["node_features"], node_state, frame["node_mask"], frame["node_adjacency"],
                frame["muscle_features"], muscle_state, frame["muscle_mask"],
                frame["muscle_incidence"], frame["cell_node_weights"], frame["parent_prior"],
                frame["family"], frame["morphotype"], frame["phase"], frame["traits"],
            )
        loss, pieces = causal_actuator_loss(
            output, frame, cell_state.float(), node_state.float(), muscle_state.float(), training,
        )
        seam = torch.zeros((), dtype=torch.float32, device=loss.device)
        seam_rows = frame["frame"] == 0
        if bool(seam_rows.any()):
            seam_mask = seam_rows[:, None, None] & frame["mask"][:, :, None]
            difference = torch.nn.functional.smooth_l1_loss(
                output["cell_state"].float()[:, :, :2], frame["target"].float()[:, :, :2], reduction="none",
            )
            seam = difference.masked_select(seam_mask.expand_as(difference)).mean()
            loss = loss + seam * training.seam_weight
        if backward:
            (loss / len(frames)).backward()
        for name in names:
            totals[name] += float(pieces[name])
        seam_total += float(seam)
        outputs.append({name: value.detach() for name, value in output.items()})
        blend = float(teacher_forcing)
        cell_state = output["cell_state"].detach() * (1.0 - blend) + frame["target"].float() * blend
        node_state = output["node_state"].detach() * (1.0 - blend) + frame["node_target"].float() * blend
        muscle_state = output["muscle_activation"].detach() * (1.0 - blend) + frame["muscle_target"].float() * blend
    metrics = {name: value / len(frames) for name, value in totals.items()}
    metrics["seam"] = seam_total / len(frames)
    metrics["loss"] += metrics["seam"] * training.seam_weight
    return metrics, outputs


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
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
        or torch.cuda.mem_get_info(0)[0] < int(contract["minimum_free_vram_bytes"])
    ):
        raise RuntimeError("muscle-causal training requires deterministic CUDA BF16 and 8 GiB free VRAM")
    require_disk_floor(output, floor_gb=100, planned_bytes=1024**3)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(contract["seed"]))
    torch.cuda.manual_seed_all(int(contract["seed"]))
    np.random.seed(int(contract["seed"]) & 0xFFFFFFFF)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    model = make_model(CausalActuatorConfig(**contract["model"])).to(device)
    v1_contract, v1_checkpoint = _v1_seed_authority()
    del v1_contract
    transfer = warm_start_from_v1(model, v1_checkpoint["ema_state"])
    if transfer["transferred_tensors"] != contract["v1_seed"]["transferred_tensors"]:
        raise ValueError("muscle-causal warm start transfer count drifted")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(contract["optimizer"]["lr"]),
        weight_decay=float(contract["optimizer"]["weight_decay"]),
    )
    history: list[dict[str, float | int]] = []
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if start:
        previous = load_checkpoint(output / checkpoint_name(start), contract, start)
        load_model_state(model, previous["model_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        ema = {name: value.to(device=device).clone() for name, value in previous["ema_state"].items()}
        for state in optimizer.state.values():
            for name, value in tuple(state.items()):
                if isinstance(value, Tensor):
                    state[name] = value.to(device=device)
        torch.set_rng_state(previous["cpu_rng_state"])
        torch.cuda.set_rng_state_all(previous["cuda_rng_state"])
        history = list(previous["history"])
    teacher = DevelopmentalMotionTeacher(
        PROJECT_ROOT / contract["teacher"]["path"],
        prior=PROJECT_ROOT / contract["prior"]["path"], replay=False,
    )
    training = CausalTrainingConfig(**contract["training"])
    sampler = DevelopmentalSequenceSampler(
        teacher, batch_size=int(contract["batch_size"]), sequence_frames=training.sequence_frames,
        seed=int(contract["seed"]), seam_numerator=training.seam_quota_numerator,
        seam_denominator=training.seam_quota_denominator,
    )
    started = time.perf_counter()
    model.train()
    for update in range(start, end):
        progress = update / max(1, int(contract["total_updates"]) - 1)
        teacher_forcing = training.teacher_forcing_start + (training.teacher_forcing_end - training.teacher_forcing_start) * progress
        frames, coordinates = sampler.sequence(update, device)
        lr = float(contract["optimizer"]["lr"]) * min(1.0, (update + 1) / max(1, int(contract["optimizer"]["warmup_updates"])))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        pieces, _ = rollout_sequence(model, frames, training, teacher_forcing=teacher_forcing, backward=True)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), float(contract["optimizer"]["gradient_clip"]))
        if not math.isfinite(float(gradient)) or float(gradient) <= 0.0:
            raise FloatingPointError("muscle-causal gradient became non-finite or zero")
        optimizer.step()
        with torch.no_grad():
            current = model.state_dict()
            for name, value in current.items():
                if value.dtype.is_floating_point:
                    ema[name].lerp_(value, 1.0 - float(contract["ema_decay"]))
                else:
                    ema[name].copy_(value)
        history.append({
            "step": update + 1,
            **{name: round(float(value), 9) for name, value in pieces.items()},
            "gradient_norm": round(float(gradient), 9), "lr": round(float(lr), 12),
            "teacher_forcing": round(float(teacher_forcing), 9),
            "seam_sequences": sum(int(item.forced_seam) for item in coordinates),
            "previous_muscle_gate": round(float(torch.sigmoid(model.actuator.previous_muscle_gate)), 9),
            "force_gate": round(float(torch.sigmoid(model.actuator.force_gate)), 9),
        })
    elapsed = time.perf_counter() - started
    state = model_state(model)
    payload = {
        "format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(),
        "contract_semantic_sha256": contract["semantic_sha256"], "update": end,
        "model_state": state,
        "ema_state": {name: value.detach().cpu().clone() for name, value in ema.items()},
        "optimizer_state": optimizer.state_dict(),
        "model_state_sha256": _state_sha256(state), "ema_state_sha256": _state_sha256(ema),
        "cpu_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "history": history,
        "runtime": {
            "segment_start": start, "segment_end": end, "elapsed_seconds": round(elapsed, 6),
            "updates_per_second": round((end - start) / max(elapsed, 1e-9), 6),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "torch": str(torch.__version__), "cuda": str(torch.version.cuda or "none"),
            "device": torch.cuda.get_device_name(device),
        },
    }
    atomic_torch(destination, payload)
    checked = load_checkpoint(destination, contract, end)
    return {
        "passed": True, "complete": end == int(contract["total_updates"]), "update": end,
        "reused": False, "checkpoint": project_path(destination), "checkpoint_sha256": sha256_file(destination),
        "model_state_sha256": checked["model_state_sha256"], "ema_state_sha256": checked["ema_state_sha256"],
        "runtime": checked["runtime"],
    }
