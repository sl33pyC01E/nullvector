from __future__ import annotations

import hashlib
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
from ..creature_stage_neural_motion.training import _config as parent_model_config
from ..creature_stage_neural_motion.training import _canonical as parent_canonical
from ..creature_stage_neural_motion.training import _state_sha256
from ..creature_stage_neural_motion_rollout.contract import source_sha256 as rollout_source_sha256
from ..creature_stage_neural_motion_rollout.training import (
    PRODUCTION_FORMAT as ROLLOUT_PRODUCTION_FORMAT,
    _load_rollout_checkpoint,
)
from .contract import (
    CHECKPOINT_FORMAT,
    DEFAULT_CORPUS,
    DEFAULT_OUTPUT,
    DEFAULT_PARENT,
    DEFAULT_PRIOR,
    PRODUCTION_FORMAT,
    PRODUCTION_SCHEMA,
    DevelopmentalActuatorConfig,
    DevelopmentalTrainingConfig,
    source_sha256,
)
from .dataset import DevelopmentalMotionTeacher, DevelopmentalSequenceSampler, project_path
from .model import DevelopmentalCellularMotionTransformer, developmental_actuator_loss


SEED: Final[int] = 0x4445564143545031
MAX_CHECKPOINT_BYTES: Final[int] = 2 * 1024**3
HISTORY_KEYS: Final[set[str]] = {
    "step", "loss", "cell_position", "cell_velocity", "node_position",
    "node_velocity", "muscle", "bone_length", "appendage", "anti_copy",
    "acceleration", "parent_prior", "outside", "seam", "gradient_norm",
    "lr", "teacher_forcing", "seam_sequences",
}


def canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_bytes(path, buffer.getvalue())


def read_json(path: Path, *, maximum_bytes: int = 4 * 1024**2) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum_bytes:
        raise ValueError("developmental actuator JSON is missing, linked, or oversized")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != canonical(payload):
        raise ValueError("developmental actuator JSON is not canonical")
    return payload


def _parent_authority(path: Path = DEFAULT_PARENT) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    contract_path = path.parent / "production_contract.json"
    if contract_path.is_symlink() or not contract_path.is_file() or not 0 < contract_path.stat().st_size <= 4 * 1024**2:
        raise ValueError("developmental actuator rollout parent contract is missing, linked, or oversized")
    parent_raw = contract_path.read_bytes()
    contract = json.loads(parent_raw)
    if not isinstance(contract, dict) or parent_raw != parent_canonical(contract):
        raise ValueError("developmental actuator rollout parent contract is not canonical")
    if (
        contract.get("format") != ROLLOUT_PRODUCTION_FORMAT
        or contract.get("source_sha256") != rollout_source_sha256()
        or contract.get("semantic_sha256")
        != hashlib.sha256(parent_canonical({key: value for key, value in contract.items() if key != "semantic_sha256"})).hexdigest()
    ):
        raise ValueError("developmental actuator rollout parent contract drifted")
    prefix = "cell_motion_rollout_"
    if not path.stem.startswith(prefix) or not path.stem[len(prefix):].isdigit():
        raise ValueError("developmental actuator rollout parent filename drifted")
    update = int(path.stem[len(prefix):])
    checkpoint = _load_rollout_checkpoint(path, contract, update)
    if path != DEFAULT_PARENT.resolve() or update != 1_000:
        raise ValueError("developmental actuator requires the sealed rollout update-1000 authority")
    return contract, checkpoint


def make_model(
    parent_contract: dict[str, Any],
    parent_checkpoint: dict[str, Any],
    config: DevelopmentalActuatorConfig,
) -> DevelopmentalCellularMotionTransformer:
    if parent_checkpoint["update"] != 1_000 or parent_contract["total_updates"] < 1_000:
        raise ValueError("developmental actuator parent EMA registry drifted")
    return DevelopmentalCellularMotionTransformer(config)


def successor_state(model: DevelopmentalCellularMotionTransformer) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("parent.")
    }


def load_successor_state(model: DevelopmentalCellularMotionTransformer, state: dict[str, Tensor]) -> None:
    result = model.load_state_dict(state, strict=False)
    expected_missing = {name for name in model.state_dict() if name.startswith("parent.")}
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise ValueError("developmental actuator successor state registry drifted")


def rollout_sequence(
    model: DevelopmentalCellularMotionTransformer,
    frames: list[dict[str, Tensor]],
    training: DevelopmentalTrainingConfig,
    *,
    teacher_forcing: float,
    backward: bool,
) -> tuple[dict[str, float], list[dict[str, Tensor]]]:
    if not 0.0 <= teacher_forcing <= 1.0:
        raise ValueError("developmental actuator rollout teacher forcing drifted")
    cell_state = frames[0]["state"].float()
    node_state = frames[0]["node_state"].float()
    muscle_state = frames[0]["muscle_state"].float()
    piece_names = (
        "loss", "cell_position", "cell_velocity", "node_position", "node_velocity",
        "muscle", "bone_length", "appendage", "anti_copy", "acceleration",
        "parent_prior", "outside",
    )
    totals = {name: 0.0 for name in piece_names}
    outputs: list[dict[str, Tensor]] = []
    seam_total = 0.0
    for frame in frames:
        with torch.autocast(
            device_type=cell_state.device.type,
            dtype=torch.bfloat16,
            enabled=cell_state.device.type == "cuda",
        ):
            output = model(
                frame["static"], cell_state, frame["mask"], frame["adjacency"],
                frame["node_features"], node_state, frame["node_mask"], frame["node_adjacency"],
                frame["muscle_features"], muscle_state, frame["muscle_mask"],
                frame["muscle_incidence"], frame["cell_node_weights"], frame["parent_prior"],
                frame["family"],
                frame["morphotype"], frame["phase"], frame["traits"],
            )
        loss, pieces = developmental_actuator_loss(
            output, frame, cell_state.float(), node_state.float(), training,
        )
        is_seam = frame["frame"] == 0
        seam = torch.zeros((), dtype=torch.float32, device=loss.device)
        if bool(is_seam.any()):
            seam_mask = is_seam[:, None, None] & frame["mask"][:, :, None]
            difference = torch.nn.functional.smooth_l1_loss(
                output["cell_state"].float()[:, :, :2], frame["target"].float()[:, :, :2], reduction="none",
            )
            seam = difference.masked_select(seam_mask.expand_as(difference)).mean()
            loss = loss + seam * training.seam_weight
        if backward:
            (loss / len(frames)).backward()
        for name in piece_names:
            totals[name] += float(pieces[name])
        seam_total += float(seam)
        outputs.append({name: value.detach() for name, value in output.items()})
        blend = float(teacher_forcing)
        cell_state = (output["cell_state"].detach() * (1.0 - blend) + frame["target"].float() * blend)
        node_state = (output["node_state"].detach() * (1.0 - blend) + frame["node_target"].float() * blend)
        muscle_state = (
            output["muscle_activation"].detach() * (1.0 - blend)
            + frame["muscle_target"].float() * blend
        )
    metrics = {name: value / len(frames) for name, value in totals.items()}
    metrics["seam"] = seam_total / len(frames)
    metrics["loss"] += metrics["seam"] * training.seam_weight
    return metrics, outputs


def _validate_contract(contract: dict[str, Any]) -> None:
    schema = json.loads(PRODUCTION_SCHEMA.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(contract), key=lambda error: list(error.path))
    if errors:
        raise ValueError(f"developmental actuator production schema drifted: {errors[0].message}")
    if (
        contract.get("format") != PRODUCTION_FORMAT
        or contract.get("source_sha256") != source_sha256()
        or contract.get("semantic_sha256")
        != semantic({key: value for key, value in contract.items() if key != "semantic_sha256"})
        or contract.get("model") != DevelopmentalActuatorConfig().to_dict()
        or contract.get("training") != DevelopmentalTrainingConfig().to_dict()
        or contract.get("total_updates", 0) % contract.get("segment_updates", 1)
    ):
        raise ValueError("developmental actuator production authority drifted")
    teacher = DevelopmentalMotionTeacher(
        PROJECT_ROOT / contract["teacher"]["path"],
        prior=PROJECT_ROOT / contract["prior"]["path"], replay=False,
    )
    if teacher.semantic_sha256 != contract["teacher"]["semantic_sha256"]:
        raise ValueError("developmental actuator production teacher drifted")
    if teacher.prior_semantic_sha256 != contract["prior"]["semantic_sha256"]:
        raise ValueError("developmental actuator production parent prior drifted")
    parent_contract, parent = _parent_authority(PROJECT_ROOT / contract["parent"]["path"])
    expected = {
        "path": project_path(DEFAULT_PARENT),
        "sha256": sha256_file(DEFAULT_PARENT),
        "update": 1_000,
        "model_state_sha256": parent["model_state_sha256"],
        "ema_state_sha256": parent["ema_state_sha256"],
        "contract_semantic_sha256": parent_contract["semantic_sha256"],
    }
    if contract["parent"] != expected:
        raise ValueError("developmental actuator production parent provenance drifted")


def prepare_production(
    output: Path = DEFAULT_OUTPUT,
    *,
    corpus: Path = DEFAULT_CORPUS,
    prior: Path = DEFAULT_PRIOR,
    total_updates: int = 2_000,
    segment_updates: int = 250,
    batch_size: int = 10,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if (
        type(total_updates) is not int or not 500 <= total_updates <= 10_000
        or type(segment_updates) is not int or not 50 <= segment_updates <= 500
        or total_updates % segment_updates
        or type(batch_size) is not int or not 5 <= batch_size <= 30 or batch_size % 5
    ):
        raise ValueError("developmental actuator bounded schedule drifted")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=4 * 1024**3)
    teacher = DevelopmentalMotionTeacher(corpus, prior=prior, replay=True)
    parent_contract, parent = _parent_authority(DEFAULT_PARENT)
    config = DevelopmentalActuatorConfig()
    training = DevelopmentalTrainingConfig()
    contract: dict[str, Any] = {
        "format": PRODUCTION_FORMAT,
        "source_sha256": source_sha256(),
        "teacher": {
            "path": project_path(teacher.root),
            "semantic_sha256": teacher.semantic_sha256,
            "corpus_semantic_sha256": teacher.manifest["semantic_sha256"],
        },
        "parent": {
            "path": project_path(DEFAULT_PARENT),
            "sha256": sha256_file(DEFAULT_PARENT),
            "update": 1_000,
            "model_state_sha256": parent["model_state_sha256"],
            "ema_state_sha256": parent["ema_state_sha256"],
            "contract_semantic_sha256": parent_contract["semantic_sha256"],
        },
        "prior": {
            "path": project_path(Path(prior)),
            "semantic_sha256": teacher.prior_semantic_sha256,
        },
        "model": config.to_dict(),
        "training": training.to_dict(),
        "seed": SEED,
        "total_updates": total_updates,
        "segment_updates": segment_updates,
        "batch_size": batch_size,
        "optimizer": {
            "name": "AdamW", "lr": 1.5e-4, "weight_decay": 1e-5,
            "warmup_updates": min(200, total_updates // 5), "gradient_clip": 1.0,
        },
        "ema_decay": 0.999,
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
            raise ValueError("developmental actuator production contract changed during resume")
    else:
        atomic_bytes(destination, encoded)
    return contract


def checkpoint_name(update: int) -> str:
    return f"developmental_actuator_{update:07d}.pt"


def load_checkpoint(path: Path, contract: dict[str, Any], expected_update: int) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("developmental actuator checkpoint is missing, linked, or oversized")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "format", "source_sha256", "contract_semantic_sha256", "update", "model_state",
        "ema_state", "optimizer_state", "model_state_sha256", "ema_state_sha256",
        "cpu_rng_state", "cuda_rng_state", "history", "runtime",
    }
    if (
        not isinstance(payload, dict) or set(payload) != required
        or payload["format"] != CHECKPOINT_FORMAT
        or payload["source_sha256"] != source_sha256()
        or payload["contract_semantic_sha256"] != contract["semantic_sha256"]
        or payload["update"] != expected_update or len(payload["history"]) != expected_update
    ):
        raise ValueError("developmental actuator checkpoint contract drifted")
    parent_contract, parent = _parent_authority(PROJECT_ROOT / contract["parent"]["path"])
    model = make_model(parent_contract, parent, DevelopmentalActuatorConfig(**contract["model"]))
    load_successor_state(model, payload["model_state"])
    if _state_sha256(successor_state(model)) != payload["model_state_sha256"]:
        raise ValueError("developmental actuator model state hash drifted")
    load_successor_state(model, payload["ema_state"])
    if _state_sha256(successor_state(model)) != payload["ema_state_sha256"]:
        raise ValueError("developmental actuator EMA state hash drifted")
    for index, row in enumerate(payload["history"], 1):
        if (
            set(row) != HISTORY_KEYS or row["step"] != index
            or any(not math.isfinite(float(value)) for key, value in row.items() if key != "step")
        ):
            raise ValueError("developmental actuator checkpoint history drifted")
    return payload


def _latest_update(output: Path, contract: dict[str, Any]) -> int:
    available: list[int] = []
    for path in output.glob("developmental_actuator_*.pt"):
        suffix = path.stem.removeprefix("developmental_actuator_")
        if suffix.isdigit():
            available.append(int(suffix))
    if not available:
        return 0
    latest = max(available)
    if latest % int(contract["segment_updates"]) or latest > int(contract["total_updates"]):
        raise ValueError("developmental actuator checkpoint schedule drifted")
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
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
        or torch.cuda.mem_get_info(0)[0] < int(contract["minimum_free_vram_bytes"])
    ):
        raise RuntimeError("developmental actuator training requires deterministic CUDA BF16 and 8 GiB free VRAM")
    require_disk_floor(output, floor_gb=100, planned_bytes=1024**3)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(contract["seed"]))
    torch.cuda.manual_seed_all(int(contract["seed"]))
    np.random.seed(int(contract["seed"]) & 0xFFFFFFFF)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    parent_contract, parent = _parent_authority(PROJECT_ROOT / contract["parent"]["path"])
    model = make_model(parent_contract, parent, DevelopmentalActuatorConfig(**contract["model"])).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(contract["optimizer"]["lr"]), weight_decay=float(contract["optimizer"]["weight_decay"]),
    )
    history: list[dict[str, float | int]] = []
    ema = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("parent.")
    }
    if start:
        previous = load_checkpoint(output / checkpoint_name(start), contract, start)
        load_successor_state(model, previous["model_state"])
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
    training = DevelopmentalTrainingConfig(**contract["training"])
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
        gradient = torch.nn.utils.clip_grad_norm_(trainable, float(contract["optimizer"]["gradient_clip"]))
        if not math.isfinite(float(gradient)) or float(gradient) <= 0.0:
            raise FloatingPointError("developmental actuator gradient became non-finite or zero")
        optimizer.step()
        with torch.no_grad():
            current = {
                name: value
                for name, value in model.state_dict().items()
                if not name.startswith("parent.")
            }
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
        })
    elapsed = time.perf_counter() - started
    model_state = successor_state(model)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": source_sha256(),
        "contract_semantic_sha256": contract["semantic_sha256"],
        "update": end,
        "model_state": model_state,
        "ema_state": {name: value.detach().cpu().clone() for name, value in ema.items()},
        "optimizer_state": optimizer.state_dict(),
        "model_state_sha256": _state_sha256(model_state),
        "ema_state_sha256": _state_sha256(ema),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "history": history,
        "runtime": {
            "segment_start": start, "segment_end": end,
            "elapsed_seconds": round(elapsed, 6),
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
        "passed": True, "complete": end == int(contract["total_updates"]),
        "update": end, "reused": False, "checkpoint": project_path(destination),
        "checkpoint_sha256": sha256_file(destination),
        "model_state_sha256": checked["model_state_sha256"],
        "ema_state_sha256": checked["ema_state_sha256"],
        "runtime": checked["runtime"],
    }
