from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from .contract import (
    CHECKPOINT_FORMAT,
    DEFAULT_OUTPUT,
    DEFAULT_TEACHER,
    FORMAT,
    CellularMotionTransformerConfig,
    source_sha256,
)
from .dataset import MotionBatchSampler, NativeMotionTeacher
from .model import CellularMotionTransformer, cellular_motion_loss


SMOKE_FORMAT = "nullvector-creature-stage-neural-motion-smoke-v1"
PRODUCTION_FORMAT = "nullvector-creature-stage-neural-motion-production-v1"
SEED = 0x43454C4C4D4F5431
MAX_CHECKPOINT_BYTES = 2 * 1024**3


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256(b"nullvector-cellular-motion-transformer-state-v1\0")
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<u8").tobytes())
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    _atomic_bytes(path, buffer.getvalue())


def _config(payload: dict[str, Any]) -> CellularMotionTransformerConfig:
    return CellularMotionTransformerConfig(**payload)


def run_cpu_smoke(
    output: Path,
    *,
    teacher_root: Path = DEFAULT_TEACHER,
    steps: int = 4,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if type(steps) is not int or not 2 <= steps <= 12:
        raise ValueError("cellular neural motion smoke step count drifted")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=512 * 1024**2)
    teacher = NativeMotionTeacher(teacher_root)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    np.random.seed(SEED & 0xFFFFFFFF)
    config = CellularMotionTransformerConfig(
        width=64,
        depth=2,
        heads=4,
        feedforward_multiplier=3,
        condition_width=128,
        dropout=0.0,
    )
    model = CellularMotionTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    sampler = MotionBatchSampler(teacher, batch_size=5, seed=SEED)
    batch = sampler.batch(0)
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predicted = model(
            batch["static"], batch["state"], batch["mask"], batch["adjacency"],
            batch["family"], batch["morphotype"], batch["motion"], batch["phase"],
            batch["controls"],
        )
        loss, pieces = cellular_motion_loss(
            predicted, batch["target"], batch["state"], batch["mask"], batch["adjacency"]
        )
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient)):
            raise FloatingPointError("cellular neural motion smoke became non-finite")
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                **{name: round(float(value), 9) for name, value in pieces.items()},
                "gradient_norm": round(float(gradient), 9),
            }
        )
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": source_sha256(),
        "teacher_semantic_sha256": teacher.semantic_sha256,
        "config": config.to_dict(),
        "steps": steps,
        "model_state": state,
        "model_state_sha256": _state_sha256(state),
        "history": history,
    }
    output.mkdir(parents=True)
    checkpoint_path = output / "smoke_checkpoint.pt"
    _atomic_torch(checkpoint_path, checkpoint)
    with torch.no_grad():
        replay = model(
            batch["static"], batch["state"], batch["mask"], batch["adjacency"],
            batch["family"], batch["morphotype"], batch["motion"], batch["phase"],
            batch["controls"],
        )
    report: dict[str, Any] = {
        "format": SMOKE_FORMAT,
        "status": "passed",
        "source_sha256": source_sha256(),
        "teacher": {
            "path": teacher.root.relative_to(PROJECT_ROOT).as_posix(),
            "semantic_sha256": teacher.semantic_sha256,
            "manifest_sha256": teacher.validation["manifest_sha256"],
            "binary_sha256": teacher.validation["binary_sha256"],
        },
        "config": config.to_dict(),
        "production_config": CellularMotionTransformerConfig().to_dict(),
        "smoke_parameters": model.parameter_count,
        "production_parameters": CellularMotionTransformer().parameter_count,
        "steps": steps,
        "history": history,
        "checkpoint": {
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256_file(checkpoint_path),
            "model_state_sha256": checkpoint["model_state_sha256"],
        },
        "gates": {
            "all_values_finite": all(
                math.isfinite(float(value))
                for row in history for key, value in row.items() if key != "step"
            ),
            "all_five_families": sorted(batch["family"].tolist()) == [0, 1, 2, 3, 4],
            "gradient_nonzero": all(float(row["gradient_norm"]) > 0.0 for row in history),
            "fixed_batch_loss_improved": float(history[-1]["loss"]) < float(history[0]["loss"]),
            "outside_cells_exact_zero": float(replay[~batch["mask"]].abs().max()) == 0.0,
            "production_model_substantial": CellularMotionTransformer().parameter_count >= 20_000_000,
        },
    }
    if not all(report["gates"].values()):
        raise ValueError(f"cellular neural motion smoke failed: {report['gates']}")
    report["semantic_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    _atomic_bytes(output / "smoke_manifest.json", _canonical(report))
    return validate_cpu_smoke(output)


def validate_cpu_smoke(output: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest_path = output / "smoke_manifest.json"
    raw = manifest_path.read_bytes()
    if not 0 < len(raw) <= 2 * 1024 * 1024:
        raise ValueError("cellular neural motion smoke manifest is oversized")
    report = json.loads(raw)
    required = {
        "format", "status", "source_sha256", "teacher", "config", "production_config",
        "smoke_parameters", "production_parameters", "steps", "history", "checkpoint",
        "gates", "semantic_sha256",
    }
    if raw != _canonical(report) or set(report) != required:
        raise ValueError("cellular neural motion smoke manifest structure drifted")
    if (
        report["format"] != SMOKE_FORMAT
        or report["status"] != "passed"
        or report["source_sha256"] != source_sha256()
        or report["semantic_sha256"]
        != hashlib.sha256(_canonical({k: v for k, v in report.items() if k != "semantic_sha256"})).hexdigest()
        or set(report["gates"]) != {
            "all_values_finite", "all_five_families", "gradient_nonzero",
            "fixed_batch_loss_improved", "outside_cells_exact_zero",
            "production_model_substantial",
        }
        or not all(report["gates"].values())
    ):
        raise ValueError("cellular neural motion smoke authority drifted")
    teacher_path = PROJECT_ROOT / report["teacher"]["path"]
    teacher = NativeMotionTeacher(teacher_path)
    if teacher.semantic_sha256 != report["teacher"]["semantic_sha256"]:
        raise ValueError("cellular neural motion smoke teacher drifted")
    checkpoint_path = output / report["checkpoint"]["path"]
    if (
        checkpoint_path.is_symlink()
        or not checkpoint_path.is_file()
        or not 0 < checkpoint_path.stat().st_size <= MAX_CHECKPOINT_BYTES
        or checkpoint_path.stat().st_size != report["checkpoint"]["bytes"]
        or _sha256_file(checkpoint_path) != report["checkpoint"]["sha256"]
    ):
        raise ValueError("cellular neural motion smoke checkpoint bytes drifted")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if set(checkpoint) != {
        "format", "source_sha256", "teacher_semantic_sha256", "config", "steps",
        "model_state", "model_state_sha256", "history",
    }:
        raise ValueError("cellular neural motion smoke checkpoint registry drifted")
    model = CellularMotionTransformer(_config(checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if (
        checkpoint["format"] != CHECKPOINT_FORMAT
        or checkpoint["source_sha256"] != source_sha256()
        or checkpoint["teacher_semantic_sha256"] != teacher.semantic_sha256
        or checkpoint["model_state_sha256"] != _state_sha256(model.state_dict())
        or checkpoint["model_state_sha256"] != report["checkpoint"]["model_state_sha256"]
        or checkpoint["history"] != report["history"]
        or model.parameter_count != report["smoke_parameters"]
        or report["production_parameters"] < 20_000_000
    ):
        raise ValueError("cellular neural motion smoke checkpoint semantics drifted")
    return {
        "passed": True,
        "steps": report["steps"],
        "smoke_parameters": report["smoke_parameters"],
        "production_parameters": report["production_parameters"],
        "model_state_sha256": checkpoint["model_state_sha256"],
        "semantic_sha256": report["semantic_sha256"],
    }


def prepare_production(
    output: Path = DEFAULT_OUTPUT,
    *,
    teacher_root: Path = DEFAULT_TEACHER,
    total_steps: int = 20_000,
    segment_steps: int = 1_000,
    batch_size: int = 10,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if (
        type(total_steps) is not int
        or type(segment_steps) is not int
        or not 1_000 <= total_steps <= 100_000
        or not 250 <= segment_steps <= 2_000
        or total_steps % segment_steps
        or type(batch_size) is not int
        or not 5 <= batch_size <= 30
        or batch_size % 5
    ):
        raise ValueError("cellular neural motion production schedule drifted")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=12 * 1024**3)
    teacher = NativeMotionTeacher(teacher_root)
    contract: dict[str, Any] = {
        "format": PRODUCTION_FORMAT,
        "source_sha256": source_sha256(),
        "seed": SEED,
        "teacher": {
            "path": teacher.root.relative_to(PROJECT_ROOT).as_posix(),
            "semantic_sha256": teacher.semantic_sha256,
            "manifest_sha256": teacher.validation["manifest_sha256"],
            "binary_sha256": teacher.validation["binary_sha256"],
            "split": {"train": [0, 1], "validation": [2], "test": [3]},
        },
        "model": CellularMotionTransformerConfig().to_dict(),
        "total_steps": total_steps,
        "segment_steps": segment_steps,
        "batch_size": batch_size,
        "optimizer": {
            "name": "AdamW", "lr": 2e-4, "weight_decay": 1e-5,
            "warmup_steps": min(1000, total_steps // 5), "gradient_clip": 1.0,
        },
        "ema_decay": 0.9995,
        "precision": "bf16-autocast-float32-loss",
        "minimum_free_vram_bytes": 16 * 1024**3,
        "family_balanced": True,
    }
    contract["semantic_sha256"] = hashlib.sha256(_canonical(contract)).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "production_contract.json"
    if path.exists():
        if json.loads(path.read_bytes()) != contract or path.read_bytes() != _canonical(contract):
            raise ValueError("cellular neural motion production contract changed during resume")
    else:
        _atomic_bytes(path, _canonical(contract))
    return contract


def checkpoint_name(step: int) -> str:
    return f"cell_motion_{step:07d}.pt"


def _load_checkpoint(path: Path, contract: dict[str, Any], expected_step: int) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("cellular neural motion checkpoint is missing or oversized")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "format", "source_sha256", "contract_semantic_sha256", "step", "model_state",
        "ema_state", "optimizer_state", "model_state_sha256", "ema_state_sha256",
        "cpu_rng_state", "cuda_rng_state", "history", "runtime",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["format"] != CHECKPOINT_FORMAT
        or payload["source_sha256"] != source_sha256()
        or payload["contract_semantic_sha256"] != contract["semantic_sha256"]
        or payload["step"] != expected_step
        or len(payload["history"]) != expected_step
    ):
        raise ValueError("cellular neural motion checkpoint contract drifted")
    model = CellularMotionTransformer(_config(contract["model"]))
    model.load_state_dict(payload["model_state"], strict=True)
    if _state_sha256(model.state_dict()) != payload["model_state_sha256"]:
        raise ValueError("cellular neural motion model state hash drifted")
    model.load_state_dict(payload["ema_state"], strict=True)
    if _state_sha256(model.state_dict()) != payload["ema_state_sha256"]:
        raise ValueError("cellular neural motion EMA state hash drifted")
    if not isinstance(payload["cpu_rng_state"], Tensor) or not isinstance(payload["cuda_rng_state"], Tensor):
        raise ValueError("cellular neural motion RNG state drifted")
    history_keys = {
        "step", "loss", "position", "velocity", "graph", "acceleration", "outside",
        "gradient_norm", "lr",
    }
    for index, row in enumerate(payload["history"], 1):
        if set(row) != history_keys or row.get("step") != index or any(
            not math.isfinite(float(value)) for key, value in row.items() if key != "step"
        ):
            raise ValueError("cellular neural motion checkpoint history drifted")
    return payload


def train_segment(output: Path = DEFAULT_OUTPUT, *, end_step: int) -> dict[str, Any]:
    output = Path(output).resolve()
    contract_path = output / "production_contract.json"
    raw_contract = contract_path.read_bytes()
    contract = json.loads(raw_contract)
    if (
        raw_contract != _canonical(contract)
        or contract["source_sha256"] != source_sha256()
        or contract["semantic_sha256"]
        != hashlib.sha256(_canonical({k: v for k, v in contract.items() if k != "semantic_sha256"})).hexdigest()
    ):
        raise ValueError("cellular neural motion training authority drifted")
    segment_steps = int(contract["segment_steps"])
    if type(end_step) is not int or end_step % segment_steps or not segment_steps <= end_step <= contract["total_steps"]:
        raise ValueError("cellular neural motion segment endpoint drifted")
    destination = output / checkpoint_name(end_step)
    if destination.exists():
        payload = _load_checkpoint(destination, contract, end_step)
        return {"passed": True, "step": end_step, "model_state_sha256": payload["model_state_sha256"], "ema_state_sha256": payload["ema_state_sha256"]}
    previous_step = end_step - segment_steps
    previous_path = output / checkpoint_name(previous_step) if previous_step else None
    if previous_path is not None and not previous_path.exists():
        raise FileNotFoundError("previous cellular neural motion segment is missing")
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or torch.cuda.mem_get_info(0)[0] < contract["minimum_free_vram_bytes"]
    ):
        raise RuntimeError(
            "cellular neural motion training requires deterministic CUDA BF16 and 16 GiB free VRAM"
        )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(contract["seed"])
    torch.cuda.manual_seed_all(contract["seed"])
    np.random.seed(contract["seed"] & 0xFFFFFFFF)
    device = torch.device("cuda", 0)
    torch.cuda.reset_peak_memory_stats(device)
    model = CellularMotionTransformer(_config(contract["model"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=contract["optimizer"]["lr"],
        weight_decay=contract["optimizer"]["weight_decay"],
    )
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, float | int]] = []
    start_step = 0
    if previous_path is not None:
        previous = _load_checkpoint(previous_path, contract, previous_step)
        start_step = previous_step
        model.load_state_dict(previous["model_state"], strict=True)
        optimizer.load_state_dict(previous["optimizer_state"])
        ema = {name: value.to(device) for name, value in previous["ema_state"].items()}
        torch.set_rng_state(previous["cpu_rng_state"])
        torch.cuda.set_rng_state(previous["cuda_rng_state"], device)
        history = list(previous["history"])
    teacher = NativeMotionTeacher(PROJECT_ROOT / contract["teacher"]["path"])
    if teacher.semantic_sha256 != contract["teacher"]["semantic_sha256"]:
        raise ValueError("cellular neural motion teacher changed during training")
    sampler = MotionBatchSampler(teacher, batch_size=contract["batch_size"], seed=contract["seed"])
    started = time.perf_counter()
    model.train()
    for step in range(start_step, end_step):
        batch = sampler.batch(step, device)
        lr = contract["optimizer"]["lr"] * min(
            1.0, (step + 1) / max(1, contract["optimizer"]["warmup_steps"])
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predicted = model(
                batch["static"], batch["state"], batch["mask"], batch["adjacency"],
                batch["family"], batch["morphotype"], batch["motion"], batch["phase"],
                batch["controls"],
            )
        loss, pieces = cellular_motion_loss(
            predicted.float(), batch["target"].float(), batch["state"].float(),
            batch["mask"], batch["adjacency"],
        )
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), contract["optimizer"]["gradient_clip"]
        )
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient)):
            raise FloatingPointError("cellular neural motion training became non-finite")
        optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if value.dtype.is_floating_point:
                    ema[name].lerp_(value.detach(), 1.0 - contract["ema_decay"])
                else:
                    ema[name].copy_(value)
        history.append(
            {
                "step": step + 1,
                **{name: round(float(value), 9) for name, value in pieces.items()},
                "gradient_norm": round(float(gradient), 9),
                "lr": round(float(lr), 12),
            }
        )
    elapsed = time.perf_counter() - started
    model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    ema_state = {name: value.detach().cpu().clone() for name, value in ema.items()}
    payload = {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": source_sha256(),
        "contract_semantic_sha256": contract["semantic_sha256"],
        "step": end_step,
        "model_state": model_state,
        "ema_state": ema_state,
        "optimizer_state": optimizer.state_dict(),
        "model_state_sha256": _state_sha256(model_state),
        "ema_state_sha256": _state_sha256(ema_state),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device),
        "history": history,
        "runtime": {
            "segment_seconds": round(elapsed, 6),
            "updates_per_second": round(segment_steps / elapsed, 6),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "device": torch.cuda.get_device_name(device),
            "torch": str(torch.__version__),
        },
    }
    _atomic_torch(destination, payload)
    checked = _load_checkpoint(destination, contract, end_step)
    return {"passed": True, "step": end_step, "model_state_sha256": checked["model_state_sha256"], "ema_state_sha256": checked["ema_state_sha256"]}
