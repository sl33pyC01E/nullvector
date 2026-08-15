from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
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
    CellularPhysiologyTransformerConfig,
    source_sha256,
)
from .dataset import NativeInterventionTeacher, PhysiologyBatchSampler
from .model import CellularPhysiologyTransformer, cellular_physiology_loss


SMOKE_FORMAT = "nullvector-creature-stage-neural-physiology-smoke-v1"
PRODUCTION_FORMAT = "nullvector-creature-stage-neural-physiology-production-v1"
SEED = 0x43454C4C50485931
MAX_CHECKPOINT_BYTES = 2 * 1024**3


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256(b"nullvector-cellular-physiology-transformer-state-v1\0")
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<u8").tobytes())
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    _atomic_bytes(path, buffer.getvalue())


def _config(payload: dict[str, Any]) -> CellularPhysiologyTransformerConfig:
    return CellularPhysiologyTransformerConfig(**payload)


def _forward(model: CellularPhysiologyTransformer, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    return model(
        batch["static"], batch["cell_state"], batch["summary_state"], batch["fluid_state"],
        batch["mask"], batch["adjacency"], batch["family"], batch["morphotype"],
        batch["intervention"], batch["phase"], batch["events"],
    )


def run_cpu_smoke(output: Path, *, teacher_root: Path = DEFAULT_TEACHER, steps: int = 4) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if type(steps) is not int or not 2 <= steps <= 12:
        raise ValueError("cellular physiology smoke step count drifted")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=768 * 1024**2)
    teacher = NativeInterventionTeacher(teacher_root)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    np.random.seed(SEED & 0xFFFFFFFF)
    config = CellularPhysiologyTransformerConfig(
        width=64, depth=2, heads=4, feedforward_multiplier=3,
        condition_width=128, fluid_width=64, fluid_depth=2, dropout=0.0,
    )
    model = CellularPhysiologyTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sampler = PhysiologyBatchSampler(teacher, batch_size=5, seed=SEED)
    # The fixed batch contains actual damage-era frames instead of mostly clean pre-event states.
    coordinates = [(family * 4, family + 4, 16 + family * 15) for family in range(5)]
    rows = [teacher.sample(*coordinate) for coordinate in coordinates]
    names = (
        "static", "mask", "adjacency", "cell_state", "summary_state", "fluid_state",
        "cell_target", "summary_target", "fluid_target", "events",
    )
    batch: dict[str, Tensor] = {
        name: torch.from_numpy(np.stack([row[name] for row in rows]).copy()) for name in names
    }
    for name in ("family", "morphotype", "intervention"):
        batch[name] = torch.tensor([int(row[name]) for row in rows], dtype=torch.long)
    batch["phase"] = torch.tensor([float(row["phase"]) for row in rows], dtype=torch.float32)
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predicted = _forward(model, batch)
        loss, pieces = cellular_physiology_loss(
            predicted, batch["cell_target"], batch["summary_target"], batch["fluid_target"],
            batch["mask"], batch["adjacency"],
        )
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient)):
            raise FloatingPointError("cellular physiology smoke became non-finite")
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
    model.eval()
    with torch.inference_mode():
        replay = _forward(model, batch)
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
        "production_config": CellularPhysiologyTransformerConfig().to_dict(),
        "smoke_parameters": model.parameter_count,
        "production_parameters": CellularPhysiologyTransformer().parameter_count,
        "steps": steps,
        "coordinates": [list(value) for value in coordinates],
        "history": history,
        "checkpoint": {
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256_file(checkpoint_path),
            "model_state_sha256": checkpoint["model_state_sha256"],
        },
        "gates": {
            "all_values_finite": all(
                math.isfinite(float(value)) for row in history for key, value in row.items() if key != "step"
            ),
            "all_five_families": sorted(batch["family"].tolist()) == list(range(5)),
            "damage_and_ablation_examples": all(int(row["intervention"]) >= 4 for row in rows),
            "fixed_batch_loss_improved": float(history[-1]["loss"]) < float(history[0]["loss"]),
            "gradient_nonzero": all(float(row["gradient_norm"]) > 0.0 for row in history),
            "outside_cells_exact_zero": float(replay[0][~batch["mask"]].abs().max()) == 0.0,
            "fluid_predictions_finite": bool(torch.isfinite(replay[2]).all()),
            "production_model_substantial": CellularPhysiologyTransformer().parameter_count >= 15_000_000,
        },
    }
    if not all(report["gates"].values()):
        raise ValueError(f"cellular physiology smoke failed: {report['gates']}")
    report["semantic_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    _atomic_bytes(output / "smoke_manifest.json", _canonical(report))
    return validate_cpu_smoke(output)


def validate_cpu_smoke(output: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest_path = output / "smoke_manifest.json"
    raw = manifest_path.read_bytes()
    if not 0 < len(raw) <= 2 * 1024 * 1024:
        raise ValueError("cellular physiology smoke manifest is oversized")
    report = json.loads(raw)
    required = {
        "format", "status", "source_sha256", "teacher", "config", "production_config",
        "smoke_parameters", "production_parameters", "steps", "coordinates", "history",
        "checkpoint", "gates", "semantic_sha256",
    }
    gate_keys = {
        "all_values_finite", "all_five_families", "damage_and_ablation_examples",
        "fixed_batch_loss_improved", "gradient_nonzero", "outside_cells_exact_zero",
        "fluid_predictions_finite", "production_model_substantial",
    }
    if (
        raw != _canonical(report) or set(report) != required
        or report["format"] != SMOKE_FORMAT or report["status"] != "passed"
        or report["source_sha256"] != source_sha256()
        or set(report["gates"]) != gate_keys or not all(report["gates"].values())
        or report["semantic_sha256"]
        != hashlib.sha256(_canonical({key: value for key, value in report.items() if key != "semantic_sha256"})).hexdigest()
    ):
        raise ValueError("cellular physiology smoke authority drifted")
    teacher = NativeInterventionTeacher(PROJECT_ROOT / report["teacher"]["path"])
    if teacher.semantic_sha256 != report["teacher"]["semantic_sha256"]:
        raise ValueError("cellular physiology smoke teacher drifted")
    checkpoint_path = output / report["checkpoint"]["path"]
    if (
        checkpoint_path.is_symlink() or not checkpoint_path.is_file()
        or checkpoint_path.stat().st_size != report["checkpoint"]["bytes"]
        or not 0 < checkpoint_path.stat().st_size <= MAX_CHECKPOINT_BYTES
        or _sha256_file(checkpoint_path) != report["checkpoint"]["sha256"]
    ):
        raise ValueError("cellular physiology smoke checkpoint bytes drifted")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if set(checkpoint) != {
        "format", "source_sha256", "teacher_semantic_sha256", "config", "steps",
        "model_state", "model_state_sha256", "history",
    }:
        raise ValueError("cellular physiology smoke checkpoint registry drifted")
    model = CellularPhysiologyTransformer(_config(checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if (
        checkpoint["format"] != CHECKPOINT_FORMAT
        or checkpoint["source_sha256"] != source_sha256()
        or checkpoint["teacher_semantic_sha256"] != teacher.semantic_sha256
        or checkpoint["model_state_sha256"] != _state_sha256(model.state_dict())
        or checkpoint["model_state_sha256"] != report["checkpoint"]["model_state_sha256"]
        or checkpoint["history"] != report["history"]
        or model.parameter_count != report["smoke_parameters"]
        or report["production_parameters"] < 15_000_000
    ):
        raise ValueError("cellular physiology smoke checkpoint semantics drifted")
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
    total_steps: int = 16_000,
    segment_steps: int = 800,
    batch_size: int = 10,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if (
        type(total_steps) is not int or type(segment_steps) is not int
        or not 4_000 <= total_steps <= 100_000 or not 200 <= segment_steps <= 2_000
        or total_steps % segment_steps or type(batch_size) is not int
        or not 5 <= batch_size <= 25 or batch_size % 5
    ):
        raise ValueError("cellular physiology production schedule drifted")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=10 * 1024**3)
    teacher = NativeInterventionTeacher(teacher_root)
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
        "model": CellularPhysiologyTransformerConfig().to_dict(),
        "total_steps": total_steps,
        "segment_steps": segment_steps,
        "batch_size": batch_size,
        "optimizer": {
            "name": "AdamW", "lr": 2e-4, "weight_decay": 1e-5,
            "warmup_steps": 1000, "gradient_clip": 1.0,
        },
        "ema_decay": 0.9995,
        "precision": "bf16-autocast-float32-loss",
        "minimum_free_vram_bytes": 16 * 1024**3,
        "family_balanced": True,
        "promotion_requires": [
            "prediction_fed_180_frame_rollouts", "all_nine_interventions",
            "causal_organ_ablation", "fluid_diffusion", "sealed_test",
        ],
    }
    contract["semantic_sha256"] = hashlib.sha256(_canonical(contract)).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "production_contract.json"
    if path.exists():
        if path.read_bytes() != _canonical(contract):
            raise ValueError("cellular physiology production contract changed during resume")
    else:
        _atomic_bytes(path, _canonical(contract))
    return contract


def assert_training_window(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    contract = json.loads((Path(output) / "production_contract.json").read_bytes())
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
        or torch.cuda.mem_get_info(0)[0] < contract["minimum_free_vram_bytes"]
    ):
        raise RuntimeError("cellular physiology training requires deterministic CUDA BF16 and 16 GiB free VRAM")
    return {"passed": True, "free_vram_bytes": int(torch.cuda.mem_get_info(0)[0])}
