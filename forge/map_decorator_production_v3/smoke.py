from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Final

import numpy as np
import torch

from ..config import PROJECT_ROOT
from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.dataset import TeacherSample, collate_teacher_samples
from ..map_decorator_production.teacher import build_production_sample
from ..map_decorator_production_v2.contract import ForegroundPatchConfig
from ..map_decorator_production_v2.patches import foreground_centered_crop
from ..map_decorator_production_v2.training import WarmStartEMA
from ..maps import MapConfig, generate_map
from ..safety import require_disk_floor
from .contract import LocatorLossConfig, LocatorModelConfig, LocatorTrainingConfig, V3_CONTRACT_SHA256
from .model import SparseLocatorDecoratorV3
from .training import make_optimizer_v3, train_batch_v3


SMOKE_FORMAT: Final[str] = "nullvector-map-decorator-v3-cpu-smoke/1.0.0"
SMOKE_SEED: Final[int] = 0x330033
SOURCE_ROOT: Final[Path] = PROJECT_ROOT / "forge" / "map_decorator_production_v3"


def source_manifest() -> dict[str, str]:
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(SOURCE_ROOT.glob("*.py"))
        if path.is_file()
    }


def source_sha256() -> str:
    manifest = source_manifest()
    if not manifest:
        raise FileNotFoundError("V3 source package is empty.")
    return json_sha256(manifest)


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"nullvector-v3-tensor-state-v1\0")
    for name in sorted(state):
        tensor = state[name].detach().contiguous().cpu()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Non-finite state tensor {name!r}.")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(memoryview(tensor.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=100.0, planned_bytes=len(encoded) + 1024 * 1024)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _teacher_sample() -> TeacherSample:
    data = generate_map(
        SMOKE_SEED,
        "garden",
        MapConfig(width=48, height=48, objective_count=3, spawn_count=5),
    )
    production = build_production_sample(
        data,
        feature_seed=SMOKE_SEED ^ 0xFEA7,
        replay_data=generate_map(data.seed, data.theme, data.config),
    )
    if any(not np.count_nonzero(production.targets[name]) for name in ("decal", "prop")):
        raise RuntimeError("Smoke authority lacks required object foreground.")
    return TeacherSample(
        features=production.features,
        targets=production.targets,
        legal_masks=production.legal_masks,
        hard_empty=production.hard_empty,
        global_conditions=production.global_conditions,
        theme_index=2,
        split=production.split,
        full_map_identity_sha256=production.full_map_identity_sha256,
        sample_identity_sha256=production.sample_identity_sha256,
        source_semantic_sha256=production.source_semantic_sha256,
        feature_tensor_sha256=production.feature_tensor_sha256,
        target_fields_sha256=production.target_fields_sha256,
        map_id=data.map_id,
        crop=None,
    )


def run_cpu_smoke(output: Path, *, steps: int = 4) -> dict[str, object]:
    output = Path(output).resolve()
    if isinstance(steps, bool) or not 1 <= steps <= 32:
        raise ValueError("Smoke steps must be in [1,32].")
    report_path = output / "smoke_report.json"
    if report_path.exists():
        return validate_cpu_smoke(report_path)
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=8 * 1024 * 1024)
    if torch.cuda.is_initialized():
        raise RuntimeError("V3 CPU smoke refuses a process that already initialized CUDA.")
    torch.manual_seed(SMOKE_SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    sample = _teacher_sample()
    patch_config = ForegroundPatchConfig(batch_size=2, decal_slots=1, prop_slots=1)
    crops = [
        foreground_centered_crop(
            sample,
            focus_head=head,
            epoch=0,
            step=0,
            slot=slot,
            seed=SMOKE_SEED,
            config=patch_config,
        )
        for slot, head in enumerate(("decal", "prop"))
    ]
    batch = collate_teacher_samples(crops)
    model_config = LocatorModelConfig(
        base_channels=8,
        condition_channels=16,
        locator_channels=8,
        locator_blocks=1,
        count_hidden_channels=8,
        count_prior=2.0,
    )
    training_config = LocatorTrainingConfig(
        learning_rate=5e-4, ema_decay=0.9, seed=SMOKE_SEED, full_mask_stride=1
    )
    loss_config = LocatorLossConfig(halo_radius=2)
    model = SparseLocatorDecoratorV3(model_config)
    optimizer = make_optimizer_v3(model, training_config)
    ema = WarmStartEMA(model, training_config.ema_decay)
    generator = torch.Generator().manual_seed(training_config.seed)
    initial_state = tensor_state_sha256(model.state_dict())
    history: list[dict[str, object]] = []
    for step in range(steps):
        result = train_batch_v3(
            model,
            optimizer,
            ema,
            batch,
            generator=generator,
            training_config=training_config,
            loss_config=loss_config,
        )
        history.append(
            {
                "step": step + 1,
                "total_loss": result["loss"]["total"],  # type: ignore[index]
                "gradient_norm": result["gradient_norm"],
                "decal_predicted_count": result["loss"]["decal_predicted_count"],  # type: ignore[index]
                "decal_target_count": result["loss"]["decal_target_count"],  # type: ignore[index]
                "prop_predicted_count": result["loss"]["prop_predicted_count"],  # type: ignore[index]
                "prop_target_count": result["loss"]["prop_target_count"],  # type: ignore[index]
                "full_mask_sample_count": result["full_mask_sample_count"],
            }
        )
    source_files = source_manifest()
    report: dict[str, object] = {
        "format": SMOKE_FORMAT,
        "status": "passed",
        "v3_contract_sha256": V3_CONTRACT_SHA256,
        "source_manifest": source_files,
        "source_sha256": json_sha256(source_files),
        "authority": {
            "map_id": sample.map_id,
            "sample_identity_sha256": sample.sample_identity_sha256,
            "full_map_identity_sha256": sample.full_map_identity_sha256,
            "source_semantic_sha256": sample.source_semantic_sha256,
            "crop_identity_sha256": [crop.sample_identity_sha256 for crop in crops],
        },
        "config": {
            "model": model_config.to_dict(),
            "training": training_config.to_dict(),
            "loss": loss_config.to_dict(),
            "steps": steps,
        },
        "runtime": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "device": "cpu",
            "cuda_initialized": torch.cuda.is_initialized(),
            "threads": torch.get_num_threads(),
        },
        "initial_model_sha256": initial_state,
        "final_model_sha256": tensor_state_sha256(model.state_dict()),
        "ema_sha256": tensor_state_sha256(ema.shadow),
        "ema_updates": ema.updates,
        "history": history,
        "gates": {
            "contract_exact": True,
            "authority_exact": True,
            "finite_training": all(np.isfinite(item["total_loss"]) for item in history),
            "model_changed": initial_state != tensor_state_sha256(model.state_dict()),
            "ema_updates_exact": ema.updates == steps,
            "cpu_only": not torch.cuda.is_initialized(),
            "not_a_quality_claim": True,
        },
    }
    if not all(report["gates"].values()):  # type: ignore[union-attr]
        raise RuntimeError("V3 CPU smoke failed a publication gate.")
    report["smoke_sha256"] = json_sha256(report)
    _atomic_json(report_path, report)
    return report


def _is_sha(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def validate_cpu_smoke(path: Path, *, exact_replay: bool = False) -> dict[str, object]:
    path = Path(path).resolve()
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "format", "status", "v3_contract_sha256", "source_manifest", "source_sha256",
        "authority", "config", "runtime", "initial_model_sha256", "final_model_sha256",
        "ema_sha256", "ema_updates", "history", "gates", "smoke_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("V3 smoke report keys violate the closed contract.")
    stored = value.pop("smoke_sha256", None)
    if stored != json_sha256(value):
        raise ValueError("V3 smoke report self-hash failed.")
    value["smoke_sha256"] = stored
    if value.get("format") != SMOKE_FORMAT or value.get("status") != "passed":
        raise ValueError("V3 smoke report format/status failed.")
    if value.get("v3_contract_sha256") != V3_CONTRACT_SHA256:
        raise ValueError("V3 smoke contract is stale.")
    if value.get("source_manifest") != source_manifest() or value.get("source_sha256") != source_sha256():
        raise ValueError("V3 smoke source provenance is stale.")
    for key in ("initial_model_sha256", "final_model_sha256", "ema_sha256"):
        if not _is_sha(value.get(key)):
            raise ValueError(f"V3 smoke {key} is not a SHA-256 identity.")
    config = value.get("config")
    if not isinstance(config, dict) or set(config) != {"model", "training", "loss", "steps"}:
        raise ValueError("V3 smoke configuration is malformed.")
    steps = config.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 32:
        raise ValueError("V3 smoke step count is invalid.")
    expected_config = {
        "model": LocatorModelConfig(base_channels=8, condition_channels=16, locator_channels=8, locator_blocks=1, count_hidden_channels=8, count_prior=2.0).to_dict(),
        "training": LocatorTrainingConfig(learning_rate=5e-4, ema_decay=0.9, seed=SMOKE_SEED, full_mask_stride=1).to_dict(),
        "loss": LocatorLossConfig(halo_radius=2).to_dict(),
        "steps": steps,
    }
    if config != expected_config:
        raise ValueError("V3 smoke configuration drifted from its exact runner.")
    sample = _teacher_sample()
    patch_config = ForegroundPatchConfig(batch_size=2, decal_slots=1, prop_slots=1)
    crops = [
        foreground_centered_crop(sample, focus_head=head, epoch=0, step=0, slot=slot, seed=SMOKE_SEED, config=patch_config)
        for slot, head in enumerate(("decal", "prop"))
    ]
    expected_authority = {
        "map_id": sample.map_id,
        "sample_identity_sha256": sample.sample_identity_sha256,
        "full_map_identity_sha256": sample.full_map_identity_sha256,
        "source_semantic_sha256": sample.source_semantic_sha256,
        "crop_identity_sha256": [crop.sample_identity_sha256 for crop in crops],
    }
    if value.get("authority") != expected_authority:
        raise ValueError("V3 smoke authority failed deterministic reconstruction.")
    runtime = value.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {"python", "torch", "device", "cuda_initialized", "threads"}:
        raise ValueError("V3 smoke runtime provenance is malformed.")
    if runtime["device"] != "cpu" or runtime["cuda_initialized"] is not False or runtime["threads"] != 1:
        raise ValueError("V3 smoke runtime violated the CPU-only contract.")
    history = value.get("history")
    history_keys = {
        "step", "total_loss", "gradient_norm", "decal_predicted_count", "decal_target_count",
        "prop_predicted_count", "prop_target_count", "full_mask_sample_count",
    }
    if not isinstance(history, list) or len(history) != steps:
        raise ValueError("V3 smoke history length is invalid.")
    for index, item in enumerate(history, start=1):
        if not isinstance(item, dict) or set(item) != history_keys or item["step"] != index or item["full_mask_sample_count"] != 2:
            raise ValueError("V3 smoke history record is malformed.")
        numeric = [item[key] for key in history_keys - {"step", "full_mask_sample_count"}]
        if any(isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(number) for number in numeric):
            raise ValueError("V3 smoke history contains invalid numeric evidence.")
        if item["decal_target_count"] <= 0 or item["prop_target_count"] <= 0:
            raise ValueError("V3 smoke history lost required object foreground.")
    if value.get("ema_updates") != steps or value.get("initial_model_sha256") == value.get("final_model_sha256"):
        raise ValueError("V3 smoke optimizer/EMA progression is inconsistent.")
    gates = value.get("gates")
    if not isinstance(gates, dict) or set(gates) != {
        "contract_exact", "authority_exact", "finite_training", "model_changed",
        "ema_updates_exact", "cpu_only", "not_a_quality_claim",
    } or not all(gates.values()):
        raise ValueError("V3 smoke gates are incomplete or failed.")
    expected_gates = {
        "contract_exact": value["v3_contract_sha256"] == V3_CONTRACT_SHA256,
        "authority_exact": value["authority"] == expected_authority,
        "finite_training": all(np.isfinite(item["total_loss"]) for item in history),
        "model_changed": value["initial_model_sha256"] != value["final_model_sha256"],
        "ema_updates_exact": value["ema_updates"] == steps,
        "cpu_only": runtime["cuda_initialized"] is False and runtime["device"] == "cpu",
        "not_a_quality_claim": True,
    }
    if gates != expected_gates:
        raise ValueError("V3 smoke gates do not match recomputed evidence.")
    if exact_replay:
        with tempfile.TemporaryDirectory(prefix="nullvector-v3-smoke-replay-") as temporary:
            replay = run_cpu_smoke(Path(temporary), steps=steps)
        if replay != value:
            raise ValueError("V3 smoke exact replay diverged.")
    return value
