from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Final
import uuid

from jsonschema import Draft202012Validator
import numpy as np
import torch

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_neural_motion.contract import MAX_CELLS as PARENT_MAX_CELLS
from ..creature_stage_neural_motion_rollout.contract import source_sha256 as rollout_source_sha256
from ..multifield_style_motion.hashing import artifact_record_from_bytes, deterministic_npz_bytes
from .compiler import _array_sha256
from .contract import (
    DEFAULT_CORPUS,
    DEFAULT_PARENT,
    DEFAULT_PRIOR,
    FRAME_COUNT,
    MAX_CELLS,
    PRIOR_SCHEMA,
    corpus_source_sha256,
)
from .dataset import DevelopmentalMotionTeacher, DevelopmentalSequenceSampler, project_path
from .training import (
    _parent_authority,
    atomic_bytes,
    canonical,
    parent_model_config,
    semantic,
    sha256_file,
)
from ..creature_stage_neural_motion.model import CellularMotionTransformer


FORMAT: Final[str] = "nullvector-creature-stage-developmental-parent-prior-v1"
ARRAY_FILE: Final[str] = "rollout1000_prior.npz"
MANIFEST_FILE: Final[str] = "prior_manifest.json"
MAX_ARCHIVE_BYTES: Final[int] = 256 * 1024**2


def prior_source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-developmental-parent-prior-source-v1\0")
    digest.update(corpus_source_sha256().encode("ascii") + b"\0")
    digest.update(rollout_source_sha256().encode("ascii") + b"\0")
    for path in (Path(__file__).resolve(), PRIOR_SCHEMA.resolve()):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _schema(payload: dict[str, Any]) -> None:
    errors = sorted(
        Draft202012Validator(json.loads(PRIOR_SCHEMA.read_text(encoding="utf-8"))).iter_errors(payload),
        key=lambda error: list(error.path),
    )
    if errors:
        raise ValueError(f"developmental parent prior schema drifted: {errors[0].message}")


def _parent_model(contract: dict[str, Any], checkpoint: dict[str, Any], device: torch.device):
    model = CellularMotionTransformer(parent_model_config(contract["model"]))
    model.load_state_dict(checkpoint["ema_state"], strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)


def build_parent_prior(
    output: Path = DEFAULT_PRIOR,
    *,
    corpus: Path = DEFAULT_CORPUS,
    parent: Path = DEFAULT_PARENT,
    device: str = "cuda",
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1024**3)
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("developmental parent prior requested unavailable CUDA")
    teacher = DevelopmentalMotionTeacher(corpus, replay=True)
    parent_contract, parent_checkpoint = _parent_authority(parent)
    model = _parent_model(parent_contract, parent_checkpoint, runtime_device)
    prior = np.zeros((teacher.specimen_count, FRAME_COUNT, MAX_CELLS, 4), dtype=np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for frame_index in range(FRAME_COUNT):
            rows = [teacher.sample(specimen, frame_index) for specimen in range(teacher.specimen_count)]
            frame = DevelopmentalSequenceSampler._stack(rows, runtime_device)
            controls = torch.zeros((teacher.specimen_count, 9), dtype=torch.float32, device=runtime_device)
            controls[:, 1] = -1.0
            motion = torch.full_like(frame["family"], 2)
            with torch.autocast(
                device_type=runtime_device.type, dtype=torch.bfloat16,
                enabled=runtime_device.type == "cuda",
            ):
                direct = model(
                    frame["static"][:, :PARENT_MAX_CELLS], frame["state"][:, :PARENT_MAX_CELLS],
                    frame["mask"][:, :PARENT_MAX_CELLS],
                    frame["adjacency"][:, :PARENT_MAX_CELLS, :PARENT_MAX_CELLS],
                    frame["family"], frame["morphotype"], motion, frame["phase"], controls,
                ).float()
            weights = frame["cell_node_weights"].float()
            direct_full = torch.zeros((teacher.specimen_count, MAX_CELLS, 4), dtype=torch.float32, device=runtime_device)
            direct_full[:, :PARENT_MAX_CELLS] = direct
            node_denominator = weights.sum(dim=1).clamp_min(1e-6)[:, :, None]
            node_prior = torch.bmm(weights.transpose(1, 2), direct_full) / node_denominator
            full = torch.bmm(weights, node_prior)
            full[:, :PARENT_MAX_CELLS] = full[:, :PARENT_MAX_CELLS] * .35 + direct * .65
            full = full * frame["mask"][:, :, None]
            prior[:, frame_index] = full.cpu().numpy()
    elapsed = time.perf_counter() - started
    if not np.isfinite(prior).all() or np.any(prior[~teacher.arrays["mask"][:, None].repeat(FRAME_COUNT, axis=1)] != 0.0):
        raise ValueError("developmental parent prior values drifted")
    archive = deterministic_npz_bytes({"parent_prior": prior})
    if not 0 < len(archive) <= MAX_ARCHIVE_BYTES:
        raise ValueError("developmental parent prior archive size drifted")
    artifact = artifact_record_from_bytes(ARRAY_FILE, archive)
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "status": "passed",
        "source_sha256": prior_source_sha256(),
        "teacher": {
            "path": project_path(teacher.root), "semantic_sha256": teacher.semantic_sha256,
            "corpus_semantic_sha256": teacher.manifest["semantic_sha256"],
        },
        "parent": {
            "path": project_path(Path(parent)), "sha256": sha256_file(Path(parent)),
            "update": 1_000, "ema_state_sha256": parent_checkpoint["ema_state_sha256"],
            "contract_semantic_sha256": parent_contract["semantic_sha256"],
        },
        "contract": {
            "specimens": teacher.specimen_count, "frames": FRAME_COUNT,
            "max_cells": MAX_CELLS, "features": 4,
            "projection": "direct-first384-plus-skeleton-weight-propagation-v1",
        },
        "array": {
            "artifact": artifact,
            "member": {"dtype": prior.dtype.str, "shape": list(prior.shape), "sha256": _array_sha256("parent_prior", prior)},
        },
        "runtime": {
            "device": str(runtime_device), "elapsed_seconds": round(elapsed, 6),
            "torch": torch.__version__, "cuda": torch.version.cuda or "none",
            "gpu": torch.cuda.get_device_name(runtime_device) if runtime_device.type == "cuda" else "cpu",
        },
    }
    manifest["semantic_sha256"] = semantic(manifest)
    _schema(manifest)
    staging = output.with_name(output.name + f".tmp-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    (staging / ARRAY_FILE).write_bytes(archive)
    (staging / MANIFEST_FILE).write_bytes(canonical(manifest))
    os.replace(staging, output)
    return validate_parent_prior(output)


def validate_parent_prior(output: Path = DEFAULT_PRIOR) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest_path = output / MANIFEST_FILE
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if raw != canonical(manifest):
        raise ValueError("developmental parent prior manifest is not canonical")
    _schema(manifest)
    if (
        manifest["format"] != FORMAT or manifest["status"] != "passed"
        or manifest["source_sha256"] != prior_source_sha256()
        or manifest["semantic_sha256"]
        != semantic({key: value for key, value in manifest.items() if key != "semantic_sha256"})
    ):
        raise ValueError("developmental parent prior authority drifted")
    teacher = DevelopmentalMotionTeacher(PROJECT_ROOT / manifest["teacher"]["path"], replay=False)
    parent_contract, parent = _parent_authority(PROJECT_ROOT / manifest["parent"]["path"])
    if (
        teacher.semantic_sha256 != manifest["teacher"]["semantic_sha256"]
        or parent["ema_state_sha256"] != manifest["parent"]["ema_state_sha256"]
        or parent_contract["semantic_sha256"] != manifest["parent"]["contract_semantic_sha256"]
    ):
        raise ValueError("developmental parent prior provenance drifted")
    archive_path = output / manifest["array"]["artifact"]["path"]
    artifact = manifest["array"]["artifact"]
    if (
        archive_path.is_symlink() or not archive_path.is_file()
        or archive_path.stat().st_size != artifact["bytes"]
        or sha256_file(archive_path) != artifact["sha256"]
    ):
        raise ValueError("developmental parent prior artifact drifted")
    with np.load(archive_path, allow_pickle=False) as archive:
        if archive.files != ["parent_prior"]:
            raise ValueError("developmental parent prior member registry drifted")
        prior = np.ascontiguousarray(archive["parent_prior"])
    record = {"dtype": prior.dtype.str, "shape": list(prior.shape), "sha256": _array_sha256("parent_prior", prior)}
    if record != manifest["array"]["member"] or prior.shape != (10, FRAME_COUNT, MAX_CELLS, 4):
        raise ValueError("developmental parent prior array drifted")
    mask = teacher.arrays["mask"][:, None, :, None]
    if not np.isfinite(prior).all() or np.any(prior[~np.broadcast_to(mask, prior.shape)] != 0.0):
        raise ValueError("developmental parent prior padded values drifted")
    return {
        "passed": True, "semantic_sha256": manifest["semantic_sha256"],
        "array_sha256": record["sha256"], "archive_sha256": artifact["sha256"],
        "specimens": 10, "frames": FRAME_COUNT,
    }
