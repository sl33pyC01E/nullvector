from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np
import torch

from ..config import PROJECT_ROOT
from ..multifield_style_motion.hashing import canonical_json_bytes
from ..safety import require_disk_floor
from .contract import NeuralCellMotionConfig, model_source_sha256
from .dataset import _read_canonical_json, sha256_file
from .model import NeuralCellMotionUNet
from .production import CONTRACT_NAME, _config_from_dict, _load_checkpoint, _semantic, checkpoint_name
from .training import _atomic_bytes


EXPORT_FORMAT = "nullvector-neural-cell-motion-onnx-bundle-v1"
MANIFEST_NAME = "neural_cell_motion_onnx.json"
MODEL_NAME = "neural_cell_motion.onnx"
MAX_ONNX_BYTES = 2 * 1024**3
EXPORT_SOURCE_FILES = (
    "forge/neural_cell_motion/export.py",
    "forge/neural_cell_motion/model.py",
    "forge/neural_cell_motion/contract.py",
)


def export_source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-neural-cell-motion-onnx-export-source-v1\0")
    for relative in EXPORT_SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file(): raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _export_onnx(model: NeuralCellMotionUNet, destination: Path) -> None:
    model = model.cpu().eval(); batch = 1
    inputs = (
        torch.zeros(batch, 60, 48, 48, dtype=torch.float32),
        torch.zeros(batch, 4, 48, 48, dtype=torch.float32),
        torch.zeros(batch, dtype=torch.int64),
        torch.zeros(batch, dtype=torch.int64),
        torch.zeros(batch, dtype=torch.int64),
        torch.zeros(batch, dtype=torch.float32),
    )
    names = ("static", "previous", "family", "motion", "facing", "phase", "state")
    torch.onnx.export(
        model, inputs, destination, input_names=list(names[:-1]), output_names=[names[-1]],
        opset_version=18, do_constant_folding=True, dynamo=False,
        dynamic_axes={name: {0: "batch"} for name in names},
    )


def _probe(batch: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed); static = np.zeros((batch, 60, 48, 48), dtype=np.float32); previous = np.zeros((batch, 4, 48, 48), dtype=np.float32)
    for index in range(batch):
        y0, y1 = 7 + index, 40 - index; x0, x1 = 8 + index * 2, 39 - index
        mask = np.zeros((48, 48), dtype=np.float32); mask[y0:y1, x0:x1] = 1; static[index, 0] = mask
        yy, xx = np.mgrid[:48, :48]; static[index, 1] = ((xx - 23.5) / 24 * mask).astype(np.float32); static[index, 2] = ((yy - 23.5) / 24 * mask).astype(np.float32)
        static[index, 3:10] = rng.normal(0, .15, (7, 48, 48)).astype(np.float32) * mask
        previous[index, :2] = rng.uniform(-.3, .3, (2, 48, 48)).astype(np.float32) * mask
        previous[index, 2:] = rng.uniform(0, 1, (2, 48, 48)).astype(np.float32) * mask
    return {
        "static": static, "previous": previous,
        "family": np.arange(batch, dtype=np.int64) % 5,
        "motion": (np.arange(batch, dtype=np.int64) * 7 + 2) % 13,
        "facing": (np.arange(batch, dtype=np.int64) * 3 + 1) % 8,
        "phase": np.linspace(.125, .875, batch, dtype=np.float32),
    }


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _parity(model: NeuralCellMotionUNet, onnx_path: Path) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(onnx_path, load_external_data=False), full_check=True)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]); model = model.cpu().eval(); cases: list[dict[str, Any]] = []
    for batch, seed in ((1, 0x4E434D01), (2, 0x4E434D02), (5, 0x4E434D05)):
        feed = _probe(batch, seed)
        with torch.inference_mode():
            reference = model(*(torch.from_numpy(feed[name]) for name in ("static", "previous", "family", "motion", "facing", "phase"))).numpy()
        actual = session.run(["state"], feed)[0].astype(np.float32, copy=False); error = np.abs(actual - reference); outside = np.abs(actual * (1 - feed["static"][:, :1]))
        cases.append({
            "batch": batch, "seed": seed, "shape": list(actual.shape),
            "max_abs_error": round(float(error.max()), 10), "mean_abs_error": round(float(error.mean()), 10),
            "outside_support_max": round(float(outside.max()), 10),
            "pytorch_output_sha256": _array_sha256(reference), "onnx_output_sha256": _array_sha256(actual),
        })
    maximum = max(case["max_abs_error"] for case in cases)
    if maximum > 2e-5 or any(case["outside_support_max"] != 0 for case in cases):
        raise ValueError("Neural motion ONNX parity exceeded its exact runtime boundary.")
    return {"provider": "CPUExecutionProvider", "case_count": len(cases), "max_abs_error": maximum, "tolerance": 2e-5, "outside_support_exact_zero": True, "cases": cases}


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_ONNX_BYTES: raise ValueError("Neural motion ONNX artifact missing or oversized.")
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def export_checkpoint(production: Path, *, step: int, destination: Path) -> dict[str, Any]:
    production = Path(production).resolve(); destination = Path(destination).resolve(); contract = _read_canonical_json(production / CONTRACT_NAME)
    if contract.get("source_sha256") != model_source_sha256() or contract.get("semantic_sha256") != _semantic({key: value for key, value in contract.items() if key != "semantic_sha256"}): raise ValueError("Neural motion export training authority drifted.")
    checkpoint_path = production / checkpoint_name(step); checkpoint = _load_checkpoint(checkpoint_path, contract, expected_step=step)
    if destination.exists(): raise FileExistsError(destination)
    require_disk_floor(destination.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"; staging.mkdir(parents=True, exist_ok=False)
    try:
        model = NeuralCellMotionUNet(_config_from_dict(contract["model"])); model.load_state_dict(checkpoint["ema_state"], strict=True); onnx_path = staging / MODEL_NAME
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True); _export_onnx(model, onnx_path); parity = _parity(model, onnx_path)
        manifest: dict[str, Any] = {
            "format": EXPORT_FORMAT, "status": "ready", "source_sha256": export_source_sha256(), "model_source_sha256": model_source_sha256(),
            "contract_semantic_sha256": contract["semantic_sha256"], "corpus_semantic_sha256": contract["corpus"]["semantic_sha256"],
            "checkpoint": {"path": checkpoint_path.name, "bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path), "step": step, "model_state_sha256": checkpoint["model_state_sha256"], "ema_state_sha256": checkpoint["ema_state_sha256"]},
            "model": {"config": contract["model"], "parameters": model.parameter_count, "opset": 18, "precision": "float32", "dynamic_batch": True},
            "io": {"inputs": {"static": ["batch", 60, 48, 48], "previous": ["batch", 4, 48, 48], "family": ["batch"], "motion": ["batch"], "facing": ["batch"], "phase": ["batch"]}, "output": {"state": ["batch", 4, 48, 48]}},
            "artifact": _artifact(onnx_path), "parity": parity,
        }
        manifest["semantic_sha256"] = _semantic(manifest); _atomic_bytes(staging / MANIFEST_NAME, canonical_json_bytes(manifest)); _validate_bundle(staging)
        os.replace(staging, destination)
    finally:
        if staging.exists(): shutil.rmtree(staging)
    return validate_export_bundle(destination, production=production, replay=True)


def _validate_bundle(bundle: Path) -> dict[str, Any]:
    import onnx

    manifest = _read_canonical_json(bundle / MANIFEST_NAME, maximum_bytes=2 * 1024 * 1024)
    required = {"format", "status", "source_sha256", "model_source_sha256", "contract_semantic_sha256", "corpus_semantic_sha256", "checkpoint", "model", "io", "artifact", "parity", "semantic_sha256"}
    if set(manifest) != required or manifest["format"] != EXPORT_FORMAT or manifest["status"] != "ready" or manifest["source_sha256"] != export_source_sha256() or manifest["model_source_sha256"] != model_source_sha256() or manifest["semantic_sha256"] != _semantic({key: value for key, value in manifest.items() if key != "semantic_sha256"}): raise ValueError("Neural motion ONNX manifest provenance drifted.")
    artifact = manifest["artifact"]; path = bundle / MODEL_NAME
    if not isinstance(artifact, dict) or set(artifact) != {"path", "bytes", "sha256"} or artifact != _artifact(path) or artifact["path"] != MODEL_NAME: raise ValueError("Neural motion ONNX artifact binding drifted.")
    graph = onnx.load(path, load_external_data=False); onnx.checker.check_model(graph, full_check=True)
    if [value.name for value in graph.graph.input] != ["static", "previous", "family", "motion", "facing", "phase"] or [value.name for value in graph.graph.output] != ["state"]: raise ValueError("Neural motion ONNX graph IO registry drifted.")
    model_record = manifest["model"]
    if not isinstance(model_record, dict) or set(model_record) != {"config", "parameters", "opset", "precision", "dynamic_batch"} or model_record["opset"] != 18 or model_record["precision"] != "float32" or model_record["dynamic_batch"] is not True: raise ValueError("Neural motion ONNX model contract drifted.")
    config = _config_from_dict(model_record["config"])
    if type(model_record["parameters"]) is not int or model_record["parameters"] != NeuralCellMotionUNet(config).parameter_count: raise ValueError("Neural motion ONNX parameter census drifted.")
    checkpoint = manifest["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"path", "bytes", "sha256", "step", "model_state_sha256", "ema_state_sha256"} or type(checkpoint["step"]) is not int or checkpoint["step"] <= 0 or type(checkpoint["bytes"]) is not int or checkpoint["bytes"] <= 0 or any(not isinstance(checkpoint[key], str) or len(checkpoint[key]) != 64 for key in ("sha256", "model_state_sha256", "ema_state_sha256")): raise ValueError("Neural motion ONNX checkpoint record drifted.")
    expected_io = {"inputs": {"static": ["batch", 60, 48, 48], "previous": ["batch", 4, 48, 48], "family": ["batch"], "motion": ["batch"], "facing": ["batch"], "phase": ["batch"]}, "output": {"state": ["batch", 4, 48, 48]}}
    if manifest["io"] != expected_io: raise ValueError("Neural motion ONNX IO contract drifted.")
    parity = manifest["parity"]
    if not isinstance(parity, dict) or set(parity) != {"provider", "case_count", "max_abs_error", "tolerance", "outside_support_exact_zero", "cases"} or parity["provider"] != "CPUExecutionProvider" or parity["case_count"] != 3 or parity["tolerance"] != 2e-5 or parity["outside_support_exact_zero"] is not True or not isinstance(parity["cases"], list) or parity["max_abs_error"] > parity["tolerance"] or [case.get("batch") for case in parity["cases"]] != [1, 2, 5]: raise ValueError("Neural motion ONNX parity evidence drifted.")
    case_keys = {"batch", "seed", "shape", "max_abs_error", "mean_abs_error", "outside_support_max", "pytorch_output_sha256", "onnx_output_sha256"}
    for case in parity["cases"]:
        if not isinstance(case, dict) or set(case) != case_keys or case["shape"] != [case["batch"], 4, 48, 48] or case["outside_support_max"] != 0 or any(not isinstance(case[key], (int, float)) or isinstance(case[key], bool) or not math.isfinite(float(case[key])) or case[key] < 0 for key in ("max_abs_error", "mean_abs_error", "outside_support_max")) or case["max_abs_error"] > parity["tolerance"] or any(not isinstance(case[key], str) or len(case[key]) != 64 for key in ("pytorch_output_sha256", "onnx_output_sha256")): raise ValueError("Neural motion ONNX parity case drifted.")
    return manifest


def validate_export_bundle(bundle: Path, *, production: Path | None = None, replay: bool = False) -> dict[str, Any]:
    bundle = Path(bundle).resolve(); manifest = _validate_bundle(bundle)
    if replay:
        if production is None: raise ValueError("Neural motion ONNX replay requires its checkpoint authority.")
        production = Path(production).resolve(); contract = _read_canonical_json(production / CONTRACT_NAME); checkpoint_path = production / checkpoint_name(manifest["checkpoint"]["step"]); checkpoint = _load_checkpoint(checkpoint_path, contract, expected_step=manifest["checkpoint"]["step"])
        expected_checkpoint = {"path": checkpoint_path.name, "bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path), "step": checkpoint["step"], "model_state_sha256": checkpoint["model_state_sha256"], "ema_state_sha256": checkpoint["ema_state_sha256"]}
        if manifest["contract_semantic_sha256"] != contract["semantic_sha256"] or manifest["corpus_semantic_sha256"] != contract["corpus"]["semantic_sha256"] or manifest["checkpoint"] != expected_checkpoint or manifest["model"]["config"] != contract["model"]: raise ValueError("Neural motion ONNX checkpoint authority drifted.")
        model = NeuralCellMotionUNet(_config_from_dict(contract["model"])); model.load_state_dict(checkpoint["ema_state"], strict=True); parity = _parity(model, bundle / MODEL_NAME)
        if parity != manifest["parity"]: raise ValueError("Neural motion ONNX exact parity replay drifted.")
    return {"passed": True, "replay": replay, "step": manifest["checkpoint"]["step"], "parameters": manifest["model"]["parameters"], "onnx_sha256": manifest["artifact"]["sha256"], "max_abs_error": manifest["parity"]["max_abs_error"], "semantic_sha256": manifest["semantic_sha256"]}
