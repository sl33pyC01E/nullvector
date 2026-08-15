from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import warnings

from jsonschema import Draft202012Validator
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from .contract import CONTROL_FEATURES, MAX_CELLS, OUTPUT_FEATURES, STATE_FEATURES, STATIC_FEATURES, source_sha256
from .evaluation import _load_authority, evaluation_source_sha256
from .training import _atomic_bytes, _canonical, _sha256_file


EXPORT_FORMAT = "nullvector-creature-stage-neural-motion-onnx-v1"
EXPORT_NAME = "cellular_motion.onnx"
MANIFEST_NAME = "export_manifest.json"
EXPORT_SCHEMA = PROJECT_ROOT / "shared/schema/creature_stage_neural_motion_export.schema.json"
EXPORT_SOURCE_FILES = (
    "forge/creature_stage_neural_motion/export.py",
    "shared/schema/creature_stage_neural_motion_export.schema.json",
)
MAX_ONNX_BYTES = 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
PARITY_MOTIONS = (0, 2, 9, 12)
PARITY_FRAMES = (0, 17, 35, 71)
PARITY_MAX_ABS = 5e-5
PARITY_MEAN_ABS = 5e-6


class _ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        static: Tensor,
        state: Tensor,
        mask: Tensor,
        adjacency: Tensor,
        family: Tensor,
        morphotype: Tensor,
        motion: Tensor,
        phase: Tensor,
        controls: Tensor,
    ) -> Tensor:
        return self.model(
            static, state, mask, adjacency, family, morphotype, motion, phase, controls
        )


def export_source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-cellular-motion-onnx-export-source-v1\0")
    for relative in EXPORT_SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path.as_posix(), sess_options=options, providers=["CPUExecutionProvider"])


def _batch(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    result = {
        name: np.stack([row[name] for row in rows]).copy()
        for name in ("static", "state", "mask", "adjacency", "controls")
    }
    for name in ("family", "morphotype", "motion"):
        result[name] = np.asarray([int(row[name]) for row in rows], dtype=np.int64)
    result["phase"] = np.asarray([float(row["phase"]) for row in rows], dtype=np.float32)
    return result


def _torch_output(model: nn.Module, batch: dict[str, np.ndarray]) -> np.ndarray:
    with torch.inference_mode():
        value = model(
            *[
                torch.from_numpy(batch[name])
                for name in (
                    "static", "state", "mask", "adjacency", "family",
                    "morphotype", "motion", "phase", "controls",
                )
            ]
        )
    return value.detach().cpu().numpy()


def _parity(model: nn.Module, teacher: Any, session: ort.InferenceSession) -> dict[str, Any]:
    model.cpu().eval()
    chassis_ids = teacher.split_chassis("validation")
    if len(chassis_ids) != 5:
        raise ValueError("cellular motion export validation split drifted")
    absolute: list[np.ndarray] = []
    maximum_outside = 0.0
    output_min = math.inf
    output_max = -math.inf
    cases: list[dict[str, Any]] = []
    for motion in PARITY_MOTIONS:
        for frame in PARITY_FRAMES:
            rows = [teacher.sample(chassis, motion, frame) for chassis in chassis_ids]
            batch = _batch(rows)
            reference = _torch_output(model, batch)
            portable = session.run(["predicted"], batch)[0]
            if portable.shape != (5, MAX_CELLS, OUTPUT_FEATURES) or portable.dtype != np.float32:
                raise ValueError("cellular motion ONNX output contract drifted")
            difference = np.abs(portable - reference)
            absolute.append(difference.reshape(-1))
            outside = ~batch["mask"]
            maximum_outside = max(maximum_outside, float(np.abs(portable[outside]).max(initial=0.0)))
            output_min = min(output_min, float(portable.min()))
            output_max = max(output_max, float(portable.max()))
            cases.append(
                {
                    "motion_id": motion,
                    "frame": frame,
                    "batch": 5,
                    "max_abs": round(float(difference.max(initial=0.0)), 10),
                    "mean_abs": round(float(difference.mean()), 10),
                }
            )
    merged = np.concatenate(absolute)
    dynamic_cases: list[dict[str, Any]] = []
    dynamic_rows = [teacher.sample(chassis, 0, 0) for chassis in chassis_ids]
    for size in (1, 3, 5):
        batch = _batch(dynamic_rows[:size])
        reference = _torch_output(model, batch)
        portable = session.run(["predicted"], batch)[0]
        if portable.shape != (size, MAX_CELLS, OUTPUT_FEATURES):
            raise ValueError("cellular motion ONNX dynamic batch drifted")
        difference = np.abs(portable - reference)
        dynamic_cases.append(
            {
                "batch": size,
                "max_abs": round(float(difference.max(initial=0.0)), 10),
                "mean_abs": round(float(difference.mean()), 10),
            }
        )
    result = {
        "split": "validation",
        "families": 5,
        "motions": list(PARITY_MOTIONS),
        "frames": list(PARITY_FRAMES),
        "cases": cases,
        "dynamic_batch_cases": dynamic_cases,
        "examples": len(cases) * 5,
        "values_compared": int(merged.size),
        "max_abs": round(float(merged.max(initial=0.0)), 10),
        "mean_abs": round(float(merged.mean()), 10),
        "p99_abs": round(float(np.quantile(merged, 0.99)), 10),
        "maximum_outside_abs": round(maximum_outside, 12),
        "output_min": round(output_min, 9),
        "output_max": round(output_max, 9),
    }
    if not all(math.isfinite(float(result[name])) for name in ("max_abs", "mean_abs", "p99_abs", "output_min", "output_max")):
        raise FloatingPointError("cellular motion ONNX parity became non-finite")
    return result


def _gates(payload: dict[str, Any]) -> dict[str, bool]:
    parity = payload["parity"]
    return {
        "onnx_checker_passed": payload["onnx"]["checker_passed"] is True,
        "heldout_matrix_complete": (
            parity["families"] == 5
            and parity["motions"] == list(PARITY_MOTIONS)
            and parity["frames"] == list(PARITY_FRAMES)
            and parity["examples"] == 80
        ),
        "dynamic_batch_verified": [row["batch"] for row in parity["dynamic_batch_cases"]] == [1, 3, 5],
        "maximum_error_within_tolerance": parity["max_abs"] <= payload["tolerances"]["max_abs"],
        "mean_error_within_tolerance": parity["mean_abs"] <= payload["tolerances"]["mean_abs"],
        "outside_cells_exact_zero": parity["maximum_outside_abs"] == 0.0,
        "bounded_output": parity["output_min"] >= -1.000001 and parity["output_max"] <= 1.000001,
        "checkpoint_exactly_bound": payload["checkpoint"]["sha256"] == payload["onnx"]["checkpoint_sha256"],
    }


def _export_model(model: nn.Module, teacher: Any, path: Path) -> None:
    rows = [teacher.sample(chassis, 0, 0) for chassis in teacher.split_chassis("validation")]
    batch = _batch(rows)
    arguments = tuple(
        torch.from_numpy(batch[name])
        for name in (
            "static", "state", "mask", "adjacency", "family", "morphotype",
            "motion", "phase", "controls",
        )
    )
    input_names = [
        "static", "state", "mask", "adjacency", "family", "morphotype",
        "motion", "phase", "controls",
    ]
    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes["predicted"] = {0: "batch"}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            _ExportWrapper(model.cpu().eval()),
            arguments,
            path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=input_names,
            output_names=["predicted"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
            external_data=False,
        )


def _inspect_onnx(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_ONNX_BYTES:
        raise ValueError("cellular motion ONNX artifact is missing or oversized")
    model = onnx.load(path, load_external_data=False)
    if any(initializer.data_location == onnx.TensorProto.EXTERNAL for initializer in model.graph.initializer):
        raise ValueError("cellular motion ONNX export unexpectedly uses external data")
    onnx.checker.check_model(model, full_check=True)
    inputs = {item.name: item for item in model.graph.input}
    if set(inputs) != {
        "static", "state", "mask", "adjacency", "family", "morphotype",
        "motion", "phase", "controls",
    } or [item.name for item in model.graph.output] != ["predicted"]:
        raise ValueError("cellular motion ONNX I/O registry drifted")
    expected_signatures = {
        "static": ["batch", MAX_CELLS, STATIC_FEATURES],
        "state": ["batch", MAX_CELLS, STATE_FEATURES],
        "mask": ["batch", MAX_CELLS],
        "adjacency": ["batch", MAX_CELLS, MAX_CELLS],
        "family": ["batch"],
        "morphotype": ["batch"],
        "motion": ["batch"],
        "phase": ["batch"],
        "controls": ["batch", CONTROL_FEATURES],
    }
    expected_types = {
        "static": onnx.TensorProto.FLOAT,
        "state": onnx.TensorProto.FLOAT,
        "mask": onnx.TensorProto.BOOL,
        "adjacency": onnx.TensorProto.BOOL,
        "family": onnx.TensorProto.INT64,
        "morphotype": onnx.TensorProto.INT64,
        "motion": onnx.TensorProto.INT64,
        "phase": onnx.TensorProto.FLOAT,
        "controls": onnx.TensorProto.FLOAT,
    }
    signatures: dict[str, dict[str, Any]] = {}
    for name, item in inputs.items():
        tensor = item.type.tensor_type
        shape = [dimension.dim_param if dimension.dim_param else int(dimension.dim_value) for dimension in tensor.shape.dim]
        if tensor.elem_type != expected_types[name] or shape != expected_signatures[name]:
            raise ValueError("cellular motion ONNX input signature drifted")
        signatures[name] = {"dtype": onnx.TensorProto.DataType.Name(tensor.elem_type), "shape": shape}
    output_tensor = model.graph.output[0].type.tensor_type
    output_shape = [dimension.dim_param if dimension.dim_param else int(dimension.dim_value) for dimension in output_tensor.shape.dim]
    if output_tensor.elem_type != onnx.TensorProto.FLOAT or output_shape != ["batch", MAX_CELLS, OUTPUT_FEATURES]:
        raise ValueError("cellular motion ONNX output signature drifted")
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "opset": max(imported.version for imported in model.opset_import),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "inputs": {name: signatures[name] for name in sorted(signatures)},
        "output": {"dtype": "FLOAT", "shape": output_shape},
        "checker_passed": True,
    }


def export_checkpoint(checkpoint_path: Path, output: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    model, authority, teacher = _load_authority(checkpoint_path)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        onnx_path = temporary / EXPORT_NAME
        _export_model(model, teacher, onnx_path)
        onnx_record = _inspect_onnx(onnx_path)
        onnx_record["checkpoint_sha256"] = authority["sha256"]
        session = _session(onnx_path)
        parity = _parity(model, teacher, session)
        payload: dict[str, Any] = {
            "format": EXPORT_FORMAT,
            "status": "exported",
            "export_source_sha256": export_source_sha256(),
            "evaluation_source_sha256": evaluation_source_sha256(),
            "model_source_sha256": source_sha256(),
            "checkpoint": authority,
            "teacher": {
                "path": teacher.root.relative_to(PROJECT_ROOT).as_posix(),
                "semantic_sha256": teacher.semantic_sha256,
                "manifest_sha256": teacher.validation["manifest_sha256"],
                "binary_sha256": teacher.validation["binary_sha256"],
            },
            "interface": {
                "max_cells": MAX_CELLS,
                "static_features": STATIC_FEATURES,
                "state_features": STATE_FEATURES,
                "control_features": CONTROL_FEATURES,
                "output_features": OUTPUT_FEATURES,
                "dynamic_batch": True,
                "stateful_runtime": "caller-feeds-previous-prediction",
            },
            "onnx": onnx_record,
            "runtime": {
                "torch": str(torch.__version__),
                "onnx": str(onnx.__version__),
                "onnxruntime": str(ort.__version__),
                "provider": "CPUExecutionProvider",
            },
            "tolerances": {"max_abs": PARITY_MAX_ABS, "mean_abs": PARITY_MEAN_ABS},
            "parity": parity,
        }
        payload["gates"] = _gates(payload)
        if not all(payload["gates"].values()):
            raise ValueError(f"cellular motion ONNX export parity failed: {payload['gates']}")
        payload["semantic_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
        _atomic_bytes(temporary / MANIFEST_NAME, _canonical(payload))
        validate_export(temporary, replay=True)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_export(output, replay=True)


def _validate_structure(payload: dict[str, Any]) -> None:
    checkpoint_keys = {
        "kind", "format", "path", "bytes", "sha256", "step",
        "model_state_sha256", "ema_state_sha256", "contract_semantic_sha256",
        "smoke_semantic_sha256",
    }
    if set(payload["checkpoint"]) != checkpoint_keys:
        raise ValueError("cellular motion ONNX checkpoint registry drifted")
    if payload["interface"] != {
        "max_cells": MAX_CELLS,
        "static_features": STATIC_FEATURES,
        "state_features": STATE_FEATURES,
        "control_features": CONTROL_FEATURES,
        "output_features": OUTPUT_FEATURES,
        "dynamic_batch": True,
        "stateful_runtime": "caller-feeds-previous-prediction",
    }:
        raise ValueError("cellular motion ONNX interface drifted")
    expected_case_keys = {
        (motion, frame) for motion in PARITY_MOTIONS for frame in PARITY_FRAMES
    }
    if (
        {(row["motion_id"], row["frame"]) for row in payload["parity"]["cases"]} != expected_case_keys
        or len(payload["parity"]["cases"]) != 16
        or [row["batch"] for row in payload["parity"]["dynamic_batch_cases"]] != [1, 3, 5]
        or payload["tolerances"] != {"max_abs": PARITY_MAX_ABS, "mean_abs": PARITY_MEAN_ABS}
        or payload["gates"] != _gates(payload)
        or not all(payload["gates"].values())
    ):
        raise ValueError("cellular motion ONNX derived evidence drifted")


def validate_export(output: Path, *, replay: bool = True) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest_path = output / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file() or not 0 < manifest_path.stat().st_size <= MAX_MANIFEST_BYTES:
        raise ValueError("cellular motion ONNX manifest is missing or oversized")
    raw = manifest_path.read_bytes()
    payload = json.loads(raw)
    errors = sorted(
        Draft202012Validator(json.loads(EXPORT_SCHEMA.read_bytes())).iter_errors(payload),
        key=lambda error: list(error.path),
    )
    if raw != _canonical(payload) or errors:
        detail = errors[0].message if errors else "noncanonical JSON"
        raise ValueError(f"cellular motion ONNX manifest structure drifted: {detail}")
    if (
        payload["format"] != EXPORT_FORMAT
        or payload["status"] != "exported"
        or payload["export_source_sha256"] != export_source_sha256()
        or payload["evaluation_source_sha256"] != evaluation_source_sha256()
        or payload["model_source_sha256"] != source_sha256()
        or payload["semantic_sha256"]
        != hashlib.sha256(_canonical({key: value for key, value in payload.items() if key != "semantic_sha256"})).hexdigest()
    ):
        raise ValueError("cellular motion ONNX authority drifted")
    _validate_structure(payload)
    artifact_path = output / payload["onnx"]["path"]
    if (
        artifact_path.is_symlink()
        or not artifact_path.is_file()
        or artifact_path.stat().st_size != payload["onnx"]["bytes"]
        or _sha256_file(artifact_path) != payload["onnx"]["sha256"]
    ):
        raise ValueError("cellular motion ONNX artifact bytes drifted")
    onnx_record = _inspect_onnx(artifact_path)
    onnx_record["checkpoint_sha256"] = payload["checkpoint"]["sha256"]
    if onnx_record != payload["onnx"]:
        raise ValueError("cellular motion ONNX artifact provenance drifted")
    checkpoint_path = PROJECT_ROOT / payload["checkpoint"]["path"]
    model, authority, teacher = _load_authority(checkpoint_path)
    if authority != payload["checkpoint"] or teacher.semantic_sha256 != payload["teacher"]["semantic_sha256"]:
        raise ValueError("cellular motion ONNX checkpoint provenance drifted")
    parity = _parity(model, teacher, _session(artifact_path))
    if parity != payload["parity"]:
        raise ValueError("cellular motion ONNX numerical replay drifted")
    return {
        "passed": True,
        "checkpoint_kind": payload["checkpoint"]["kind"],
        "onnx_sha256": payload["onnx"]["sha256"],
        "onnx_bytes": payload["onnx"]["bytes"],
        "examples": payload["parity"]["examples"],
        "max_abs": payload["parity"]["max_abs"],
        "mean_abs": payload["parity"]["mean_abs"],
        "semantic_sha256": payload["semantic_sha256"],
        "gates": payload["gates"],
    }
