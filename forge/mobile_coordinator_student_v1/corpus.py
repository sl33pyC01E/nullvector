from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch

from ..android_ensemble_v2.export import sha256_file
from ..config import PROJECT_ROOT
from ..nature_colony_nn import NeuralColonyRuntime
from ..nature_colony_nn.corpus import build_corpus as build_colony_corpus
from ..nature_counterfactual_nn import NeuralCounterfactualRuntime
from ..nature_macro_nn import NeuralMacroPatchRuntime
from ..nature_macro_nn.corpus import validate_corpus as validate_macro_corpus
from ..nature_society_nn import NeuralSocietyRuntime
from ..nature_society_nn.corpus import build_corpus as build_society_corpus
from ..nature_timeline_nn import NeuralTimelineRuntime
from ..nature_timeline_nn.corpus import build_corpus as build_timeline_corpus
from ..playable_neural_runtime_v1.runtime import _component_table
from ..safety import require_disk_floor
from .contract import CORPUS_FORMAT, canonical, corpus_source_sha256


DEFAULT_MACRO = PROJECT_ROOT / "outputs/nature_macro_nn/corpus_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/mobile_coordinator_student_v1/corpus_001"
ARRAYS = (
    "current", "previous", "global_state", "previous_global", "members", "member_mask", "society", "sequence",
    "target_macro", "target_macro_global", "target_role", "target_member_action", "target_activity", "target_labor",
    "target_diplomacy", "target_project", "target_timeline", "target_event", "target_confidence",
    "target_counter_state", "target_benefit", "target_risk",
)


def _infer(model: torch.nn.Module, inputs: tuple[np.ndarray, ...], batch: int, device: torch.device) -> list[np.ndarray]:
    outputs: list[list[np.ndarray]] | None = None
    with torch.inference_mode():
        for start in range(0, len(inputs[0]), batch):
            tensors = tuple(torch.from_numpy(value[start:start + batch]).to(device) for value in inputs)
            result = model(*tensors); result = result if isinstance(result, tuple) else (result,)
            if outputs is None: outputs = [[] for _ in result]
            for rows, value in zip(outputs, result): rows.append(value.float().cpu().numpy())
    assert outputs is not None
    return [np.concatenate(rows) for rows in outputs]


def _release(model: torch.nn.Module) -> None:
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def _load_macro_inputs(root: Path, samples: int) -> dict[str, np.ndarray]:
    report = validate_macro_corpus(root); rows = {name: [] for name in ("previous", "current", "previous_global", "global_state")}
    manifest = json.loads((root / "manifest.json").read_bytes())
    for record in manifest["shards"]:
        with np.load(root / record["artifact"]["path"], allow_pickle=False) as archive:
            for name in rows: rows[name].append(archive[name])
    result = {name: np.concatenate(values)[:samples] for name, values in rows.items()}
    if len(result["current"]) != samples or report["pairs"] < samples: raise ValueError("macro corpus is too small")
    return result


def build(output: Path = DEFAULT_OUTPUT, *, samples: int = 1536, macro_root: Path = DEFAULT_MACRO, device: str = "cuda") -> dict[str, object]:
    output, macro_root = Path(output).resolve(), Path(macro_root).resolve()
    if output.exists(): raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1 << 30)
    if not 256 <= samples <= 8192: raise ValueError("coordinator corpus sample count drifted")
    selected_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    table = _component_table(); artifact = lambda name: PROJECT_ROOT / table[name]["artifact"]["path"]
    parents = {name: {"path": table[name]["artifact"]["path"], "sha256": sha256_file(artifact(name))} for name in ("macro_patch", "colony", "society", "timeline", "counterfactual")}
    macro = _load_macro_inputs(macro_root, samples)
    colony = build_colony_corpus(colonies=samples, seed=0x434F4F5244434F4C)
    society = build_society_corpus(samples=samples, seed=0x434F4F5244534F43)
    timeline = build_timeline_corpus(samples=samples, seed=0x434F4F524454494D)
    arrays: dict[str, np.ndarray] = {
        "current": macro["current"].astype(np.float16), "previous": macro["previous"].astype(np.float16),
        "global_state": macro["global_state"].astype(np.float32), "previous_global": macro["previous_global"].astype(np.float32),
        "members": colony["features"].astype(np.float16), "member_mask": colony["mask"].astype(np.uint8),
        "society": society["features"].astype(np.float16), "sequence": timeline["sequence"].astype(np.float16),
    }
    model = NeuralMacroPatchRuntime.from_checkpoint(artifact("macro_patch"), device=selected_device).model.eval()
    values = _infer(model, (arrays["current"].astype(np.float32), arrays["previous"].astype(np.float32), arrays["global_state"], arrays["previous_global"]), 24, selected_device)
    arrays["target_macro"], arrays["target_macro_global"] = values[0].astype(np.float16), values[1].astype(np.float16); _release(model)
    model = NeuralColonyRuntime.from_checkpoint(artifact("colony"), device=selected_device).model.eval()
    values = _infer(model, (arrays["members"].astype(np.float32), arrays["member_mask"].astype(np.bool_)), 96, selected_device)
    arrays["target_role"], arrays["target_member_action"] = values[0].astype(np.float16), values[1].astype(np.float16); _release(model)
    model = NeuralSocietyRuntime.from_checkpoint(artifact("society"), device=selected_device).model.eval()
    values = _infer(model, (arrays["society"].astype(np.float32),), 256, selected_device)
    arrays["target_activity"], arrays["target_labor"], arrays["target_diplomacy"], arrays["target_project"] = (value.astype(np.float16) for value in values); _release(model)
    model = NeuralTimelineRuntime.from_checkpoint(artifact("timeline"), device=selected_device).model.eval()
    values = _infer(model, (arrays["sequence"].astype(np.float32),), 128, selected_device)
    arrays["target_timeline"], arrays["target_event"], arrays["target_confidence"] = values[0].astype(np.float16), values[1].astype(np.float16), values[2].astype(np.float16); _release(model)
    model = NeuralCounterfactualRuntime.from_checkpoint(artifact("counterfactual"), device=selected_device).model.eval()
    repeated = np.repeat(arrays["sequence"].astype(np.float32), 5, axis=0); actions = np.tile(np.arange(5, dtype=np.int64), samples)
    values = _infer(model, (repeated, actions), 256, selected_device)
    arrays["target_counter_state"] = values[0].reshape(samples, 5, 64).astype(np.float16); arrays["target_benefit"] = values[1].reshape(samples, 5).astype(np.float16); arrays["target_risk"] = values[2].reshape(samples, 5).astype(np.float16); _release(model)
    if set(arrays) != set(ARRAYS) or any(len(value) != samples or not np.isfinite(value).all() for value in arrays.values()): raise ValueError("coordinator corpus closure drifted")
    staging = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}"); staging.mkdir(parents=True)
    try:
        artifact_path = staging / "corpus.npz"; np.savez_compressed(artifact_path, **arrays)
        semantic = hashlib.sha256(CORPUS_FORMAT.encode())
        for name in ARRAYS:
            value = np.ascontiguousarray(arrays[name]); semantic.update(name.encode() + value.dtype.str.encode() + np.asarray(value.shape, dtype="<i8").tobytes() + value.tobytes())
        payload = {"format": CORPUS_FORMAT, "status": "ready", "source_sha256": corpus_source_sha256(), "samples": samples, "parents": parents, "macro_corpus_manifest_sha256": validate_macro_corpus(macro_root, load_arrays=False)["manifest_sha256"], "arrays": {name: {"shape": list(value.shape), "dtype": value.dtype.str} for name, value in arrays.items()}, "semantic_sha256": semantic.hexdigest(), "artifact": {"path": artifact_path.name, "bytes": artifact_path.stat().st_size, "sha256": sha256_file(artifact_path)}}
        payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest(); (staging / "manifest.json").write_bytes(canonical(payload)); os.replace(staging, output)
    finally:
        if staging.exists(): shutil.rmtree(staging)
    return payload
