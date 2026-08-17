from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from ..cellular_nca.contract import CellularNCAConfig
from ..cellular_nca.corpus import load_corpus
from ..cellular_nca.evaluation import _mae
from ..cellular_nca.model import OrganismCellularAutomaton
from ..cellular_nca.teacher import make_scenarios, teacher_step
from ..cellular_nca_causal.evaluation import _relative_change, _teacher_relative
from ..cellular_nca_causal_v3.training import load_final
from ..map_topology_neural_production.checkpoint import tensor_state_sha256
from ..safety import require_disk_floor
from .contract import DEFAULT_AUTHORITY, DEFAULT_OUTPUT, FORMAT, RUNTIME_FORMAT, canonical, sha256_file, source_sha256


MANIFEST_NAME = "selection_manifest.json"
RUNTIME_NAME = "runtime.pt"
SYSTEMS = (("circulation", 28, 1), ("respiration", 31, 4), ("digestion", 34, 3), ("neural", 37, 8))


def _atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    return tensor_state_sha256({name: value.detach().cpu() for name, value in state.items()})


@torch.inference_mode()
def _candidate_metrics(model: torch.nn.Module, static: torch.Tensor, initial: torch.Tensor, bonds: torch.Tensor, injured: torch.Tensor, live: torch.Tensor, teacher: torch.Tensor) -> dict[str, Any]:
    predicted = injured.clone()
    for _ in range(32):
        predicted = model(static, predicted, live)
    rollout = _mae(predicted, teacher, static)
    rows = []
    for name, start, readout in SYSTEMS:
        rows.append({
            "system": name,
            "readout": readout,
            "relative_change": round(_relative_change(model, static, initial, bonds, start, readout, 16), 8),
            "teacher_relative_change": round(_teacher_relative(static, initial, bonds, start, readout, 16, .1), 8),
        })
    counterfactual_mae = float(np.mean([abs(row["relative_change"] - row["teacher_relative_change"]) for row in rows]))
    return {"organ_counterfactuals": rows, "counterfactual_mae": round(counterfactual_mae, 8), "rollout_mae": rollout}


def _gates(metrics: dict[str, Any], parent_error: float) -> dict[str, bool]:
    rows = metrics["organ_counterfactuals"]; rollout = metrics["rollout_mae"]
    return {
        "all_values_finite": bool(np.isfinite([metrics["counterfactual_mae"], *rollout.values(), *[row["relative_change"] for row in rows]]).all()),
        "all_four_organs_reduce_readout": all(row["relative_change"] < -.005 for row in rows),
        "counterfactual_error_improves_parent_25_percent": metrics["counterfactual_mae"] < parent_error * .75,
        "health_rollout_mae_below_0_02": rollout["health"] < .02,
        "fluid_rollout_mae_below_0_04": rollout["fluid"] < .04,
        "neural_rollout_mae_below_0_06": rollout["neural_activity"] < .06,
    }


@torch.inference_mode()
def evaluate(authority: Path = DEFAULT_AUTHORITY, output: Path = DEFAULT_OUTPUT, *, device_name: str = "cuda") -> dict[str, Any]:
    authority = Path(authority).resolve(); output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=256 * 1024**2)
    if device_name == "cuda" and (os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available()):
        raise RuntimeError("selection requires deterministic CUDA")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(0)
    ema_model, checkpoint, contract = load_final(authority)
    raw_model = OrganismCellularAutomaton(CellularNCAConfig(**contract["model"])); raw_model.load_state_dict(checkpoint["model_state"], strict=True); raw_model.eval()
    arrays = load_corpus(Path(contract["corpus_parent"]["path"]))["arrays"]
    device = torch.device(device_name)
    static = torch.from_numpy(arrays["static"]).to(device); initial = torch.from_numpy(arrays["initial_state"]).to(device); bonds = torch.from_numpy(arrays["live_bonds"]).to(device)
    generator = torch.Generator(device=device).manual_seed(contract["seed"] ^ 0x4556414C)
    injured, live = make_scenarios(static, initial, bonds, generator)
    teacher = injured.clone()
    for _ in range(32):
        teacher = teacher_step(static, teacher, live, .1)
    parent_error = 0.04293266
    candidates: dict[str, Any] = {}
    models = {"raw": raw_model, "ema": ema_model}
    for name, model in models.items():
        model.to(device).eval()
        metrics = _candidate_metrics(model, static, initial, bonds, injured, live, teacher)
        candidates[name] = {"metrics": metrics, "gates": _gates(metrics, parent_error)}
        model.cpu(); torch.cuda.empty_cache() if device.type == "cuda" else None
    passing = [name for name, row in candidates.items() if all(row["gates"].values())]
    selected = min(passing, key=lambda name: candidates[name]["metrics"]["counterfactual_mae"]) if passing else None
    output.mkdir(parents=True)
    runtime_path = output / RUNTIME_NAME
    if selected is not None:
        selected_state = checkpoint["model_state"] if selected == "raw" else checkpoint["ema_state"]
        runtime = {
            "format": RUNTIME_FORMAT, "source_sha256": source_sha256(), "selected": selected,
            "authority_checkpoint_sha256": sha256_file(authority / "causal_v3_segment_0000256.pt"),
            "model": contract["model"], "model_state": selected_state,
            "model_state_sha256": _state_hash(selected_state),
        }
        temporary = output / f".{RUNTIME_NAME}.tmp-{os.getpid()}"; torch.save(runtime, temporary); os.replace(temporary, runtime_path)
    manifest: dict[str, Any] = {
        "format": FORMAT, "status": "ready" if selected is not None else "experimental", "source_sha256": source_sha256(),
        "authority": {"path": str(authority), "checkpoint_sha256": sha256_file(authority / "causal_v3_segment_0000256.pt"), "ema_state_sha256": checkpoint["ema_state_sha256"], "model_state_sha256": checkpoint["model_state_sha256"]},
        "evaluation": {"parent_counterfactual_mae": parent_error, "candidates": candidates, "selected": selected},
        "runtime": None if selected is None else {"path": RUNTIME_NAME, "bytes": runtime_path.stat().st_size, "sha256": sha256_file(runtime_path)},
        "gates": {"at_least_one_candidate_passes": selected is not None, "selected_candidate_passes_every_gate": selected is not None and all(candidates[selected]["gates"].values())},
        "limitations": ["The model imitates a deterministic physiology scaffold.", "Static cell topology and bond geometry remain authoritative inputs."],
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    _atomic(output / MANIFEST_NAME, canonical(manifest))
    validate(output)
    return manifest


def validate(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output).resolve(); raw = (output / MANIFEST_NAME).read_bytes(); manifest = json.loads(raw)
    if raw != canonical(manifest) or manifest.get("format") != FORMAT or manifest.get("source_sha256") != source_sha256():
        raise ValueError("selection manifest provenance drifted")
    expected = hashlib.sha256(canonical({key: value for key, value in manifest.items() if key != "manifest_sha256"})).hexdigest()
    if manifest.get("manifest_sha256") != expected:
        raise ValueError("selection manifest hash drifted")
    selected = manifest["evaluation"]["selected"]
    if (manifest["status"] == "ready") != (selected is not None) or (selected is not None and not all(manifest["evaluation"]["candidates"][selected]["gates"].values())):
        raise ValueError("selection gate/status drifted")
    if selected is not None:
        path = output / manifest["runtime"]["path"]
        if path.stat().st_size != manifest["runtime"]["bytes"] or sha256_file(path) != manifest["runtime"]["sha256"]:
            raise ValueError("selection runtime artifact drifted")
        load_runtime(output)
    return {"passed": manifest["status"] == "ready", "selected": selected, "manifest_sha256": manifest["manifest_sha256"]}


def load_runtime(output: Path = DEFAULT_OUTPUT):
    output = Path(output).resolve(); manifest = json.loads((output / MANIFEST_NAME).read_bytes()); payload = torch.load(output / RUNTIME_NAME, map_location="cpu", weights_only=True)
    required = {"format", "source_sha256", "selected", "authority_checkpoint_sha256", "model", "model_state", "model_state_sha256"}
    if set(payload) != required or payload["format"] != RUNTIME_FORMAT or payload["source_sha256"] != source_sha256() or payload["selected"] != manifest["evaluation"]["selected"]:
        raise ValueError("selection runtime contract drifted")
    if _state_hash(payload["model_state"]) != payload["model_state_sha256"]:
        raise ValueError("selection runtime state drifted")
    model = OrganismCellularAutomaton(CellularNCAConfig(**payload["model"])); model.load_state_dict(payload["model_state"], strict=True); model.eval()
    return model, payload, manifest
