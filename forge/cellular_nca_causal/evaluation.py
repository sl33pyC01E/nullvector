from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from jsonschema import Draft202012Validator

from ..cellular_nca.contract import canonical_json_bytes
from ..cellular_nca.corpus import load_corpus
from ..cellular_nca.evaluation import _damage_system, _mae
from ..cellular_nca.teacher import make_scenarios, teacher_step
from ..cellular_nca.training import load_final_checkpoint as load_parent_checkpoint
from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from .contract import MANIFEST_NAME, REQUIRED_GATES, TRAINING_NAME, VISUAL_NAME, causal_source_sha256, read_canonical_json, sha256_bytes, sha256_file
from .curriculum import SYSTEMS
from .training import _load_telemetry, checkpoint_name, load_final_checkpoint


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent); temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def _manifest_hash(manifest: dict[str, Any]) -> str:
    import hashlib
    return hashlib.sha256(canonical_json_bytes({key: value for key, value in manifest.items() if key != "manifest_sha256"})).hexdigest()


def _relative_change(model: torch.nn.Module, static: torch.Tensor, initial: torch.Tensor, bonds: torch.Tensor, start: int, readout: int, steps: int) -> float:
    damaged = _damage_system(initial, static, start); control = initial.clone()
    for _ in range(steps): damaged = model(static, damaged, bonds); control = model(static, control, bonds)
    body = static[:, :1]; damaged_value = (damaged[:, readout : readout + 1] * body).sum() / body.sum(); control_value = (control[:, readout : readout + 1] * body).sum() / body.sum()
    return float((damaged_value - control_value) / control_value.clamp_min(1e-6))


def _teacher_relative(static: torch.Tensor, initial: torch.Tensor, bonds: torch.Tensor, start: int, readout: int, steps: int, dt: float) -> float:
    damaged = _damage_system(initial, static, start); control = initial.clone()
    for _ in range(steps): damaged = teacher_step(static, damaged, bonds, dt); control = teacher_step(static, control, bonds, dt)
    body = static[:, :1]; damaged_value = (damaged[:, readout : readout + 1] * body).sum() / body.sum(); control_value = (control[:, readout : readout + 1] * body).sum() / body.sum()
    return float((damaged_value - control_value) / control_value.clamp_min(1e-6))


def _render(rows: list[dict[str, Any]]) -> bytes:
    import io
    width, height = 960, 110 + 112 * len(rows); image = Image.new("RGB", (width, height), (3, 8, 14)); draw = ImageDraw.Draw(image)
    draw.text((18, 16), "ORGAN COUNTERFACTUAL RESPONSE // 16 RECURRENT STEPS", fill=(91, 239, 255)); draw.text((18, 38), "relative capacity loss after targeted organ ablation // farther left is stronger failure response", fill=(148, 172, 190)); zero_x = 710
    draw.line((zero_x, 70, zero_x, height - 20), fill=(55, 76, 91), width=1)
    colors = {"parent": (119, 135, 154), "causal": (184, 255, 73), "teacher": (255, 83, 224)}
    for row_index, row in enumerate(rows):
        y = 86 + row_index * 112; draw.text((18, y), row["system"].upper(), fill=(225, 236, 243))
        for index, key in enumerate(("parent", "causal", "teacher")):
            value = float(row[f"{key}_relative_change"]); bar_y = y + 24 + index * 23; x = zero_x + int(max(-.75, min(.1, value)) * 720)
            draw.text((126, bar_y - 2), key.upper(), fill=colors[key]); draw.rectangle((min(x, zero_x), bar_y, max(x, zero_x), bar_y + 12), fill=colors[key]); draw.text((730, bar_y - 2), f"{value:+.5f}", fill=colors[key])
    stream = io.BytesIO(); image.save(stream, format="PNG", compress_level=9, optimize=False); return stream.getvalue()


@torch.no_grad()
def _evaluate_payload(output: Path, *, device_name: str, rollout_steps: int, causal_steps: int) -> tuple[dict[str, Any], bytes]:
    output = Path(output).resolve(); require_disk_floor(output, floor_gb=100, planned_bytes=64 * 1024**2)
    if rollout_steps != 32 or causal_steps != 16: raise ValueError("Causal evaluation horizon drifted.")
    device = torch.device(device_name)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(0)
    if device.type == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available(): raise RuntimeError("Deterministic CUDA evaluation unavailable.")
    causal_model, checkpoint, contract = load_final_checkpoint(output); parent_output = PROJECT_ROOT / contract["parent"]["path"]; parent_model, _, _ = load_parent_checkpoint(parent_output)
    causal_model.to(device).eval(); parent_model.to(device).eval(); corpus = load_corpus(parent_output); arrays = corpus["arrays"]; static = torch.from_numpy(arrays["static"]).to(device); initial = torch.from_numpy(arrays["initial_state"]).to(device); bonds = torch.from_numpy(arrays["live_bonds"]).to(device)
    generator = torch.Generator(device=device).manual_seed(contract["seed"] ^ 0x4556414C); injured, live = make_scenarios(static, initial, bonds, generator); teacher = injured.clone(); parent = injured.clone(); causal = injured.clone()
    for _ in range(rollout_steps): teacher = teacher_step(static, teacher, live, contract["curriculum"]["teacher_dt"]); parent = parent_model(static, parent, live); causal = causal_model(static, causal, live)
    parent_mae = _mae(parent, teacher, static); causal_mae = _mae(causal, teacher, static)
    rows: list[dict[str, Any]] = []
    for name, start, readout in SYSTEMS:
        rows.append({"system": name, "readout": readout, "parent_relative_change": round(_relative_change(parent_model, static, initial, bonds, start, readout, causal_steps), 8), "causal_relative_change": round(_relative_change(causal_model, static, initial, bonds, start, readout, causal_steps), 8), "teacher_relative_change": round(_teacher_relative(static, initial, bonds, start, readout, causal_steps, contract["curriculum"]["teacher_dt"]), 8)})
    parent_error = float(np.mean([abs(row["parent_relative_change"] - row["teacher_relative_change"]) for row in rows])); causal_error = float(np.mean([abs(row["causal_relative_change"] - row["teacher_relative_change"]) for row in rows]))
    visual = _render(rows); telemetry = _load_telemetry(output); finite_values = [value for row in rows for key, value in row.items() if key.endswith("relative_change")] + list(parent_mae.values()) + list(causal_mae.values()) + [parent_error, causal_error]
    gates = {"all_values_finite": bool(np.isfinite(finite_values).all()), "all_four_organs_reduce_their_readout": all(row["causal_relative_change"] < -.005 for row in rows), "counterfactual_error_improves_over_parent": causal_error < parent_error * .75, "general_health_mae_below_0_02": causal_mae["health"] < .02, "general_fluid_mae_below_0_04": causal_mae["fluid"] < .04, "general_neural_mae_below_0_06": causal_mae["neural_activity"] < .06, "general_rollout_not_regressed_50_percent": all(causal_mae[name] <= max(parent_mae[name] * 1.5, parent_mae[name] + .002) for name in ("health", "fluid", "energy", "oxygen", "neural_activity"))}
    segment_count = contract["total_steps"] // contract["segment_steps"]; attempt_count = len(telemetry["attempts"]); retry_count = attempt_count - segment_count
    if retry_count < 0 or set(gates) != set(REQUIRED_GATES) or sum(row["artifact_valid"] for row in telemetry["attempts"]) != segment_count: raise ValueError("Causal evaluation provenance census drifted.")
    manifest = {"format": "nullvector-organism-neural-cellular-automaton-causal-v2", "status": "ready" if all(gates.values()) else "experimental", "source_sha256": causal_source_sha256(), "parent": contract["parent"], "training": {"total_steps": contract["total_steps"], "segment_steps": contract["segment_steps"], "batch_size": contract["batch_size"], "attempt_count": attempt_count, "retry_count": retry_count, "recovered_checkpoint_count": sum(row["recovered_checkpoint"] for row in telemetry["attempts"]), "first_loss": checkpoint["history"][0]["loss"], "final_loss": checkpoint["history"][-1]["loss"], "runtime": checkpoint["runtime"]}, "provenance": {"training_contract_sha256": sha256_file(output / TRAINING_NAME), "checkpoint_path": checkpoint_name(contract["total_steps"]), "checkpoint_sha256": sha256_file(output / checkpoint_name(contract["total_steps"])), "model_state_sha256": checkpoint["model_state_sha256"], "ema_state_sha256": checkpoint["ema_state_sha256"]}, "evaluation": {"rollout_steps": rollout_steps, "causal_steps": causal_steps, "organ_counterfactuals": rows, "parent_counterfactual_mae": round(parent_error, 8), "causal_counterfactual_mae": round(causal_error, 8), "parent_rollout_mae": parent_mae, "causal_rollout_mae": causal_mae}, "visual": {"path": VISUAL_NAME, "bytes": len(visual), "sha256": sha256_bytes(visual), "visually_inspected": False}, "gates": gates, "limitations": ["The v2 curriculum still imitates the deterministic physiology teacher.", "Static anatomy and bond topology remain authoritative inputs.", "Ecological behavior and learned bond fracture are outside this compact causal fine-tune."]}
    manifest["manifest_sha256"] = _manifest_hash(manifest); return manifest, visual


def evaluate(output: Path, *, device_name: str = "cpu", rollout_steps: int = 32, causal_steps: int = 16) -> dict[str, Any]:
    output = Path(output).resolve()
    if (output / MANIFEST_NAME).exists(): raise FileExistsError("Finalized causal NCA output is immutable.")
    manifest, visual = _evaluate_payload(output, device_name=device_name, rollout_steps=rollout_steps, causal_steps=causal_steps)
    _atomic_bytes(output / VISUAL_NAME, visual); _atomic_bytes(output / MANIFEST_NAME, canonical_json_bytes(manifest)); return manifest


def validate_output(output: Path, *, rerun_evaluation: bool = True, device_name: str = "cpu") -> dict[str, Any]:
    output = Path(output).resolve(); path = output / MANIFEST_NAME; manifest = read_canonical_json(path); schema = json.loads((PROJECT_ROOT / "shared/schema/cellular_nca_causal_manifest.schema.json").read_text(encoding="utf-8")); errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda error: list(error.absolute_path))
    if errors: raise ValueError(f"Causal NCA schema validation failed: {errors[0].message}")
    if manifest.get("source_sha256") != causal_source_sha256() or manifest.get("manifest_sha256") != _manifest_hash(manifest): raise ValueError("Causal NCA manifest drifted.")
    if set(manifest["gates"]) != set(REQUIRED_GATES) or (manifest["status"] == "ready") != all(manifest["gates"].values()): raise ValueError("Causal NCA gate/status contract drifted.")
    if [(row["system"], row["readout"]) for row in manifest["evaluation"]["organ_counterfactuals"]] != [(name, readout) for name, _, readout in SYSTEMS]: raise ValueError("Causal NCA organ census drifted.")
    if manifest["provenance"]["training_contract_sha256"] != sha256_file(output / TRAINING_NAME): raise ValueError("Causal NCA contract drifted.")
    checkpoint = output / PurePosixPath(manifest["provenance"]["checkpoint_path"]); visual = output / PurePosixPath(manifest["visual"]["path"])
    if sha256_file(checkpoint) != manifest["provenance"]["checkpoint_sha256"] or visual.stat().st_size != manifest["visual"]["bytes"] or sha256_file(visual) != manifest["visual"]["sha256"]: raise ValueError("Causal NCA artifact drifted.")
    load_final_checkpoint(output); telemetry = _load_telemetry(output)
    if manifest["training"]["attempt_count"] != len(telemetry["attempts"]) or manifest["training"]["recovered_checkpoint_count"] != sum(row["recovered_checkpoint"] for row in telemetry["attempts"]): raise ValueError("Causal NCA telemetry census drifted.")
    mode = "metadata_only"
    if rerun_evaluation:
        expected, expected_visual = _evaluate_payload(output, device_name=device_name, rollout_steps=32, causal_steps=16)
        if expected != manifest or expected_visual != visual.read_bytes(): raise ValueError("Causal NCA exact evaluation replay drifted.")
        mode = "exact_replay"
    return {"passed": all(manifest["gates"].values()), "status": manifest["status"], "mode": mode, "manifest_sha256": manifest["manifest_sha256"], "checkpoint_sha256": manifest["provenance"]["checkpoint_sha256"], "visual_sha256": manifest["visual"]["sha256"]}
