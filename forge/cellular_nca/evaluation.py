from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor
from jsonschema import Draft202012Validator

from ..safety import require_disk_floor
from .contract import DYNAMIC_NAMES, FORMAT, canonical_json_bytes, sha256_file, source_sha256
from ..config import PROJECT_ROOT
from .corpus import CORPUS_MANIFEST_NAME, load_corpus
from .teacher import make_scenarios, teacher_step
from .training import TELEMETRY_NAME, TRAINING_NAME, checkpoint_name, load_final_checkpoint


MANIFEST_NAME = "cellular_nca_manifest.json"
SHEET_NAME = "cellular_dynamics_rollout.png"


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent); temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def _png_bytes(image: Image.Image) -> bytes:
    import io
    stream = io.BytesIO(); image.save(stream, format="PNG", compress_level=9, optimize=False); return stream.getvalue()


def _render(static: np.ndarray, state: np.ndarray, family_id: int) -> Image.Image:
    palettes = np.asarray(((40, 214, 255), (255, 106, 72), (125, 255, 72), (214, 72, 255), (255, 204, 64)), dtype=np.float32)
    body = static[0]; health, fluid, energy, oxygen, clot, scar, wound, neural, surface, biomass, alive = state[0], state[1], state[3], state[4], state[5], state[6], state[7], state[8], state[9], state[10], state[11]
    base = palettes[family_id][None, None] * (.16 + .56 * health[..., None])
    base[..., 0] += 175 * wound + 85 * scar + 70 * biomass
    base[..., 1] += 105 * energy + 70 * clot
    base[..., 2] += 120 * fluid + 95 * oxygen + 130 * neural
    base *= body[..., None] * (.3 + .7 * alive[..., None])
    base[..., 0] += 42 * surface; base[..., 1] += 120 * surface; base[..., 2] += 240 * surface
    # Visible organ cores: circulation red, respiration cyan, digestion amber,
    # neural magenta; the overlay remains subordinate to actual cell health.
    roles = (static[28:31].sum(0), static[31:34].sum(0), static[34:37].sum(0), static[37:40].sum(0))
    colors = ((255, 38, 70), (40, 230, 255), (255, 184, 48), (236, 70, 255))
    for role, color in zip(roles, colors, strict=True):
        core = np.clip(role, 0, 1) * body * health
        base = base * (1 - core[..., None] * .42) + np.asarray(color)[None, None] * core[..., None] * .42
    image = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(image, "RGB").resize((192, 192), Image.Resampling.NEAREST)


def _mae(predicted: Tensor, target: Tensor, static: Tensor) -> dict[str, float]:
    body = static[:, :1]; support = torch.cat((body.expand(-1, 9, -1, -1), torch.ones_like(body).expand(-1, 2, -1, -1), body), 1)
    error = (predicted - target).abs() * support; denominator = support.sum((0, 2, 3)).clamp_min(1)
    values = error.sum((0, 2, 3)) / denominator
    return {name: round(float(values[index]), 8) for index, name in enumerate(DYNAMIC_NAMES)}


def _damage_system(state: Tensor, static: Tensor, start: int) -> Tensor:
    result = state.clone(); system = static[:, start : start + 3].sum(1, keepdim=True).clamp(0, 1); result[:, 0:1] *= 1 - .78 * system; result[:, 1:2] *= 1 - .52 * system; result[:, 7:8] = torch.maximum(result[:, 7:8], .9 * system); result[:, 11:12] = (result[:, 0:1] > .025).float() * static[:, :1]
    return result


@torch.no_grad()
def evaluate(output: Path, *, device_name: str = "cuda", rollout_steps: int = 32) -> dict[str, Any]:
    output = Path(output).resolve(); require_disk_floor(output, floor_gb=100, planned_bytes=128 * 1024**2)
    if rollout_steps != 32: raise ValueError("Cellular NCA authoritative rollout length drifted.")
    device = torch.device(device_name)
    if device.type == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or not torch.cuda.is_available(): raise RuntimeError("Deterministic CUDA evaluation is unavailable.")
        torch.use_deterministic_algorithms(True)
    model, checkpoint, contract = load_final_checkpoint(output); model.to(device).eval(); corpus = load_corpus(output); arrays = corpus["arrays"]
    static = torch.from_numpy(arrays["static"]).to(device); initial = torch.from_numpy(arrays["initial_state"]).to(device); bonds = torch.from_numpy(arrays["live_bonds"]).to(device)
    generator = torch.Generator(device=device).manual_seed(contract["seed"] ^ 0x4556414C); injured, live = make_scenarios(static, initial, bonds, generator)
    neural = injured.clone(); teacher = injured.clone(); metrics_by_step: dict[str, Any] = {}
    for step in range(1, rollout_steps + 1):
        teacher = teacher_step(static, teacher, live, contract["teacher_dt"]); neural = model(static, neural, live)
        if step in (1, 2, 4, 8, 16, 32): metrics_by_step[str(step)] = _mae(neural, teacher, static)
    clean = initial.clone()
    for _ in range(16): clean = model(static, clean, bonds)
    clean_drift = _mae(clean, initial, static)
    # Causal organ ablations on the same 45 anatomies.
    causal: dict[str, Any] = {}
    for name, start, readout in (("circulation", 28, 1), ("respiration", 31, 4), ("digestion", 34, 3), ("neural", 37, 8)):
        damaged = _damage_system(initial, static, start); control = initial.clone()
        for _ in range(16): damaged = model(static, damaged, bonds); control = model(static, control, bonds)
        body = static[:, :1]; damaged_value = (damaged[:, readout : readout + 1] * body).sum() / body.sum(); control_value = (control[:, readout : readout + 1] * body).sum() / body.sum()
        causal[name] = {"readout": DYNAMIC_NAMES[readout], "control": round(float(control_value), 8), "damaged": round(float(damaged_value), 8), "relative_change": round(float((damaged_value - control_value) / control_value.clamp_min(1e-6)), 8)}
    outside = 1 - static[:, :1]; surface_outside = float((neural[:, 9:10] * outside).sum() / outside.sum())
    internal_escape = float((neural[:, :9] * outside).abs().max())
    representatives = [0, 9, 18, 27, 36]; columns = 3; tile = 192; header = 64; label = 28
    canvas = Image.new("RGB", (columns * tile, header + 5 * (tile + label)), (3, 8, 14)); draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "NEURAL CELLULAR ORGANISM DYNAMICS // 32-STEP INJURY ROLLOUT", fill=(94, 239, 255)); draw.text((12, 31), "wounds + internal fluids + oxygen + energy + neural activity + clot + scar + diffuse surface puddles", fill=(150, 174, 190))
    for column, title in enumerate(("INJURED T0", "REFERENCE T32", "NEURAL T32")): draw.text((column * tile + 8, header - 18), title, fill=(184, 255, 73))
    family_names = ("HUMANOID", "ANIMALIAN", "PLANTLIKE", "ANOMALY", "MACHINE")
    injured_np = injured.detach().cpu().numpy(); teacher_np = teacher.detach().cpu().numpy(); neural_np = neural.detach().cpu().numpy(); static_np = static.detach().cpu().numpy()
    for row, index in enumerate(representatives):
        y = header + row * (tile + label)
        for column, values in enumerate((injured_np, teacher_np, neural_np)): canvas.paste(_render(static_np[index], values[index], row), (column * tile, y))
        draw.text((8, y + tile + 5), family_names[row], fill=(184, 255, 73))
    sheet = _png_bytes(canvas); _atomic_bytes(output / SHEET_NAME, sheet)
    final_mae = metrics_by_step["32"]; gates = {
        "all_values_finite": all(np.isfinite(list(row.values())).all() for row in metrics_by_step.values()),
        "body_state_never_escapes_chassis": internal_escape == 0.0,
        "surface_fluid_diffuses_outside_chassis": surface_outside > 0,
        "health_rollout_mae_below_0_08": final_mae["health"] < .08,
        "fluid_rollout_mae_below_0_08": final_mae["fluid"] < .08,
        "neural_rollout_mae_below_0_10": final_mae["neural_activity"] < .10,
        "organ_ablation_is_causal": all(causal[name]["relative_change"] < -.005 for name in causal),
        "clean_health_drift_below_0_05": clean_drift["health"] < .05,
    }
    telemetry = json.loads((output / TELEMETRY_NAME).read_text(encoding="utf-8"))
    manifest = {
        "format": FORMAT, "status": "ready" if all(gates.values()) else "experimental", "source_sha256": source_sha256(),
        "model": contract["model"], "parameter_count": contract["parameter_count"], "training": {"total_steps": contract["total_steps"], "segment_steps": contract["segment_steps"], "batch_size": contract["batch_size"], "attempt_count": len(telemetry["attempts"]), "retry_count": len(telemetry["attempts"]) - contract["total_steps"] // contract["segment_steps"], "final_loss": checkpoint["history"][-1]["loss"], "first_loss": checkpoint["history"][0]["loss"], "runtime_segments": [row["runtime"] for row in [checkpoint]]},
        "provenance": {"corpus_manifest_sha256": sha256_file(output / CORPUS_MANIFEST_NAME), "training_contract_sha256": sha256_file(output / TRAINING_NAME), "checkpoint_path": checkpoint_name(contract["total_steps"]), "checkpoint_sha256": sha256_file(output / checkpoint_name(contract["total_steps"])), "model_state_sha256": checkpoint["model_state_sha256"], "ema_state_sha256": checkpoint["ema_state_sha256"]},
        "evaluation": {"rollout_steps": rollout_steps, "mae_by_step": metrics_by_step, "clean_drift": clean_drift, "causal_organ_ablation": causal, "surface_fluid_outside_mean": round(surface_outside, 10), "internal_escape_max": internal_escape},
        "visual": {"path": SHEET_NAME, "bytes": len(sheet), "sha256": sha256_file(output / SHEET_NAME), "visually_inspected": False},
        "gates": gates,
        "limitations": ["This model imitates a deterministic physiology teacher; it is not yet trained from observed ecological trajectories.", "The anatomy chassis and severed bond graph remain authoritative inputs in v1.", "Long-horizon behavior beyond the audited 32-step rollout is experimental."],
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest); _atomic_bytes(output / MANIFEST_NAME, canonical_json_bytes(manifest)); return manifest


def _manifest_hash(manifest: dict[str, Any]) -> str:
    import hashlib
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_output(output: Path) -> dict[str, Any]:
    output = Path(output).resolve(); path = output / MANIFEST_NAME; encoded = path.read_bytes(); manifest = json.loads(encoded)
    schema = json.loads((PROJECT_ROOT / "shared/schema/cellular_nca_manifest.schema.json").read_text(encoding="utf-8")); errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda error: list(error.absolute_path))
    if errors: raise ValueError(f"Cellular NCA schema validation failed: {errors[0].message}")
    if encoded != canonical_json_bytes(manifest) or manifest.get("format") != FORMAT or manifest.get("source_sha256") != source_sha256() or manifest.get("manifest_sha256") != _manifest_hash(manifest): raise ValueError("Cellular NCA manifest drifted.")
    if manifest["provenance"]["corpus_manifest_sha256"] != sha256_file(output / CORPUS_MANIFEST_NAME) or manifest["provenance"]["training_contract_sha256"] != sha256_file(output / TRAINING_NAME): raise ValueError("Cellular NCA authority drifted.")
    checkpoint = output / PurePosixPath(manifest["provenance"]["checkpoint_path"])
    if sha256_file(checkpoint) != manifest["provenance"]["checkpoint_sha256"]: raise ValueError("Cellular NCA checkpoint artifact drifted.")
    visual = output / PurePosixPath(manifest["visual"]["path"])
    if visual.stat().st_size != manifest["visual"]["bytes"] or sha256_file(visual) != manifest["visual"]["sha256"]: raise ValueError("Cellular NCA visual artifact drifted.")
    load_final_checkpoint(output); load_corpus(output)
    return {"passed": all(manifest["gates"].values()), "status": manifest["status"], "manifest_sha256": manifest["manifest_sha256"], "checkpoint_sha256": manifest["provenance"]["checkpoint_sha256"], "visual_sha256": manifest["visual"]["sha256"], "parameter_count": manifest["parameter_count"]}
