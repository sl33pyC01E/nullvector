from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch

from ..cellular_nca.contract import BOND_CHANNELS, DYNAMIC_CHANNELS, STATIC_CHANNELS, CellularNCAConfig
from ..cellular_nca.model import OrganismCellularAutomaton
from ..cellular_nca.teacher import cellular_loss, make_scenarios
from ..cellular_nca_causal.curriculum import PRE_ROLL_CHOICES, SYSTEMS, causal_contrast_loss, make_targeted_pairs
from ..safety import require_disk_floor
from .contract import (
    CHECKPOINT_FORMAT, CORPUS, CORPUS_SHA256, DEFAULT_OUTPUT, FORMAT, TEACHER, TEACHER_SHA256,
    MobileCellNCAConfig, MobileCellNCAPlan, canonical, config_dict, file_sha256,
    source_sha256, tensor_state_sha256,
)
from .evaluation import evaluate_candidate
from .model import MobileCellNCA


CHECKPOINT_NAME = "runtime.pt"
REPORT_NAME = "report.json"
ONNX_NAME = "mobile_cell_nca_fp32.onnx"
SHEET_NAME = "mobile_cell_nca_rollout.png"


def _load() -> tuple[dict[str, np.ndarray], OrganismCellularAutomaton]:
    if file_sha256(CORPUS) != CORPUS_SHA256 or file_sha256(TEACHER) != TEACHER_SHA256:
        raise ValueError("Mobile NCA authority drifted.")
    with np.load(CORPUS, allow_pickle=False) as archive:
        if set(archive.files) != {"static", "initial_state", "live_bonds", "family_id", "sample_id"}:
            raise ValueError("Mobile NCA corpus member contract drifted.")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    count = len(arrays["family_id"])
    if arrays["static"].shape != (count, STATIC_CHANNELS, 48, 48) or arrays["static"].dtype != np.float32:
        raise ValueError("Mobile NCA static corpus drifted.")
    if arrays["initial_state"].shape != (count, DYNAMIC_CHANNELS, 48, 48) or arrays["initial_state"].dtype != np.float32:
        raise ValueError("Mobile NCA state corpus drifted.")
    if arrays["live_bonds"].shape != (count, BOND_CHANNELS, 48, 48) or arrays["live_bonds"].dtype != np.float32:
        raise ValueError("Mobile NCA bond corpus drifted.")
    if arrays["family_id"].dtype != np.uint8 or sorted(np.unique(arrays["family_id"]).tolist()) != list(range(5)):
        raise ValueError("Mobile NCA family corpus drifted.")
    if len(np.unique(arrays["sample_id"])) != count:
        raise ValueError("Mobile NCA sample identity drifted.")
    payload = torch.load(TEACHER, map_location="cpu", weights_only=True)
    teacher = OrganismCellularAutomaton(CellularNCAConfig(**payload["model"]))
    teacher.load_state_dict(payload["model_state"], strict=True)
    teacher.eval().requires_grad_(False)
    return arrays, teacher


def _split(family: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    heldout = np.asarray([np.flatnonzero(family == value)[-1] for value in range(5)], dtype=np.int64)
    heldout_set = set(heldout.tolist())
    train = np.asarray([index for index in range(len(family)) if index not in heldout_set], dtype=np.int64)
    if len(train) + len(heldout) != len(family) or len(heldout) != 5:
        raise ValueError("Mobile NCA split contract drifted.")
    return train, heldout


def _weighted_score(metrics: dict[str, Any]) -> float:
    failed = sum(not value for value in metrics["gates"].values())
    final = metrics["mae_by_step"]["32"]
    return failed * 100 + metrics["response_by_step"]["32"]["normalized_l1"] + metrics["organ_counterfactual_mae"] + float(np.mean(list(final.values())))


def _benchmark(model: MobileCellNCA, static: torch.Tensor, state: torch.Tensor, bonds: torch.Tensor, device: torch.device) -> dict[str, float | int]:
    with torch.inference_mode():
        for _ in range(12): model(static, state, bonds)
        if device.type == "cuda": torch.cuda.synchronize(device)
        began = time.perf_counter(); ticks = 128
        for _ in range(ticks): model(static, state, bonds)
        if device.type == "cuda": torch.cuda.synchronize(device)
    seconds = time.perf_counter() - began
    return {"ticks": ticks, "ticks_per_second": ticks / seconds, "milliseconds_per_tick": seconds * 1000 / ticks}


def _export_onnx(model: MobileCellNCA, static: torch.Tensor, state: torch.Tensor, bonds: torch.Tensor, destination: Path) -> dict[str, Any]:
    model = copy.deepcopy(model).cpu().eval(); inputs = (static[:1].cpu(), state[:1].cpu(), bonds[:1].cpu())
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    torch.onnx.export(model, inputs, temporary, input_names=("static", "state", "live_bonds"), output_names=("next_state",), opset_version=18, dynamo=False)
    os.replace(temporary, destination)
    import onnxruntime as ort
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    feed = {"static": inputs[0].numpy(), "state": inputs[1].numpy(), "live_bonds": inputs[2].numpy()}
    with torch.inference_mode(): expected = model(*inputs).numpy()
    actual = session.run(None, feed)[0]; max_abs = float(np.max(np.abs(actual - expected)))
    for _ in range(8): session.run(None, feed)
    began = time.perf_counter(); ticks = 64
    for _ in range(ticks): session.run(None, feed)
    seconds = time.perf_counter() - began
    return {"bytes": destination.stat().st_size, "sha256": file_sha256(destination), "max_abs_parity": max_abs, "cpu_ticks_per_second": ticks / seconds, "cpu_milliseconds_per_tick": seconds * 1000 / ticks}


def _train_step(model: MobileCellNCA, teacher: OrganismCellularAutomaton, static: torch.Tensor, initial: torch.Tensor, bonds: torch.Tensor, generator: torch.Generator, rollout_steps: int, step: int) -> tuple[torch.Tensor, dict[str, float]]:
    causal = step % 4 == 0; system_ids: torch.Tensor | None = None
    if causal:
        system_ids = (torch.arange(len(static), device=static.device) + step // 4) % len(SYSTEMS)
        choices = torch.as_tensor(PRE_ROLL_CHOICES, device=static.device)
        pre_rolls = choices[torch.randint(0, len(choices), (len(static),), generator=generator, device=static.device)]
        control, damaged = make_targeted_pairs(static, initial, bonds, system_ids.long(), pre_rolls.long())
        static = torch.cat((static, static), 0); bonds = torch.cat((bonds, bonds), 0); target = torch.cat((control, damaged), 0)
    else:
        target, bonds = make_scenarios(static, initial, bonds, generator)
    predicted = target.clone(); losses: list[torch.Tensor] = []; pieces: dict[str, float] = {}; final_target = target; final_predicted = predicted
    for horizon in range(rollout_steps):
        previous = predicted
        with torch.no_grad(), torch.autocast(static.device.type, dtype=torch.bfloat16, enabled=static.device.type == "cuda"):
            final_target = teacher(static, final_target, bonds).float()
        with torch.autocast(static.device.type, dtype=torch.bfloat16, enabled=static.device.type == "cuda"):
            final_predicted = model(static, predicted, bonds).float(); loss, detail = cellular_loss(final_predicted, final_target, static, previous=previous)
        losses.append(loss * (1 + horizon / rollout_steps)); predicted = final_predicted
        if horizon == rollout_steps - 1: pieces = {name: float(value) for name, value in detail.items() if name in ("reconstruction", "velocity", "edge")}
    total = torch.stack(losses).mean()
    if causal and system_ids is not None:
        batch = len(system_ids)
        contrast, contrast_parts = causal_contrast_loss(final_predicted[:batch], final_predicted[batch:], final_target[:batch], final_target[batch:], static[:batch], system_ids.long())
        total = total + .14 * contrast; pieces.update({name: float(value) for name, value in contrast_parts.items()})
    pieces["causal_batch"] = float(causal)
    return total, pieces


def train(output: Path = DEFAULT_OUTPUT, *, config: MobileCellNCAConfig = MobileCellNCAConfig(), plan: MobileCellNCAPlan = MobileCellNCAPlan(), device: str = "cuda") -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1 << 30)
    target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    if target.type == "cuda": torch.cuda.set_per_process_memory_fraction(.48, 0); torch.cuda.reset_peak_memory_stats(target)
    torch.set_num_threads(4); arrays, teacher = _load(); teacher.to(target).requires_grad_(False); train_ids, heldout_ids = _split(arrays["family_id"])
    static_all = torch.from_numpy(arrays["static"]).to(target); state_all = torch.from_numpy(arrays["initial_state"]).to(target); bond_all = torch.from_numpy(arrays["live_bonds"]).to(target); family_all = torch.from_numpy(arrays["family_id"].astype(np.int64)).to(target)
    torch.manual_seed(plan.seed); generator = torch.Generator(device=target).manual_seed(plan.seed); rng = np.random.default_rng(plan.seed)
    model = MobileCellNCA(config).to(target); ema = copy.deepcopy(model).eval().requires_grad_(False); optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=1e-4, fused=target.type == "cuda")
    history: list[dict[str, Any]] = []; began = time.perf_counter(); model.train()
    for step in range(1, plan.steps + 1):
        ids = torch.from_numpy(rng.choice(train_ids, plan.batch_size, replace=True)).to(target); optimizer.zero_grad(set_to_none=True)
        loss, pieces = _train_step(model, teacher, static_all[ids], state_all[ids], bond_all[ids], generator, plan.rollout_steps, step)
        if not bool(torch.isfinite(loss)): raise FloatingPointError("Mobile cell NCA became non-finite.")
        loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if not bool(torch.isfinite(gradient)): raise FloatingPointError("Mobile cell NCA gradient became non-finite.")
        optimizer.step()
        with torch.no_grad(): torch._foreach_mul_(list(ema.parameters()), plan.ema_decay); torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - plan.ema_decay)
        if step == 1 or step % 100 == 0 or step == plan.steps: history.append({"step": step, "loss": float(loss), "gradient_norm": float(gradient), **pieces})
    heldout = torch.from_numpy(heldout_ids).to(target); evaluation_seed = 0x56414C4E4341
    raw_metrics, raw_sheet = evaluate_candidate(model.eval(), teacher, static_all[heldout], state_all[heldout], bond_all[heldout], family_all[heldout], seed=evaluation_seed)
    ema_metrics, ema_sheet = evaluate_candidate(ema.eval(), teacher, static_all[heldout], state_all[heldout], bond_all[heldout], family_all[heldout], seed=evaluation_seed)
    selected = "raw" if _weighted_score(raw_metrics) <= _weighted_score(ema_metrics) else "ema"; chosen = model if selected == "raw" else ema; metrics = raw_metrics if selected == "raw" else ema_metrics; sheet = raw_sheet if selected == "raw" else ema_sheet
    parameters = chosen.parameter_count; benchmark = _benchmark(chosen, static_all[heldout[:1]], state_all[heldout[:1]], bond_all[heldout[:1]], target); state = {name: value.detach().cpu() for name, value in chosen.state_dict().items()}; state_hash = tensor_state_sha256(state); source_hash = source_sha256()
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(parents=True)
    checkpoint = {"format": CHECKPOINT_FORMAT, "source_sha256": source_hash, "teacher_sha256": TEACHER_SHA256, "corpus_sha256": CORPUS_SHA256, "model_config": config_dict(config), "training_plan": config_dict(plan), "selected": selected, "state_sha256": state_hash, "state": state}
    torch.save(checkpoint, staging / CHECKPOINT_NAME); (staging / SHEET_NAME).write_bytes(sheet); export = _export_onnx(chosen, static_all[heldout[:1]], state_all[heldout[:1]], bond_all[heldout[:1]], staging / ONNX_NAME)
    peak_reserved = int(torch.cuda.max_memory_reserved(target)) if target.type == "cuda" else 0
    artifact_gates = {"under_750k_parameters": parameters < 750_000, "onnx_under_4_mib": export["bytes"] < 4 * 1024**2, "onnx_parity_below_2e_5": export["max_abs_parity"] < 2e-5, "desktop_onnx_cpu_above_30hz": export["cpu_ticks_per_second"] >= 30, "training_peak_below_12_gib": peak_reserved < 12 * 1024**3}
    gates = {**metrics["gates"], **artifact_gates}; status = "mobile_cell_nca_ready" if all(gates.values()) else "quality_failed"; teacher_parameters = sum(value.numel() for value in teacher.parameters())
    report: dict[str, Any] = {"format": FORMAT, "status": status, "source_sha256": source_hash, "teacher_sha256": TEACHER_SHA256, "corpus_sha256": CORPUS_SHA256, "model_config": config_dict(config), "training_plan": config_dict(plan), "parameters": parameters, "teacher_parameters": teacher_parameters, "compression_ratio": parameters / teacher_parameters, "selection": {"selected": selected, "raw_score": _weighted_score(raw_metrics), "ema_score": _weighted_score(ema_metrics)}, "metrics": metrics, "benchmark": benchmark, "export": export, "gates": gates, "history": history, "runtime": {"seconds": time.perf_counter() - began, "peak_reserved_bytes": peak_reserved, "device": str(target)}, "artifacts": {"checkpoint": {"path": CHECKPOINT_NAME, "bytes": (staging / CHECKPOINT_NAME).stat().st_size, "sha256": file_sha256(staging / CHECKPOINT_NAME)}, "onnx": {"path": ONNX_NAME, **export}, "contact_sheet": {"path": SHEET_NAME, "bytes": len(sheet), "sha256": file_sha256(staging / SHEET_NAME)}, "state_sha256": state_hash}, "limitations": ["The student replaces local per-cell physiology, but mutable fracture topology remains an explicit bond input.", "The 10M neural teacher was originally trained against the deterministic physiology scaffold.", "Ecological decisions, locomotion and reproduction are separate neural modules in this foundation stage."]}
    (staging / REPORT_NAME).write_bytes(canonical(report)); os.replace(staging, output); validate(output); return report


def validate(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output).resolve(); encoded = (output / REPORT_NAME).read_bytes(); report = json.loads(encoded)
    if encoded != canonical(report) or report.get("format") != FORMAT or report.get("source_sha256") != source_sha256(): raise ValueError("Mobile NCA report provenance drifted.")
    if report.get("teacher_sha256") != TEACHER_SHA256 or report.get("corpus_sha256") != CORPUS_SHA256: raise ValueError("Mobile NCA authority binding drifted.")
    for name in ("checkpoint", "onnx", "contact_sheet"):
        artifact = report["artifacts"][name]; path = output / artifact["path"]
        if not path.is_file() or path.stat().st_size != artifact["bytes"] or file_sha256(path) != artifact["sha256"]: raise ValueError(f"Mobile NCA {name} artifact drifted.")
    checkpoint = torch.load(output / CHECKPOINT_NAME, map_location="cpu", weights_only=True); required = {"format", "source_sha256", "teacher_sha256", "corpus_sha256", "model_config", "training_plan", "selected", "state_sha256", "state"}
    if set(checkpoint) != required or checkpoint["format"] != CHECKPOINT_FORMAT or checkpoint["source_sha256"] != source_sha256(): raise ValueError("Mobile NCA checkpoint contract drifted.")
    model = MobileCellNCA(MobileCellNCAConfig(**checkpoint["model_config"])); model.load_state_dict(checkpoint["state"], strict=True)
    if tensor_state_sha256(model.state_dict()) != checkpoint["state_sha256"] or checkpoint["state_sha256"] != report["artifacts"]["state_sha256"]: raise ValueError("Mobile NCA model state drifted.")
    passed = all(report["gates"].values()) and report["status"] == "mobile_cell_nca_ready"
    return {"passed": passed, "status": report["status"], "parameters": report["parameters"], "checkpoint_sha256": report["artifacts"]["checkpoint"]["sha256"], "onnx_sha256": report["artifacts"]["onnx"]["sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--steps", type=int, default=2600); parser.add_argument("--device", default="cuda"); parser.add_argument("--validate", action="store_true"); args = parser.parse_args(argv)
    payload = validate(args.output) if args.validate else train(args.output, plan=MobileCellNCAPlan(steps=args.steps), device=args.device); print(json.dumps(payload, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
