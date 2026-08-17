from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F

from ..creature_stage_developmental.development import develop
from ..creature_stage_developmental.genomes import review_genomes
from ..creature_stage_manipulation_v1.articulation import ArticulatedBody
from ..creature_stage_neural_motion.training import _state_sha256
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, FORMAT, ModelConfig, TrainingConfig, source_sha256
from .dataset import PoseCorpus, build_corpus
from .model import NeuralLimbPose
from .runtime import NeuralLimbPoseDriver


def _loss(model: NeuralLimbPose, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(batch["nodes"], batch["context"], batch["mask"])
    active = batch["mask"][:, :, None].expand_as(output.pose)
    position = F.smooth_l1_loss(output.pose[active], batch["target"].float()[active], beta=.03)
    predicted_delta = output.pose[:, 1:] - output.pose[:, :-1]
    target_delta = batch["target"].float()[:, 1:] - batch["target"].float()[:, :-1]
    edge_mask = (batch["mask"][:, 1:] & batch["mask"][:, :-1])[:, :, None].expand_as(predicted_delta)
    edge = F.smooth_l1_loss(predicted_delta[edge_mask], target_delta[edge_mask], beta=.02)
    confidence = (1.0 - output.confidence.float()).square().mean()
    loss = position * 2.0 + edge * .75 + confidence * .02
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("neural limb pose loss became non-finite")
    return loss, {"loss": float(loss.detach()), "position": float(position.detach()), "edge": float(edge.detach())}


@torch.inference_mode()
def _validation(model: NeuralLimbPose, corpus: PoseCorpus, device: torch.device) -> dict[str, float]:
    errors: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    for start in range(0, corpus.samples, 1024):
        indices = torch.arange(start, min(start + 1024, corpus.samples))
        batch = corpus.batch(indices, device); output = model(batch["nodes"], batch["context"], batch["mask"])
        error = torch.linalg.vector_norm(output.pose - batch["target"].float(), dim=-1) * batch["scale"][:, None]
        errors.append(error[batch["mask"]].float().cpu()); confidences.append(output.confidence.float().cpu())
    error = torch.cat(errors); confidence = torch.cat(confidences)
    return {
        "pose_mae_px": float(error.mean()),
        "pose_p95_px": float(torch.quantile(error, .95)),
        "pose_p99_px": float(torch.quantile(error, .99)),
        "confidence_mean": float(confidence.mean()),
    }


@torch.inference_mode()
def _rollout(model: NeuralLimbPose, device: torch.device) -> dict[str, float]:
    driver = NeuralLimbPoseDriver(model, device)
    node_errors: list[float] = []
    maximum_length_error = 0.0
    maximum_velocity = 0.0; maximum_acceleration = 0.0
    cases = 0
    for genome in review_genomes()[1::2]:
        organism = develop(genome)
        for appendage in range(len(organism.genome.appendages)):
            teacher = ArticulatedBody.from_organism(organism)
            neural = ArticulatedBody.from_organism(organism); neural.pose_driver = driver
            root = teacher.root(appendage)
            reach = teacher.geometry(appendage).length
            local_endpoints: list[np.ndarray] = []
            for frame in range(72):
                phase = frame / 72.0
                radius = reach * (.48 + .34 * (0.5 + 0.5 * math.sin(math.tau * phase)))
                angle = -.75 + 1.5 * math.sin(math.tau * phase)
                target = root + np.asarray((math.cos(angle), math.sin(angle)), np.float64) * radius
                teacher.solve(appendage, target, .72, delta=.05, actuation=.92, load=.8)
                neural.solve(appendage, target, .72, delta=.05, actuation=.92, load=.8)
                node_errors.append(float(np.linalg.norm(
                    teacher.chain(appendage) - neural.chain(appendage), axis=1,
                ).mean()))
                local_endpoints.append(neural.endpoint(appendage).copy())
                maximum_length_error = max(maximum_length_error, neural.max_length_error())
            endpoint_array = np.stack(local_endpoints)
            velocity = np.diff(endpoint_array, axis=0)
            acceleration = np.diff(velocity, axis=0)
            maximum_velocity = max(maximum_velocity, float(np.linalg.norm(velocity, axis=1).max()))
            maximum_acceleration = max(maximum_acceleration, float(np.linalg.norm(acceleration, axis=1).max()))
            cases += 1
    return {
        "rollout_cases": float(cases),
        "rollout_node_mae_px": float(np.mean(node_errors)),
        "maximum_bone_length_error_px": float(maximum_length_error),
        "maximum_endpoint_velocity_px": maximum_velocity,
        "maximum_endpoint_acceleration_px": maximum_acceleration,
    }


def _gates(metrics: dict[str, float]) -> dict[str, bool]:
    gates = {
        "inverse_pose_accuracy": metrics["pose_mae_px"] <= .11 and metrics["pose_p95_px"] <= .28,
        "closed_loop_trajectory_accuracy": metrics["rollout_node_mae_px"] <= .30,
        "bone_lengths_exact": metrics["maximum_bone_length_error_px"] < 1e-4,
        # The accepted grounded controller's measured p95 terminal envelope is
        # 2.332 px/frame velocity and 2.330 px/frame^2 acceleration.
        "grounded_motion_floor": metrics["maximum_endpoint_velocity_px"] <= 2.3321 and metrics["maximum_endpoint_acceleration_px"] <= 2.3297,
        "confidence_calibrated": metrics["confidence_mean"] >= .95,
    }
    gates["all_passed"] = all(gates.values())
    return gates


def _quality(metrics: dict[str, float]) -> float:
    return -metrics["pose_mae_px"] - metrics["pose_p95_px"] - metrics["rollout_node_mae_px"]


def train(output: Path, *, updates: int | None = None, device: str = "cuda") -> dict[str, object]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    defaults = TrainingConfig(); config = TrainingConfig(updates=defaults.updates if updates is None else updates)
    target_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    train_corpus = build_corpus(split="train", cases_per_appendage=config.cases_per_appendage)
    validation_corpus = build_corpus(split="validation", cases_per_appendage=max(160, config.cases_per_appendage // 4))
    torch.manual_seed(config.seed); np.random.seed(config.seed & 0xFFFFFFFF)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed); torch.cuda.reset_peak_memory_stats(target_device)
    model_config = ModelConfig(); model = NeuralLimbPose(model_config).to(target_device).train()
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, fused=target_device.type == "cuda")
    generator = torch.Generator().manual_seed(config.seed); history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for update in range(1, config.updates + 1):
        indices = torch.randint(0, train_corpus.samples, (config.batch_size,), generator=generator)
        batch = train_corpus.batch(indices, target_device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(target_device.type, dtype=torch.bfloat16, enabled=target_device.type == "cuda"):
            loss, pieces = _loss(model, batch)
        loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(float(gradient)):
            raise FloatingPointError("neural limb pose gradient became non-finite")
        optimizer.step()
        with torch.no_grad():
            for ema_value, value in zip(ema.parameters(), model.parameters(), strict=True):
                ema_value.lerp_(value, 1 - config.ema_decay)
        if update == 1 or update % 100 == 0 or update == config.updates:
            row = {"update": update, **{name: round(value, 8) for name, value in pieces.items()}, "gradient": round(float(gradient), 8)}
            history.append(row); print(json.dumps(row), flush=True)
    seconds = time.perf_counter() - started
    candidates: dict[str, dict[str, float]] = {}
    for name, candidate in (("raw", model.eval()), ("ema", ema.eval())):
        candidates[name] = {**_validation(candidate, validation_corpus, target_device), **_rollout(candidate, target_device)}
    selected_name = max(candidates, key=lambda name: (int(_gates(candidates[name])["all_passed"]), _quality(candidates[name])))
    selected = model if selected_name == "raw" else ema
    metrics = candidates[selected_name]; gates = _gates(metrics)
    state = {name: value.detach().cpu().clone() for name, value in selected.state_dict().items()}
    report: dict[str, object] = {
        "format": FORMAT, "status": "passed" if gates["all_passed"] else "failed-quality",
        "source_sha256": source_sha256(), "train_corpus_sha256": train_corpus.semantic_sha256,
        "validation_corpus_sha256": validation_corpus.semantic_sha256,
        "model_config": model_config.to_dict(), "training_config": config.to_dict(),
        "parameters": model.parameter_count, "selected_weights": selected_name,
        "candidate_metrics": candidates, "metrics": metrics, "gates": gates, "history": history,
        "runtime": {"device": str(target_device), "seconds": seconds, "updates_per_second": config.updates / seconds,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(target_device)) if target_device.type == "cuda" else 0},
    }
    payload = {
        "format": CHECKPOINT_FORMAT, "source_sha256": report["source_sha256"],
        "model_config": model_config.to_dict(), "model_state": state,
        "model_state_sha256": _state_sha256(state), "report": report,
    }
    stage = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"; stage.mkdir(parents=True)
    checkpoint = stage / "runtime.pt"; torch.save(payload, checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()
    report["checkpoint"] = {"path": checkpoint.name, "bytes": len(checkpoint_bytes), "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(), "model_state_sha256": payload["model_state_sha256"]}
    (stage / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(stage, output)
    return report
