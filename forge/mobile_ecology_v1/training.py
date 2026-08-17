from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
import torch.nn.functional as F

from ..nature_behavior_nn.corpus import BehaviorCorpus, load_corpus
from ..nature_sim_v2.contract import INTENTS
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, DEFAULT_CORPUS, DEFAULT_OUTPUT, FORMAT, MobileEcologyConfig, MobileEcologyPlan, canonical, config_dict, file_sha256, source_sha256
from .model import MobileEcologyGraph, MobileEcologyPolicy


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"nullvector-mobile-ecology-state-v1\0")
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy(); digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0" + np.asarray(value.shape, dtype="<i8").tobytes()); digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _tensors(corpus: BehaviorCorpus, device: torch.device) -> dict[str, torch.Tensor]:
    return {"self": torch.from_numpy(corpus.self_features).to(device), "resource": torch.from_numpy(corpus.resource).to(device), "neighbor": torch.from_numpy(corpus.neighbor).to(device), "mask": torch.from_numpy(corpus.neighbor_mask).to(device), "intent": torch.from_numpy(corpus.intent.astype(np.int64)).to(device), "direction": torch.from_numpy(corpus.direction).to(device), "world": torch.from_numpy(corpus.world_id.astype(np.int64)).to(device)}


@torch.inference_mode()
def _evaluate(model: MobileEcologyPolicy, tensors: dict[str, torch.Tensor], ids: torch.Tensor) -> dict[str, Any]:
    model.eval(); correct = 0; count = 0; direction_dot = 0.; direction_count = 0; direction_error = 0.; confusion = np.zeros((len(INTENTS), len(INTENTS)), dtype=np.int64)
    for start in range(0, len(ids), 2048):
        selected = ids[start:start + 2048]; value = model(tensors["self"][selected], tensors["resource"][selected], tensors["neighbor"][selected], tensors["mask"][selected]); predicted = value.intent_logits.argmax(1); target = tensors["intent"][selected]
        correct += int((predicted == target).sum()); count += len(selected); direction = value.direction; truth = tensors["direction"][selected]; active = truth.norm(dim=1) > .1
        direction_dot += float(F.cosine_similarity(direction[active], truth[active]).sum()); direction_count += int(active.sum()); direction_error += float((direction - truth).abs().sum())
        np.add.at(confusion, (target.cpu().numpy(), predicted.cpu().numpy()), 1)
    per_intent = {}; recalls = []
    for index, name in enumerate(INTENTS):
        total = int(confusion[index].sum()); recall = None if total == 0 else float(confusion[index, index] / total); per_intent[name] = {"count": total, "recall": recall, "predicted": int(confusion[:, index].sum())}
        if recall is not None: recalls.append(recall)
    rare = [per_intent[name]["recall"] for name in ("forage", "mate", "photosynthesize") if per_intent[name]["recall"] is not None]
    return {"intent_accuracy": correct / max(1, count), "macro_recall": float(np.mean(recalls)), "rare_recall": float(np.mean(rare)), "direction_cosine": direction_dot / max(1, direction_count), "direction_mae": direction_error / max(1, count * 2), "per_intent": per_intent, "confusion": confusion.tolist()}


def _score(metrics: dict[str, Any]) -> float: return (1 - metrics["intent_accuracy"]) + .35 * (1 - metrics["macro_recall"]) + .35 * (1 - metrics["rare_recall"]) + .3 * (1 - metrics["direction_cosine"]) + metrics["direction_mae"]


def train(output: Path = DEFAULT_OUTPUT, *, corpus_path: Path = DEFAULT_CORPUS, config: MobileEcologyConfig = MobileEcologyConfig(), plan: MobileEcologyPlan = MobileEcologyPlan(), device: str = "cuda") -> dict[str, Any]:
    output = Path(output).resolve(); corpus_path = Path(corpus_path).resolve()
    if output.exists(): raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=512 << 20); target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    if target.type == "cuda": torch.cuda.set_per_process_memory_fraction(.35, 0); torch.cuda.reset_peak_memory_stats(target)
    corpus = load_corpus(corpus_path); tensors = _tensors(corpus, target); worlds = torch.unique(tensors["world"]); held_worlds = worlds[-2:]; validation = torch.where(torch.isin(tensors["world"], held_worlds))[0]; training = torch.where(~torch.isin(tensors["world"], held_worlds))[0]
    counts = torch.bincount(tensors["intent"][training], minlength=len(INTENTS)).float(); weights = torch.where(counts > 0, counts.sum().sqrt() / counts.clamp_min(1).sqrt(), torch.zeros_like(counts)); weights = (weights / weights[counts > 0].mean()).clamp(.35, 6)
    torch.manual_seed(plan.seed); generator = torch.Generator(device=target).manual_seed(plan.seed); model = MobileEcologyPolicy(config).to(target); ema = copy.deepcopy(model).eval().requires_grad_(False); optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=1e-3, fused=target.type == "cuda"); history = []; began = time.perf_counter()
    model.train()
    for step in range(1, plan.steps + 1):
        chosen = training[torch.randint(len(training), (plan.batch_size,), generator=generator, device=target)]; optimizer.zero_grad(set_to_none=True); value = model(tensors["self"][chosen], tensors["resource"][chosen], tensors["neighbor"][chosen], tensors["mask"][chosen]); truth = tensors["direction"][chosen]
        intent_loss = F.cross_entropy(value.intent_logits, tensors["intent"][chosen], weight=weights, label_smoothing=.015); direction_loss = F.smooth_l1_loss(value.direction, truth); angular = 1 - F.cosine_similarity(value.direction, truth).mean(); urgency_target = truth.norm(dim=1).clamp(0, 1); urgency_loss = F.binary_cross_entropy_with_logits(value.urgency, urgency_target); loss = intent_loss + 4.0 * direction_loss + .8 * angular + .15 * urgency_loss
        if not bool(torch.isfinite(loss)): raise FloatingPointError("mobile ecology became non-finite")
        loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        with torch.no_grad(): torch._foreach_mul_(list(ema.parameters()), plan.ema_decay); torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - plan.ema_decay)
        if step == 1 or step % 100 == 0 or step == plan.steps: history.append({"step": step, "loss": float(loss), "intent": float(intent_loss), "direction": float(direction_loss), "angular": float(angular), "urgency": float(urgency_loss), "gradient": float(gradient)})
    raw_metrics = _evaluate(model, tensors, validation); ema_metrics = _evaluate(ema, tensors, validation); selected_name = "raw" if _score(raw_metrics) <= _score(ema_metrics) else "ema"; selected = model if selected_name == "raw" else ema; metrics = raw_metrics if selected_name == "raw" else ema_metrics; state = {name: value.detach().cpu() for name, value in selected.state_dict().items()}; staging = output.with_name(f".{output.name}.tmp-{os.getpid()}"); staging.mkdir(parents=True)
    try:
        checkpoint = {"format": CHECKPOINT_FORMAT, "source_sha256": source_sha256(), "corpus_sha256": corpus.semantic_sha256, "config": config_dict(config), "plan": config_dict(plan), "selected": selected_name, "state_sha256": _state_hash(state), "state": state}; torch.save(checkpoint, staging / "runtime.pt")
        sample = validation[:8]; graph = MobileEcologyGraph(copy.deepcopy(selected).cpu().eval()); inputs = (tensors["self"][sample].cpu(), tensors["resource"][sample].cpu(), tensors["neighbor"][sample].cpu(), tensors["mask"][sample].float().cpu()); onnx_path = staging / "mobile_ecology_fp32.onnx"
        torch.onnx.export(graph, inputs, onnx_path, input_names=("self_features", "resource", "neighbor", "neighbor_mask"), output_names=("intent_logits", "direction", "urgency"), dynamic_axes={"self_features":{0:"batch"},"resource":{0:"batch"},"neighbor":{0:"batch"},"neighbor_mask":{0:"batch"},"intent_logits":{0:"batch"},"direction":{0:"batch"},"urgency":{0:"batch"}}, opset_version=18, dynamo=False)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]); feed = {name: value.numpy() for name, value in zip(("self_features", "resource", "neighbor", "neighbor_mask"), inputs)}
        with torch.inference_mode(): expected = graph(*inputs); expected_values = (expected[0].numpy(), expected[1].numpy(), expected[2].numpy())
        actual = session.run(None, feed); parity = max(float(np.max(np.abs(a - b))) for a, b in zip(actual, expected_values))
        for _ in range(12): session.run(None, feed)
        bench_began = time.perf_counter(); runs = 256
        for _ in range(runs): session.run(None, feed)
        milliseconds = (time.perf_counter() - bench_began) * 1000 / runs
        gates = {"intent_accuracy_above_0_90": metrics["intent_accuracy"] >= .90, "macro_recall_above_0_78": metrics["macro_recall"] >= .78, "rare_recall_above_0_55": metrics["rare_recall"] >= .55, "direction_cosine_above_0_72": metrics["direction_cosine"] >= .72, "direction_mae_below_0_22": metrics["direction_mae"] <= .22, "under_250k_parameters": selected.parameter_count < 250_000, "onnx_under_2_mib": onnx_path.stat().st_size < 2 * 1024**2, "onnx_parity_below_2e_5": parity < 2e-5, "desktop_cpu_under_2ms": milliseconds < 2}
        report = {"format": FORMAT, "status": "mobile_neural_ecology_ready" if all(gates.values()) else "quality_failed", "source_sha256": source_sha256(), "corpus_sha256": corpus.semantic_sha256, "corpus_file_sha256": file_sha256(corpus_path), "config": config_dict(config), "plan": config_dict(plan), "parameters": selected.parameter_count, "selected": selected_name, "metrics": metrics, "raw_metrics": raw_metrics, "ema_metrics": ema_metrics, "history": history, "onnx": {"path": onnx_path.name, "bytes": onnx_path.stat().st_size, "sha256": file_sha256(onnx_path), "max_abs_parity": parity, "desktop_cpu_milliseconds": milliseconds}, "checkpoint": {"path": "runtime.pt", "bytes": (staging / "runtime.pt").stat().st_size, "sha256": file_sha256(staging / "runtime.pt"), "state_sha256": _state_hash(state)}, "runtime": {"device": str(target), "seconds": time.perf_counter() - began, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target)) if target.type == "cuda" else 0}, "gates": gates, "limitations": ["The compact policy replaces organism intent and steering, not collision or material physics.", "Rare intents absent from the current scaffold corpus cannot be learned until their curriculum exists."]}; (staging / "report.json").write_bytes(canonical(report)); os.replace(staging, output)
    finally:
        if staging.exists(): shutil.rmtree(staging)
    return validate(output)


def validate(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output).resolve(); encoded = (output / "report.json").read_bytes(); report = json.loads(encoded)
    if encoded != canonical(report) or report.get("format") != FORMAT or report.get("source_sha256") != source_sha256(): raise ValueError("mobile ecology report drifted")
    for name in ("onnx", "checkpoint"):
        row = report[name]; path = output / row["path"]
        if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]: raise ValueError(f"mobile ecology {name} drifted")
    checkpoint = torch.load(output / report["checkpoint"]["path"], map_location="cpu", weights_only=True); model = MobileEcologyPolicy(MobileEcologyConfig(**checkpoint["config"])); model.load_state_dict(checkpoint["state"], strict=True)
    if _state_hash(model.state_dict()) != checkpoint["state_sha256"] or checkpoint["state_sha256"] != report["checkpoint"]["state_sha256"]: raise ValueError("mobile ecology state drifted")
    return {"passed": report["status"] == "mobile_neural_ecology_ready" and all(report["gates"].values()), "status": report["status"], "parameters": report["parameters"], "metrics": report["metrics"], "onnx": report["onnx"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS); parser.add_argument("--steps", type=int, default=1800); parser.add_argument("--device", default="cuda"); parser.add_argument("--validate", action="store_true"); args = parser.parse_args(argv)
    value = validate(args.output) if args.validate else train(args.output, corpus_path=args.corpus, plan=MobileEcologyPlan(steps=args.steps), device=args.device); print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
