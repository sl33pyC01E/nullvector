from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .contract import CHECKPOINT_FORMAT, FORMAT, ModelConfig, TrainingConfig, config_dict, source_sha256
from .dataset import build_corpus
from .model import NeuralGrasperController


def _state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode() + value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def evaluate(model, corpus, device):
    outputs = []
    for start in range(0, len(corpus.identity), 256):
        ids = torch.arange(start, min(start + 256, len(corpus.identity)))
        batch = corpus.batch(ids, device); outputs.append(model(batch["owner_meta"], batch["owner_mask"], batch["target"], batch["global_state"]))
    appendage = torch.cat([item.appendage_logits.cpu() for item in outputs]); engage = torch.cat([item.engage_logit.cpu() for item in outputs]); reach = torch.cat([item.reach.cpu() for item in outputs]); force = torch.cat([item.force.cpu() for item in outputs]); types = torch.cat([item.type_logits.cpu() for item in outputs]); brace = torch.cat([item.brace.cpu() for item in outputs]); release = torch.cat([item.release_logit.cpu() for item in outputs]); throw = torch.cat([item.throw_impulse.cpu() for item in outputs])
    active = corpus.engage_target > .5
    manipulating = active | (corpus.release_target > .5)
    predicted_active = engage.sigmoid() > .5
    true_positive = int((predicted_active & active).sum()); false_positive = int((predicted_active & ~active).sum()); false_negative = int((~predicted_active & active).sum())
    f1 = 2 * true_positive / max(1, 2 * true_positive + false_positive + false_negative)
    return {
        "samples": len(active), "appendage_accuracy": float((appendage.argmax(-1)[manipulating] == corpus.appendage_target[manipulating]).float().mean()),
        "engage_f1": f1, "reach_mae": float(F.l1_loss(reach[active], corpus.reach_target[active])),
        "force_mae": float(F.l1_loss(force[active], corpus.force_target[active])),
        "target_type_accuracy": float((types.argmax(-1) == corpus.type_target).float().mean()),
        "brace_mae": float(F.l1_loss(brace[active | (corpus.release_target > .5)], corpus.brace_target[active | (corpus.release_target > .5)])),
        "release_accuracy": float(((release.sigmoid() > .5) == (corpus.release_target > .5)).float().mean()),
        "throw_mae": float(F.l1_loss(throw[corpus.release_target > .5], corpus.throw_target[corpus.release_target > .5])),
    }


def _gates(metrics):
    gates = {
        "appendage_accuracy": metrics["appendage_accuracy"] >= .94,
        "engage_f1": metrics["engage_f1"] >= .96,
        "reach_mae": metrics["reach_mae"] <= .06,
        "force_mae": metrics["force_mae"] <= .07,
        "target_type_accuracy": metrics["target_type_accuracy"] >= .98,
        "brace_mae": metrics["brace_mae"] <= .07,
        "release_accuracy": metrics["release_accuracy"] >= .98,
        "throw_mae": metrics["throw_mae"] <= .06,
    }
    gates["all_passed"] = all(gates.values())
    return gates


def _quality(metrics):
    return (
        metrics["appendage_accuracy"] + metrics["engage_f1"] + metrics["target_type_accuracy"]
        + metrics["release_accuracy"] - metrics["reach_mae"] - metrics["force_mae"]
        - metrics["brace_mae"] - metrics["throw_mae"]
    )


def train(output: Path, *, model_config=ModelConfig(), training=TrainingConfig(), device="cuda"):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    target = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(training.seed); np.random.seed(training.seed & 0xffffffff); torch.set_num_threads(min(12, os.cpu_count() or 1))
    train_corpus = build_corpus(split="train"); validation_corpus = build_corpus(split="validation")
    model = NeuralGrasperController(model_config).to(target); ema = copy.deepcopy(model).eval(); optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=1e-4, fused=target.type == "cuda")
    generator = torch.Generator().manual_seed(training.seed); history = []
    for step in range(1, training.steps + 1):
        ids = torch.randint(0, len(train_corpus.identity), (training.batch_size,), generator=generator); batch = train_corpus.batch(ids, target)
        optimizer.zero_grad(set_to_none=True); result = model(batch["owner_meta"], batch["owner_mask"], batch["target"], batch["global_state"]); active = batch["engage_target"] > .5; releasing = batch["release_target"] > .5; manipulating = active | releasing
        select = F.cross_entropy(result.appendage_logits[manipulating], batch["appendage_target"][manipulating]); engage = F.binary_cross_entropy_with_logits(result.engage_logit, batch["engage_target"])
        reach = F.smooth_l1_loss(result.reach[active], batch["reach_target"][active]); force = F.smooth_l1_loss(result.force[active], batch["force_target"][active]); types = F.cross_entropy(result.type_logits, batch["type_target"]); brace = F.smooth_l1_loss(result.brace[active], batch["brace_target"][active])
        release = F.binary_cross_entropy_with_logits(result.release_logit, batch["release_target"]); throw = F.smooth_l1_loss(result.throw_impulse[releasing], batch["throw_target"][releasing])
        loss = 1.5 * select + 1.4 * engage + 5.0 * reach + force + .5 * types + brace + 1.4 * release + 3.0 * throw
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        with torch.no_grad():
            for ema_value, value in zip(ema.parameters(), model.parameters()): ema_value.lerp_(value, 1 - training.ema_decay)
        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": round(float(loss), 7), "select": round(float(select), 7), "engage": round(float(engage), 7), "reach": round(float(reach), 7)}; history.append(row); print(json.dumps(row), flush=True)
    candidates = {"raw": evaluate(model.eval(), validation_corpus, target), "ema": evaluate(ema.eval(), validation_corpus, target)}
    selected_name = max(candidates, key=lambda name: (int(_gates(candidates[name])["all_passed"]), _quality(candidates[name])))
    selected = model if selected_name == "raw" else ema; metrics = candidates[selected_name]; gates = _gates(metrics)
    state = {name: value.detach().cpu().to(torch.bfloat16) if value.is_floating_point() else value.detach().cpu() for name, value in selected.state_dict().items()}
    report = {"format": FORMAT, "source_sha256": source_sha256(), "parameters": model.parameter_count, "device": str(target), "train_corpus_sha256": train_corpus.semantic_sha256, "validation_corpus_sha256": validation_corpus.semantic_sha256, "model_config": config_dict(model_config), "training_config": config_dict(training), "selected_weights": selected_name, "candidate_metrics": candidates, "metrics": metrics, "gates": gates, "history": history}
    payload = {"format": CHECKPOINT_FORMAT, "status": "evaluated", "source_sha256": source_sha256(), "model_config": config_dict(model_config), "model_state": state, "model_state_sha256": _state_hash(state), "report": report}
    output.mkdir(parents=True); torch.save(payload, output / "runtime.pt"); (output / "report.json").write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"); return report
