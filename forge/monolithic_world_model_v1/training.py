from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..neural_world_state_v1.contract import WorldStateModelConfig
from ..neural_world_state_v1.data import build_corpus
from ..neural_world_state_v1.model import build_model as build_world_codec
from ..recurrent_world_context_v1.contract import ContextModelConfig
from ..recurrent_world_context_v1.model import build_model as build_adapter
from ..recurrent_world_student_v5.model import PerceptionRecurrentWorldStudent
from ..safety import require_disk_floor
from ..world_latent_dit.contract import ModelConfig as RecurrentConfig
from .contract import (
    CHECKPOINT_FORMAT, CONTEXT_ADAPTER, CONTEXT_ADAPTER_SHA256, DEFAULT_OUTPUT, FORMAT,
    DECODER_SHA256, RECURRENT, RECURRENT_SHA256, WORLD_STATE, WORLD_STATE_SHA256,
    DirectContextConfig, DistillationPlan, canonical, config_dict, file_sha256, source_sha256,
)
from .model import FusedStructuredActionModel, build_encoder


def _load_teachers(device: torch.device):
    for path, expected in ((WORLD_STATE, WORLD_STATE_SHA256), (CONTEXT_ADAPTER, CONTEXT_ADAPTER_SHA256), (RECURRENT, RECURRENT_SHA256)):
        if file_sha256(path) != expected:
            raise ValueError(f"Monolithic teacher drifted: {path.name}")
    world_payload = torch.load(WORLD_STATE, map_location="cpu", weights_only=True)
    codec = build_world_codec(WorldStateModelConfig(**world_payload["model_config"]))
    codec.load_state_dict(world_payload["state"], strict=True)
    adapter_payload = torch.load(CONTEXT_ADAPTER, map_location="cpu", weights_only=True)
    adapter = build_adapter(ContextModelConfig(**adapter_payload["model_config"]))
    adapter.load_state_dict(adapter_payload["state"], strict=True)
    return codec.to(device).eval(), adapter.to(device).eval()


@torch.inference_mode()
def _teacher_targets(corpus, codec, adapter, device: torch.device) -> np.ndarray:
    rows = []
    for start in range(0, len(corpus.terrain), 128):
        end = start + 128
        terrain = torch.from_numpy(corpus.terrain[start:end]).to(device).long()
        city = torch.from_numpy(corpus.city[start:end]).to(device).long()
        continuous = torch.from_numpy(corpus.continuous[start:end].astype(np.float32)).to(device)
        condition = torch.from_numpy(corpus.condition[start:end]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            spatial, global_state, _, _ = codec.encode(terrain, city, continuous, condition, sample=False)
            target = adapter(torch.cat((global_state.float(), spatial.float().mean((2, 3))), 1))
        rows.append(target.float().cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def _atomic_torch(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(output: Path = DEFAULT_OUTPUT, *, config: DirectContextConfig = DirectContextConfig(),
          plan: DistillationPlan = DistillationPlan(), device: str = "cuda") -> dict[str, object]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1 << 30)
    target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    if target.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.48, 0)
        torch.cuda.reset_peak_memory_stats(target)
    torch.set_num_threads(4)
    torch.manual_seed(plan.seed)
    rng = np.random.default_rng(plan.seed)
    started = time.perf_counter()
    corpus = build_corpus(plan.corpus_size, seed=plan.seed ^ 0x434F52505553)
    codec, adapter = _load_teachers(target)
    targets = _teacher_targets(corpus, codec, adapter, target)
    del codec, adapter
    if target.type == "cuda":
        torch.cuda.empty_cache()
    split = plan.corpus_size * 7 // 8
    model = build_encoder(config).to(target)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=1e-3, fused=target.type == "cuda")
    history = []
    for step in range(1, plan.steps + 1):
        index_np = rng.integers(0, split, plan.batch_size)
        index = torch.from_numpy(index_np).to(target)
        terrain = torch.from_numpy(corpus.terrain[index_np]).to(target).long()
        city = torch.from_numpy(corpus.city[index_np]).to(target).long()
        continuous = torch.from_numpy(corpus.continuous[index_np].astype(np.float32)).to(target)
        condition = torch.from_numpy(corpus.condition[index_np]).to(target)
        desired = torch.from_numpy(targets[index_np]).to(target)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"):
            prediction = model(terrain, city, continuous, condition)
            loss = F.smooth_l1_loss(prediction.float(), desired)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Monolithic context distillation became non-finite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        with torch.no_grad():
            torch._foreach_mul_(list(ema.parameters()), plan.ema_decay)
            torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - plan.ema_decay)
        if step == 1 or step % 100 == 0 or step == plan.steps:
            history.append({"step": step, "loss": float(loss)})
    model = ema.eval()
    predicted = []
    began = time.perf_counter()
    with torch.inference_mode():
        for start in range(split, plan.corpus_size, 128):
            end = start + 128
            with torch.autocast(target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"):
                value = model(
                    torch.from_numpy(corpus.terrain[start:end]).to(target).long(),
                    torch.from_numpy(corpus.city[start:end]).to(target).long(),
                    torch.from_numpy(corpus.continuous[start:end].astype(np.float32)).to(target),
                    torch.from_numpy(corpus.condition[start:end]).to(target),
                )
            predicted.append(value.float().cpu())
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    inference_seconds = time.perf_counter() - began
    prediction = torch.cat(predicted)
    desired = torch.from_numpy(targets[split:])
    mae = float(F.l1_loss(prediction, desired))
    cosine = float(F.cosine_similarity(prediction, desired, dim=1).mean())

    recurrent_payload = torch.load(RECURRENT, map_location="cpu", weights_only=True)
    recurrent_config = RecurrentConfig(**recurrent_payload["model_config"])
    fused = FusedStructuredActionModel(config, recurrent_config)
    fused.context.load_state_dict(model.cpu().state_dict(), strict=True)
    fused.recurrent.load_state_dict(recurrent_payload["state"], strict=True)
    state = {name: value.detach().cpu() for name, value in fused.state_dict().items()}
    gates = {
        "context_mae": mae <= .025,
        "context_cosine": cosine >= .97,
        "target_30fps_context": (len(desired) / inference_seconds) >= 30,
        "single_action_model_plus_vae": True,
    }
    report = {
        "format": FORMAT,
        "status": "monolithic_foundation_ready" if all(gates.values()) else "quality_failed",
        "source_sha256": source_sha256(),
        "teacher": {"world_state_sha256": WORLD_STATE_SHA256, "context_adapter_sha256": CONTEXT_ADAPTER_SHA256, "recurrent_sha256": RECURRENT_SHA256, "decoder_sha256": DECODER_SHA256},
        "corpus_sha256": corpus.sha256,
        "model_config": config_dict(config),
        "recurrent_config": recurrent_payload["model_config"],
        "training_plan": config_dict(plan),
        "parameters": sum(value.numel() for value in state.values()),
        "context_parameters": sum(value.numel() for value in model.parameters()),
        "metrics": {"context_mae": mae, "context_cosine": cosine, "context_samples_per_second": len(desired) / inference_seconds},
        "runtime": {"elapsed_seconds": time.perf_counter() - started, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target)) if target.type == "cuda" else 0},
        "gates": gates,
        "deployment_shape": {"main_model": "fused_structured_action_model", "rasterizer": "continuous_world_frame_vae", "external_world_codec_required": False, "external_context_adapter_required": False},
        "history": history,
        "limitations": ["Damage conservation and topology safety remain supervised by the deterministic scaffold.", "The fused action model still inherits the six-world recurrent training distribution."],
    }
    payload = {"format": CHECKPOINT_FORMAT, "source_sha256": report["source_sha256"], "status": report["status"], "model_config": report["model_config"], "recurrent_config": report["recurrent_config"], "state": state, "normalization": recurrent_payload["normalization"], "inference": recurrent_payload["inference"], "report": report}
    output.mkdir(parents=True)
    _atomic_torch(output / "runtime.pt", payload)
    report["checkpoint_sha256"] = file_sha256(output / "runtime.pt")
    (output / "report.json").write_bytes(canonical(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--corpus-size", type=int, default=4096)
    args = parser.parse_args(argv)
    plan = DistillationPlan(steps=args.steps, corpus_size=args.corpus_size)
    print(json.dumps(train(args.output, plan=plan, device=args.device), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
