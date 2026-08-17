from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from ..maps.io import file_sha256
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, FORMAT, WorldStateModelConfig, WorldStateTrainingConfig, canonical, source_sha256
from .data import build_corpus
from .model import build_model


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()): digest.update(name.encode() + b"\0" + value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(output: Path, *, config: WorldStateModelConfig = WorldStateModelConfig(), plan: WorldStateTrainingConfig = WorldStateTrainingConfig(), device: str = "cuda") -> dict[str, object]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 << 30); target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu"); corpus = build_corpus(plan.corpus_size, seed=plan.seed); split = int(plan.corpus_size * .875)
    terrain = torch.from_numpy(corpus.terrain).long(); city = torch.from_numpy(corpus.city).long(); continuous = torch.from_numpy(corpus.continuous).float(); condition = torch.from_numpy(corpus.condition).float(); model = build_model(config).to(target); optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=1e-3, fused=target.type == "cuda"); ema = {name: value.detach().float().cpu().clone() for name, value in model.state_dict().items()}; generator = torch.Generator().manual_seed(plan.seed); history = []; started = time.perf_counter()
    city_class_weights = torch.tensor((.2, 1, 1, 1, 3, 8, 12, 10), device=target)
    if target.type == "cuda": torch.cuda.reset_peak_memory_stats(target)
    for step in range(1, plan.steps + 1):
        index = torch.randint(split, (plan.batch_size,), generator=generator); t = terrain[index].to(target); c = city[index].to(target); x = continuous[index].to(target); q = condition[index].to(target); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"):
            result = model(t, c, x, q); terrain_loss = F.cross_entropy(result.terrain.float(), t); city_loss = F.cross_entropy(result.city.float(), c, weight=city_class_weights); continuous_loss = F.l1_loss(result.continuous.float(), x); condition_loss = F.mse_loss(result.condition.float(), q); kl = -.5 * (1 + result.logvar.float() - result.mean.float().square() - result.logvar.float().exp()).mean(); loss = terrain_loss + 1.4 * city_loss + 4 * continuous_loss + condition_loss + plan.kl_weight * kl
        if not bool(torch.isfinite(loss)): raise FloatingPointError("World-state VAE loss is non-finite.")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items(): ema[name].mul_(plan.ema_decay).add_(value.detach().float().cpu(), alpha=1 - plan.ema_decay)
        if step == 1 or step % 100 == 0 or step == plan.steps: history.append({"step": step, "loss": float(loss), "terrain": float(terrain_loss), "city": float(city_loss), "continuous": float(continuous_loss), "condition": float(condition_loss), "kl": float(kl)})
    model.load_state_dict(ema); model.eval(); held = slice(split, None); t = terrain[held].to(target); c = city[held].to(target); x = continuous[held].to(target); q = condition[held].to(target)
    with torch.inference_mode(), torch.autocast(target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"): result = model(t, c, x, q, sample=False)
    terrain_accuracy = float((result.terrain.argmax(1) == t).float().mean()); city_prediction = result.city.argmax(1); city_accuracy = float((city_prediction == c).float().mean()); city_recall = {str(index): float((city_prediction[c == index] == index).float().mean()) for index in range(8)}; minimum_specialized_recall = min(city_recall[str(index)] for index in (4, 5, 6, 7)); foreground_union = (city_prediction != 0) | (c != 0); city_iou = float((((city_prediction != 0) & (c != 0)).sum() / foreground_union.sum().clamp_min(1))); continuous_mae = float(F.l1_loss(result.continuous.float(), x)); condition_mae = float(F.l1_loss(result.condition.float(), q)); latent_std = float(result.mean.float().std()); gates = {"terrain_accuracy": terrain_accuracy >= .98, "city_accuracy": city_accuracy >= .96, "city_foreground_iou": city_iou >= .82, "minimum_specialized_city_recall": minimum_specialized_recall >= .72, "continuous_mae": continuous_mae <= .04, "condition_mae": condition_mae <= .05, "latent_active": latent_std >= .05}; status = "ready" if all(gates.values()) else "quality_failed"
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}; report = {"format": FORMAT, "status": status, "source_sha256": source_sha256(), "corpus_sha256": corpus.sha256, "model_config": asdict(config), "training_config": asdict(plan), "parameters": sum(value.numel() for value in model.parameters()), "elapsed_seconds": time.perf_counter() - started, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target)) if target.type == "cuda" else 0, "terrain_accuracy": terrain_accuracy, "city_accuracy": city_accuracy, "city_foreground_iou": city_iou, "city_class_recall": city_recall, "minimum_specialized_city_recall": minimum_specialized_recall, "continuous_mae": continuous_mae, "condition_mae": condition_mae, "latent_std": latent_std, "gates": gates, "history": history}
    payload = {"format": CHECKPOINT_FORMAT, "source_sha256": report["source_sha256"], "corpus_sha256": corpus.sha256, "model_config": asdict(config), "state": state, "state_sha256": _state_sha256(state), "report": report}; output.mkdir(parents=True); torch.save(payload, output / "runtime.pt"); report["checkpoint_sha256"] = file_sha256(output / "runtime.pt"); (output / "report.json").write_bytes(canonical(report)); return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--steps", type=int, default=3000); parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--corpus-size", type=int, default=8192); args = parser.parse_args(argv); plan = WorldStateTrainingConfig(steps=args.steps, batch_size=args.batch_size, corpus_size=args.corpus_size); print(json.dumps(train(args.output, plan=plan, device=args.device), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
