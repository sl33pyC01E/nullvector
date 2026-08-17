from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..neural_city_layout_v1.contract import CLASSES
from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, FORMAT, GrowthModelConfig, GrowthTrainingConfig, canonical_json_bytes, source_sha256
from .model import build_model
from .teacher import build_growth_corpus, extract_local_patch, local_condition_vector


def train(output: Path, *, model_config: GrowthModelConfig = GrowthModelConfig(), training_config: GrowthTrainingConfig = GrowthTrainingConfig(), device: str = "cuda") -> dict[str, object]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3); target_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    examples = build_growth_corpus(training_config.corpus_size, seed=training_config.seed); current = torch.from_numpy(np.stack([extract_local_patch(item.current, item.condition.site) for item in examples])).long(); target = torch.from_numpy(np.stack([extract_local_patch(item.target, item.condition.site) for item in examples])).long(); conditions = torch.from_numpy(np.stack([local_condition_vector(item.condition) for item in examples])).float(); changed = current != target
    digest = hashlib.sha256(); [digest.update(bytes.fromhex(item.identity)) for item in examples]; corpus_sha = digest.hexdigest(); split = int(len(examples) * .875)
    model = build_model(model_config).to(target_device); optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=.01, fused=target_device.type == "cuda"); ema = {name: value.detach().float().cpu().clone() for name, value in model.state_dict().items()}; generator = torch.Generator(device="cpu").manual_seed(training_config.seed ^ 0x545241494E); history = []; started = time.perf_counter()
    if target_device.type == "cuda": torch.cuda.reset_peak_memory_stats(target_device)
    model.train()
    for step in range(1, training_config.steps + 1):
        index = torch.randint(split, (training_config.batch_size,), generator=generator); x = current[index].clone(); y = target[index].to(target_device); c = conditions[index].to(target_device); delta = changed[index].to(target_device)
        # Free-running city states are imperfect. Structured dropout and sparse
        # class corruption teach each growth tick to recover its local patch
        # while adding the requested construction, instead of compounding drift.
        corrupt_batch = torch.rand((training_config.batch_size, 1, 1), generator=generator) < .65
        corrupt_rate = torch.rand((training_config.batch_size, 1, 1), generator=generator) * .10 + .02
        corrupt = (torch.rand(x.shape, generator=generator) < corrupt_rate) & corrupt_batch
        replacement = torch.randint(len(CLASSES), x.shape, generator=generator)
        erase = torch.rand(x.shape, generator=generator) < .65
        x[corrupt] = torch.where(erase[corrupt], torch.zeros_like(x[corrupt]), replacement[corrupt]); x = x.to(target_device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(target_device.type, dtype=torch.bfloat16, enabled=target_device.type == "cuda"):
            logits = model(x, c); loss_map = F.cross_entropy(logits.float(), y, reduction="none"); weights = torch.where(delta, 12.0, torch.where(y != 0, 2.0, .35)); loss = (loss_map * weights).sum() / weights.sum().clamp_min(1)
        if not bool(torch.isfinite(loss)): raise FloatingPointError("Growth loss is non-finite.")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items(): ema[name].mul_(training_config.ema_decay).add_(value.detach().float().cpu(), alpha=1 - training_config.ema_decay)
        if step == 1 or step % 100 == 0 or step == training_config.steps: history.append({"step": step, "loss": float(loss.detach().cpu())})
    current_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}; model.load_state_dict(ema); model.eval(); held = slice(split, None); x = current[held].to(target_device); y = target[held].to(target_device); c = conditions[held].to(target_device); delta = changed[held].to(target_device)
    with torch.inference_mode(), torch.autocast(target_device.type, dtype=torch.bfloat16, enabled=target_device.type == "cuda"): prediction = model(x, c).argmax(1)
    true_positive = ((prediction == y) & delta).sum(); change_accuracy = float((true_positive / delta.sum().clamp_min(1)).cpu()); unchanged = ~delta; preservation = float(((prediction == y) & unchanged).sum().float().div(unchanged.sum().clamp_min(1)).cpu()); predicted_change = prediction != x; intersection = (predicted_change & delta).sum(); union = (predicted_change | delta).sum(); change_iou = float((intersection.float() / union.clamp_min(1)).cpu()); noops = ~changed[held].flatten(1).any(1); noop_exact = float((prediction[noops] == y[noops]).all(dim=(1, 2)).float().mean().cpu()) if bool(noops.any()) else 0.0
    report = {"format": FORMAT, "status": "trained_experimental", "source_sha256": source_sha256(), "corpus_sha256": corpus_sha, "corpus_size": len(examples), "model_config": asdict(model_config), "training_config": asdict(training_config), "parameters": sum(value.numel() for value in model.parameters()), "history": history, "heldout_change_accuracy": change_accuracy, "heldout_change_iou": change_iou, "heldout_preservation_accuracy": preservation, "heldout_noop_exact_rate": noop_exact, "elapsed_seconds": time.perf_counter() - started, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target_device)) if target_device.type == "cuda" else 0}
    payload = {"format": CHECKPOINT_FORMAT, "source_sha256": report["source_sha256"], "corpus_sha256": corpus_sha, "model_config": asdict(model_config), "training_config": asdict(training_config), "model_state": current_state, "ema_state": ema, "optimizer_state": optimizer.state_dict(), "report": report}; output.mkdir(parents=True, exist_ok=False); torch.save(payload, output / "checkpoint.pt"); (output / "report.json").write_bytes(canonical_json_bytes(report)); return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--steps", type=int, default=2400); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--corpus-size", type=int, default=8192); args = parser.parse_args(argv); print(json.dumps(train(args.output, training_config=GrowthTrainingConfig(steps=args.steps, batch_size=args.batch_size, corpus_size=args.corpus_size), device=args.device), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
