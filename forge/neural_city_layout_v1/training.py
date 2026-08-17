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

from ..safety import require_disk_floor
from .contract import CHECKPOINT_FORMAT, CLASSES, FORMAT, MASK_TOKEN, ModelConfig, TrainingConfig, canonical_json_bytes, source_sha256
from .model import build_model
from .teacher import build_corpus


def _tensor_corpus(config: TrainingConfig):
    examples = build_corpus(config.corpus_size, seed=config.seed)
    targets = torch.from_numpy(np.stack([item.target for item in examples])).long()
    conditions = torch.from_numpy(np.stack([item.condition.vector() for item in examples])).float()
    identities = [item.identity for item in examples]
    digest = hashlib.sha256()
    for identity in identities: digest.update(bytes.fromhex(identity))
    return targets, conditions, identities, digest.hexdigest()


def train(output: Path, *, model_config: ModelConfig = ModelConfig(), training_config: TrainingConfig = TrainingConfig(), device: str = "cuda") -> dict[str, object]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    target_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    targets, conditions, identities, corpus_sha = _tensor_corpus(training_config)
    split = int(len(targets) * .875)
    train_targets, held_targets = targets[:split], targets[split:]
    train_conditions, held_conditions = conditions[:split], conditions[split:]
    counts = torch.bincount(train_targets.flatten(), minlength=len(CLASSES)).float()
    weights = (counts.sum() / counts.clamp_min(1)).pow(.6); weights /= weights.mean()
    model = build_model(model_config).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=.01, fused=target_device.type == "cuda")
    ema = {name: value.detach().float().cpu().clone() for name, value in model.state_dict().items()}
    generator = torch.Generator(device="cpu").manual_seed(training_config.seed ^ 0x545241494E)
    history = []; started = time.perf_counter()
    if target_device.type == "cuda": torch.cuda.reset_peak_memory_stats(target_device)
    model.train()
    for step in range(1, training_config.steps + 1):
        index = torch.randint(len(train_targets), (training_config.batch_size,), generator=generator)
        clean = train_targets[index].to(target_device); cond = train_conditions[index].to(target_device)
        fraction = torch.rand((training_config.batch_size, 1, 1), generator=generator) * .70 + .25
        full_blank = torch.rand((training_config.batch_size, 1, 1), generator=generator) < .35
        mask = (torch.rand(clean.shape, generator=generator) < fraction) | full_blank
        tokens = clean.clone(); tokens[mask.to(target_device)] = MASK_TOKEN
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(target_device.type, dtype=torch.bfloat16, enabled=target_device.type == "cuda"):
            logits = model(tokens, cond)
            loss_map = F.cross_entropy(logits.float(), clean, weight=weights.to(target_device), reduction="none")
            loss = loss_map[mask.to(target_device)].mean()
        if not bool(torch.isfinite(loss)): raise FloatingPointError("City training loss is non-finite.")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items(): ema[name].mul_(training_config.ema_decay).add_(value.detach().float().cpu(), alpha=1 - training_config.ema_decay)
        if step == 1 or step % 100 == 0 or step == training_config.steps:
            history.append({"step": step, "loss": float(loss.detach().cpu())})
    current = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    model.load_state_dict(ema, strict=True); model.eval()
    # Mask 50% of held-out maps. This measures actual learned completion rather
    # than easy all-empty full-grid accuracy.
    held_generator = torch.Generator(device="cpu").manual_seed(training_config.seed ^ 0x4556414C)
    sample_count = min(256, len(held_targets)); clean = held_targets[:sample_count].to(target_device); cond = held_conditions[:sample_count].to(target_device)
    mask = torch.rand(clean.shape, generator=held_generator) < .5; tokens = clean.clone(); tokens[mask.to(target_device)] = MASK_TOKEN
    with torch.inference_mode(), torch.autocast(target_device.type, dtype=torch.bfloat16, enabled=target_device.type == "cuda"):
        prediction = model(tokens, cond).argmax(1)
    masked_accuracy = float((prediction[mask.to(target_device)] == clean[mask.to(target_device)]).float().mean().cpu())
    foreground = mask.to(target_device) & (clean != 0)
    foreground_accuracy = float((prediction[foreground] == clean[foreground]).float().mean().cpu()) if bool(foreground.any()) else 0.0
    predicted_foreground = float((prediction[mask.to(target_device)] != 0).float().mean().cpu())
    parameters = sum(value.numel() for value in model.parameters())
    report = {
        "format": FORMAT,
        "status": "trained_experimental",
        "source_sha256": source_sha256(),
        "corpus_sha256": corpus_sha,
        "corpus_size": len(targets),
        "train_size": split,
        "heldout_size": len(targets) - split,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "parameters": parameters,
        "history": history,
        "heldout_masked_accuracy": masked_accuracy,
        "heldout_foreground_accuracy": foreground_accuracy,
        "heldout_predicted_foreground_fraction": predicted_foreground,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target_device)) if target_device.type == "cuda" else 0,
    }
    payload = {
        "format": CHECKPOINT_FORMAT,
        "source_sha256": report["source_sha256"],
        "corpus_sha256": corpus_sha,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "model_state": current,
        "ema_state": ema,
        "optimizer_state": optimizer.state_dict(),
        "report": report,
    }
    output.mkdir(parents=True, exist_ok=False)
    torch.save(payload, output / "checkpoint.pt")
    (output / "report.json").write_bytes(canonical_json_bytes(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the neural categorical city-layout prior")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--corpus-size", type=int, default=4096)
    args = parser.parse_args(argv)
    report = train(args.output, training_config=TrainingConfig(steps=args.steps, batch_size=args.batch_size, corpus_size=args.corpus_size), device=args.device)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
