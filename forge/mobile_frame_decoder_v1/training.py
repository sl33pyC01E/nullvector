from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from ..safety import require_disk_floor
from ..world_frame_vae.contract import ModelConfig as TeacherConfig
from ..world_frame_vae.model import WorldFrameVAE
from .contract import CHECKPOINT_FORMAT, DEFAULT_OUTPUT, FORMAT, LATENT_SHARDS, TEACHER, TEACHER_SHA256, MobileDecoderConfig, MobileDecoderPlan, canonical, config_dict, file_sha256, source_sha256
from .model import build_model


def _load_latents() -> np.ndarray:
    rows = []
    for path in sorted(LATENT_SHARDS.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive: rows.append(archive["latent"])
    if len(rows) != 6: raise ValueError("Mobile decoder latent corpus drifted.")
    return np.concatenate(rows).astype(np.float32)


def _edge_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    horizontal = F.l1_loss(first[:, :, :, 1:] - first[:, :, :, :-1], second[:, :, :, 1:] - second[:, :, :, :-1])
    vertical = F.l1_loss(first[:, :, 1:, :] - first[:, :, :-1, :], second[:, :, 1:, :] - second[:, :, :-1, :])
    return (horizontal + vertical) * .5


def _contact(path: Path, teacher: torch.Tensor, student: torch.Tensor) -> None:
    count = min(6, len(teacher)); canvas = Image.new("RGB", (512, count * 256 + 36), (5, 8, 12)); draw = ImageDraw.Draw(canvas); draw.text((8, 8), "teacher | mobile student", fill=(80, 235, 215))
    for index in range(count):
        left = Image.fromarray(teacher[index].mul(255).byte().permute(1, 2, 0).cpu().numpy())
        right = Image.fromarray(student[index].mul(255).byte().permute(1, 2, 0).cpu().numpy())
        canvas.paste(left, (0, 36 + index * 256)); canvas.paste(right, (256, 36 + index * 256))
    canvas.save(path, optimize=True)


def train(output: Path = DEFAULT_OUTPUT, *, config: MobileDecoderConfig = MobileDecoderConfig(), plan: MobileDecoderPlan = MobileDecoderPlan(), device: str = "cuda") -> dict[str, object]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    if file_sha256(TEACHER) != TEACHER_SHA256: raise ValueError("Mobile decoder teacher drifted.")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1 << 30); target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    if target.type == "cuda": torch.cuda.set_per_process_memory_fraction(.48, 0); torch.cuda.reset_peak_memory_stats(target)
    torch.set_num_threads(4); torch.manual_seed(plan.seed); rng = np.random.default_rng(plan.seed); latents = _load_latents(); split = len(latents) - 128
    teacher_payload = torch.load(TEACHER, map_location="cpu", weights_only=True); teacher = WorldFrameVAE(TeacherConfig(**teacher_payload["model_config"])); teacher.load_state_dict(teacher_payload["state"], strict=True); teacher.to(target).eval().requires_grad_(False)
    model = build_model(config).to(target); ema = copy.deepcopy(model).eval().requires_grad_(False); optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=1e-4, fused=target.type == "cuda"); history = []; started = time.perf_counter()
    for step in range(1, plan.steps + 1):
        indices = rng.integers(0, split, plan.batch_size); latent = torch.from_numpy(latents[indices]).to(target); optimizer.zero_grad(set_to_none=True)
        with torch.inference_mode(), torch.autocast(target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"): desired = teacher.decode(latent).float()
        with torch.autocast(target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"):
            prediction = model(latent).float()
            horizontal = torch.nn.functional.pad((desired[:, :, :, 1:] - desired[:, :, :, :-1]).abs().mean(1, keepdim=True), (0, 1, 0, 0))
            vertical = torch.nn.functional.pad((desired[:, :, 1:, :] - desired[:, :, :-1, :]).abs().mean(1, keepdim=True), (0, 0, 0, 1))
            foreground = desired.amax(1, keepdim=True)
            weight = 1 + 3 * foreground + 12 * torch.clamp(horizontal + vertical, 0, .25)
            pixel = ((prediction - desired).abs() * weight).mean()
            half = F.l1_loss(F.avg_pool2d(prediction, 2), F.avg_pool2d(desired, 2))
            edge = _edge_loss(prediction, desired)
            loss = pixel + .25 * half + .65 * edge
        if not bool(torch.isfinite(loss)): raise FloatingPointError("Mobile decoder loss became non-finite.")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        with torch.no_grad(): torch._foreach_mul_(list(ema.parameters()), plan.ema_decay); torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - plan.ema_decay)
        if step == 1 or step % 100 == 0 or step == plan.steps: history.append({"step": step, "loss": float(loss), "pixel": float(pixel), "edge": float(edge)})
    model = ema.eval(); validation = torch.from_numpy(latents[split:]).to(target); predictions = []; desireds = []; began = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(validation), 4):
            latent = validation[start:start + 4]
            with torch.autocast(target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"): predictions.append(model(latent).float()); desireds.append(teacher.decode(latent).float())
    if target.type == "cuda": torch.cuda.synchronize(target)
    inference_seconds = time.perf_counter() - began; prediction = torch.cat(predictions); desired = torch.cat(desireds); mae = float(F.l1_loss(prediction, desired)); edge_mae = float(_edge_loss(prediction, desired)); mse = float(F.mse_loss(prediction, desired)); psnr = float(-10 * np.log10(max(mse, 1e-12))); foreground_mask = (desired.amax(1, keepdim=True) > .18).expand_as(desired); foreground_mae = float((prediction - desired).abs()[foreground_mask].mean()); teacher_sharpness = float((desired[:, :, :, 1:] - desired[:, :, :, :-1]).abs().mean() + (desired[:, :, 1:, :] - desired[:, :, :-1, :]).abs().mean()); student_sharpness = float((prediction[:, :, :, 1:] - prediction[:, :, :, :-1]).abs().mean() + (prediction[:, :, 1:, :] - prediction[:, :, :-1, :]).abs().mean()); sharpness_ratio = student_sharpness / max(teacher_sharpness, 1e-9); gates = {"teacher_mae": mae <= .04, "foreground_mae": foreground_mae <= .035, "teacher_psnr": psnr >= 24, "sharpness_ratio": .82 <= sharpness_ratio <= 1.18, "desktop_target_30fps": len(validation) / inference_seconds >= 30, "parameter_reduction": sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in teacher.parameters()) * .2}; status = "mobile_decoder_ready" if all(gates.values()) else "quality_failed"
    state = {name: value.detach().cpu() for name, value in model.cpu().state_dict().items()}; report = {"format": FORMAT, "status": status, "source_sha256": source_sha256(), "teacher_sha256": TEACHER_SHA256, "model_config": config_dict(config), "training_plan": config_dict(plan), "parameters": sum(value.numel() for value in state.values()), "teacher_parameters": sum(value.numel() for value in teacher.parameters()), "metrics": {"mae": mae, "foreground_mae": foreground_mae, "edge_mae": edge_mae, "psnr": psnr, "sharpness_ratio": sharpness_ratio, "frames_per_second": len(validation) / inference_seconds, "milliseconds_per_frame": inference_seconds * 1000 / len(validation)}, "runtime": {"elapsed_seconds": time.perf_counter() - started, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target)) if target.type == "cuda" else 0}, "gates": gates, "history": history, "visual_inspection_required": True, "limitations": ["Promotion is against the desktop VAE teacher; physical Galaxy S25 Ultra parity remains required."]}; output.mkdir(parents=True); temporary = output / f".runtime.pt.tmp-{os.getpid()}"; torch.save({"format": CHECKPOINT_FORMAT, "source_sha256": report["source_sha256"], "status": status, "model_config": report["model_config"], "state": state, "report": report}, temporary); os.replace(temporary, output / "runtime.pt"); report["checkpoint_sha256"] = file_sha256(output / "runtime.pt"); (output / "report.json").write_bytes(canonical(report)); _contact(output / "contact_sheet.png", desired[:6], prediction[:6]); return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--device", default="cuda"); parser.add_argument("--steps", type=int, default=3000); args = parser.parse_args(argv); print(json.dumps(train(args.output, plan=MobileDecoderPlan(steps=args.steps), device=args.device), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
