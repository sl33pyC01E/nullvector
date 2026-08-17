from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from ..maps.io import file_sha256
from ..neural_city_layout_v1.evaluation import PALETTE as CITY_PALETTE
from .contract import CHECKPOINT_FORMAT, FORMAT, WorldStateModelConfig, canonical, source_sha256
from .data import THEMES, build_corpus
from .model import build_model


TERRAIN_PALETTE = np.asarray(((4, 10, 14), (55, 65, 76), (25, 96, 80), (92, 72, 59), (26, 104, 132), (118, 104, 70), (96, 52, 112), (72, 136, 66)), np.uint8)


def _composite(terrain: np.ndarray, city: np.ndarray, continuous: np.ndarray) -> np.ndarray:
    image = TERRAIN_PALETTE[terrain].astype(np.float32); biomass = continuous[3]; mineral = continuous[4]; energy = continuous[6]
    image[..., 1] += biomass * 75; image[..., :2] += mineral[..., None] * np.asarray((45, 33)); image[..., 2] += energy * 95
    occupied = city != 0; image[occupied] = CITY_PALETTE[city[occupied]]
    return np.clip(image, 0, 255).astype(np.uint8)


def _sheet(source: list[np.ndarray], reconstructed: list[np.ndarray]) -> Image.Image:
    scale = 7; tile = 32 * scale; header = 26; gap = 8; image = Image.new("RGB", (gap + 3 * (tile + gap), gap + len(source) * (tile + header + gap)), (3, 8, 12)); draw = ImageDraw.Draw(image)
    for row, (left, right) in enumerate(zip(source, reconstructed, strict=True)):
        y = gap + row * (tile + header + gap); difference = np.abs(left.astype(np.int16) - right.astype(np.int16)).max(2).astype(np.uint8); diff = np.zeros_like(left); diff[..., 0] = difference
        for column, (label, field) in enumerate(((f"{THEMES[row]} SOURCE", left), ("NEURAL RECON", right), ("RGB DIFFERENCE", diff))):
            x = gap + column * (tile + gap); draw.text((x, y), label.upper(), fill=(105, 226, 240)); image.paste(Image.fromarray(field).resize((tile, tile), Image.Resampling.NEAREST), (x, y + header))
    return image


@torch.inference_mode()
def evaluate(checkpoint: Path, output: Path, *, device: str = "cuda", visually_inspected: bool = False) -> dict[str, object]:
    checkpoint = Path(checkpoint).resolve(); output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256(): raise ValueError("World-state checkpoint provenance drifted.")
    target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu"); model = build_model(WorldStateModelConfig(**payload["model_config"])); model.load_state_dict(payload["state"], strict=True); model.to(target).eval(); corpus = build_corpus(1024, seed=payload["report"]["training_config"]["seed"] ^ 0x4556414C); indices = [next(index for index in range(theme, 1024, 6)) for theme in range(6)]; t = torch.from_numpy(corpus.terrain[indices]).long().to(target); c = torch.from_numpy(corpus.city[indices]).long().to(target); x = torch.from_numpy(corpus.continuous[indices]).float().to(target); q = torch.from_numpy(corpus.condition[indices]).float().to(target)
    if target.type == "cuda": torch.cuda.reset_peak_memory_stats(target)
    with torch.autocast(target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"): result = model(t, c, x, q, sample=False)
    predicted_t = result.terrain.argmax(1); predicted_c = result.city.argmax(1); terrain_accuracy = float((predicted_t == t).float().mean()); city_iou = float((((predicted_c != 0) & (c != 0)).sum() / ((predicted_c != 0) | (c != 0)).sum().clamp_min(1))); city_recall = {str(index): float((predicted_c[c == index] == index).float().mean()) if bool((c == index).any()) else 1.0 for index in range(8)}; minimum_specialized_recall = min(city_recall[str(index)] for index in (4, 5, 6, 7)); continuous_mae = float(F.l1_loss(result.continuous.float(), x)); condition_mae = float(F.l1_loss(result.condition.float(), q))
    source = [_composite(corpus.terrain[index], corpus.city[index], corpus.continuous[index].astype(np.float32)) for index in indices]; reconstructed = [_composite(predicted_t[row].cpu().numpy(), predicted_c[row].cpu().numpy(), result.continuous[row].float().cpu().numpy()) for row in range(6)]
    single_t, single_c, single_x, single_q = t[:1], c[:1], x[:1], q[:1]
    with torch.autocast(target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"):
        for _ in range(20): model(single_t, single_c, single_x, single_q, sample=False)
        if target.type == "cuda": torch.cuda.synchronize(target)
        started = time.perf_counter()
        for _ in range(300): model(single_t, single_c, single_x, single_q, sample=False)
        if target.type == "cuda": torch.cuda.synchronize(target)
    elapsed = time.perf_counter() - started; output.mkdir(parents=True); _sheet(source, reconstructed).save(output / "contact_sheet.png"); np.savez_compressed(output / "latents.npz", spatial=result.mean.float().cpu().numpy().astype(np.float16), global_state=result.global_state.float().cpu().numpy().astype(np.float16))
    gates = {"terrain_accuracy": terrain_accuracy >= .97, "city_foreground_iou": city_iou >= .90, "minimum_specialized_city_recall": minimum_specialized_recall >= .65, "continuous_mae": continuous_mae <= .06, "condition_mae": condition_mae <= .05, "realtime_30fps": elapsed / 300 <= 1 / 30}; report = {"format": FORMAT, "status": "runtime_ready" if all(gates.values()) else "quality_failed", "source_sha256": source_sha256(), "checkpoint_sha256": file_sha256(checkpoint), "terrain_accuracy": terrain_accuracy, "city_foreground_iou": city_iou, "city_class_recall": city_recall, "minimum_specialized_city_recall": minimum_specialized_recall, "continuous_mae": continuous_mae, "condition_mae": condition_mae, "single_state_milliseconds": elapsed / 300 * 1000, "single_state_hz": 300 / elapsed, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target)) if target.type == "cuda" else 0, "spatial_latent_shape": list(result.mean.shape[1:]), "global_state_shape": list(result.global_state.shape[1:]), "approximate_float16_compression_ratio": (2 * 32 * 32 + 7 * 32 * 32 * 2 + len(q[0]) * 2) / ((20 * 8 * 8 + 64) * 2), "gates": gates, "visually_inspected": bool(visually_inspected), "artifacts": {}}
    for name in ("contact_sheet.png", "latents.npz"): path = output / name; report["artifacts"][name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    (output / "report.json").write_bytes(canonical(report)); return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--visually-inspected", action="store_true"); args = parser.parse_args(argv); print(json.dumps(evaluate(args.checkpoint, args.output, device=args.device, visually_inspected=args.visually_inspected), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
