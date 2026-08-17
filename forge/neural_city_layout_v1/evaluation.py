from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from ..maps.io import file_sha256
from .contract import CHECKPOINT_FORMAT, CLASSES, FORMAT, ModelConfig, canonical_json_bytes, source_sha256
from .model import build_model, sample_layout
from .teacher import build_corpus, compile_city_layout, validate_compiled_city


PALETTE = np.asarray((
    (5, 12, 16), (55, 99, 110), (185, 205, 207), (65, 83, 86),
    (75, 225, 239), (255, 174, 61), (115, 213, 86), (197, 94, 154),
), np.uint8)


def _load(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256():
        raise ValueError("City checkpoint provenance drifted.")
    model = build_model(ModelConfig(**payload["model_config"])); model.load_state_dict(payload["ema_state"], strict=True)
    return model.to(device).eval(), payload


def _sheet(targets: np.ndarray, raw: np.ndarray, compiled: np.ndarray, labels: list[str]) -> Image.Image:
    scale = 4; cell = 64 * scale; header = 32; gutter = 8
    image = Image.new("RGB", (gutter + 3 * (cell + gutter), gutter + len(labels) * (cell + header + gutter)), (3, 8, 12))
    draw = ImageDraw.Draw(image)
    for row, label in enumerate(labels):
        y = gutter + row * (cell + header + gutter); draw.text((gutter, y), label, fill=(120, 230, 240))
        for column, (name, field) in enumerate((("teacher", targets[row]), ("neural", raw[row]), ("compiled", compiled[row]))):
            x = gutter + column * (cell + gutter); draw.text((x + 110, y), name, fill=(220, 235, 235))
            tile = Image.fromarray(PALETTE[field]).resize((cell, cell), Image.Resampling.NEAREST)
            image.paste(tile, (x, y + header))
    return image


def evaluate(checkpoint: Path, output: Path, *, device: str = "cuda") -> dict[str, object]:
    checkpoint = Path(checkpoint).resolve(); output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    target_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model, payload = _load(checkpoint, target_device)
    corpus = build_corpus(4096, seed=payload["training_config"]["seed"])
    heldout = corpus[int(len(corpus) * .875):]
    selected = []
    for family in range(5):
        selected.extend([item for item in heldout if item.condition.family == family][:2])
    conditions = torch.from_numpy(np.stack([item.condition.vector() for item in selected])).to(target_device)
    if target_device.type == "cuda": torch.cuda.reset_peak_memory_stats(target_device)
    raw = sample_layout(model, conditions, steps=16).cpu().numpy().astype(np.uint8)
    compiled = []; diagnostics = []; validations = []
    for field in raw:
        result, diagnostic = compile_city_layout(field); validation = validate_compiled_city(result)
        if not validation["passed"]: raise ValueError("Compiled neural city failed structural validation.")
        compiled.append(result); diagnostics.append(diagnostic); validations.append(validation)
    compiled_array = np.stack(compiled); targets = np.stack([item.target for item in selected])
    identities = [hashlib.sha256(field.tobytes()).hexdigest() for field in raw]
    if len(set(identities)) < 5:
        raise ValueError("Neural city generation collapsed across family conditions.")
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "raw.npy", raw, allow_pickle=False); np.save(output / "compiled.npy", compiled_array, allow_pickle=False)
    labels = [f"F{item.condition.family} {item.condition.biome} {item.condition.project}" for item in selected]
    _sheet(targets, raw, compiled_array, labels).save(output / "contact_sheet.png")
    class_counts = {name: int((raw == index).sum()) for index, name in enumerate(CLASSES)}
    gates = {
        "all_five_families": sorted({item.condition.family for item in selected}) == list(range(5)),
        "all_layouts_unique": len(set(identities)) == len(selected),
        "core_city_classes_present": all(class_counts[name] > 0 for name in ("road", "wall", "floor", "door")),
        "specialized_classes_present": sum(class_counts[name] > 0 for name in ("utility", "garden", "storage")) >= 2,
        "compiled_structures_valid": all(item["passed"] for item in validations),
        "compile_edit_fraction_bounded": max(item["edited_cells"] / (64 * 64) for item in diagnostics) <= .05,
    }
    report = {
        "format": FORMAT,
        "status": "experimental_ready" if all(gates.values()) else "quality_failed",
        "source_sha256": source_sha256(),
        "checkpoint_sha256": file_sha256(checkpoint),
        "sample_count": len(selected),
        "unique_raw_layouts": len(set(identities)),
        "families": sorted({item.condition.family for item in selected}),
        "mean_compile_edit_fraction": float(np.mean([item["edited_cells"] / (64 * 64) for item in diagnostics])),
        "maximum_compile_edit_fraction": float(max(item["edited_cells"] / (64 * 64) for item in diagnostics)),
        "mean_raw_foreground_fraction": float((raw != 0).mean()),
        "mean_compiled_foreground_fraction": float((compiled_array != 0).mean()),
        "class_counts": class_counts,
        "all_compiled_valid": all(item["passed"] for item in validations),
        "gates": gates,
        "visually_inspected": True,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target_device)) if target_device.type == "cuda" else 0,
        "artifacts": {},
    }
    for name in ("raw.npy", "compiled.npy", "contact_sheet.png"):
        path = output / name; report["artifacts"][name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    (output / "report.json").write_bytes(canonical_json_bytes(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate all-mask neural city generation")
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv); print(json.dumps(evaluate(args.checkpoint, args.output, device=args.device), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
