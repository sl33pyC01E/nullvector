from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from ..maps.io import file_sha256
from ..neural_city_layout_v1.contract import CLASSES, GRID_SIZE
from ..neural_city_layout_v1.evaluation import PALETTE
from ..neural_city_layout_v1.teacher import CLASS_INDEX, _condition, validate_compiled_city
from .contract import ACTIONS, CHECKPOINT_FORMAT, FORMAT, GrowthCondition, GrowthModelConfig, canonical_json_bytes, source_sha256
from .model import build_model
from .projection import compile_growth_state, project_neural_growth
from .teacher import _origin, apply_teacher_growth, extract_local_patch, local_condition_vector, paste_local_patch


def _load(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256(): raise ValueError("Growth checkpoint provenance drifted.")
    model = build_model(GrowthModelConfig(**payload["model_config"])); model.load_state_dict(payload["ema_state"], strict=True); return model.to(device).eval(), payload


def _render_sheet(teacher: list[np.ndarray], neural: list[np.ndarray]) -> Image.Image:
    scale = 4; tile_size = GRID_SIZE * scale; gutter = 8; header = 28
    image = Image.new("RGB", (gutter + 3 * (tile_size + gutter), gutter + 5 * (tile_size + header + gutter)), (3, 8, 12)); draw = ImageDraw.Draw(image)
    for family, (target, generated) in enumerate(zip(teacher, neural, strict=True)):
        y = gutter + family * (tile_size + header + gutter); draw.text((gutter, y), f"FAMILY {family} // SIX GROWTH TICKS", fill=(105, 226, 240)); changed = target != generated
        overlay = PALETTE[generated].copy(); overlay[changed] = np.asarray((255, 70, 160), np.uint8)
        for column, (name, field) in enumerate((("TEACHER", PALETTE[target]), ("NEURAL", PALETTE[generated]), ("DIFF", overlay))):
            x = gutter + column * (tile_size + gutter); draw.text((x + 92, y), name, fill=(220, 235, 235)); image.paste(Image.fromarray(field).resize((tile_size, tile_size), Image.Resampling.NEAREST), (x, y + header))
    return image


@torch.inference_mode()
def evaluate(checkpoint: Path, output: Path, *, device: str = "cuda", visually_inspected: bool = False) -> dict[str, object]:
    checkpoint = Path(checkpoint).resolve(); output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    target_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu"); model, payload = _load(checkpoint, target_device)
    if target_device.type == "cuda": torch.cuda.reset_peak_memory_stats(target_device)
    final_teacher = []; final_neural = []; steps = []; noop_passes = 0; compiled_valid = True; connected_passes = 0; semantic_passes = 0
    for family in range(5):
        city = replace(_condition(0x47524F575448 + family * 7919), family=family, building_target=6); teacher = np.zeros((GRID_SIZE, GRID_SIZE), np.uint8); neural = teacher.copy()
        for stage in range(6):
            action = ACTIONS[(stage + family) % len(ACTIONS)]; origin = _origin(city, stage); site = ((origin[0] % GRID_SIZE) / (GRID_SIZE - 1), (origin[1] % GRID_SIZE) / (GRID_SIZE - 1)); condition = GrowthCondition(city, action, (1, 1, 1, 1), site, stage); target, _ = apply_teacher_growth(teacher, condition)
            current_patch = extract_local_patch(neural, site); current_tensor = torch.from_numpy(current_patch)[None].long().to(target_device); condition_tensor = torch.from_numpy(local_condition_vector(condition))[None].to(target_device)
            with torch.autocast(target_device.type, dtype=torch.bfloat16, enabled=target_device.type == "cuda"): raw_patch = model(current_tensor, condition_tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
            raw = paste_local_patch(neural, raw_patch, site); projected, projection = project_neural_growth(neural, raw, condition); compiled, diagnostics = compile_growth_state(projected); validation = validate_compiled_city(compiled); compiled_valid &= bool(validation["passed"])
            occupied_union = (target != 0) | (compiled != 0); categorical = float((target[occupied_union] == compiled[occupied_union]).mean()) if bool(occupied_union.any()) else 1.0; occupied_iou = float(((target != 0) & (compiled != 0)).sum() / max(1, ((target != 0) | (compiled != 0)).sum())); prior = neural != 0; preservation = float((compiled[prior] == neural[prior]).mean()) if bool(prior.any()) else 1.0
            noop = GrowthCondition(city, action, (0, 0, 0, 0), site, stage); noop_vector = torch.from_numpy(local_condition_vector(noop))[None].to(target_device); noop_patch = extract_local_patch(compiled, site)
            with torch.autocast(target_device.type, dtype=torch.bfloat16, enabled=target_device.type == "cuda"): noop_raw_patch = model(torch.from_numpy(noop_patch)[None].long().to(target_device), noop_vector).argmax(1)[0].cpu().numpy().astype(np.uint8)
            noop_raw = paste_local_patch(compiled, noop_raw_patch, site)
            noop_prediction, _ = project_neural_growth(compiled, noop_raw, noop); noop_exact = bool(np.array_equal(noop_prediction, compiled)); noop_passes += int(noop_exact)
            expected_feature = {"habitat": "garden", "granary": "storage", "market": "storage"}.get(action, "utility")
            added = (compiled != 0) & (neural == 0)
            semantic_cells = int((added & (compiled == CLASS_INDEX[expected_feature])).sum())
            connected = diagnostics["toroidal_components"] == 1; connected_passes += int(connected)
            semantic_ok = semantic_cells > 0; semantic_passes += int(semantic_ok)
            steps.append({"family": family, "stage": stage, "action": action, "occupied_iou_diagnostic": occupied_iou, "categorical_foreground_accuracy_diagnostic": categorical, "prior_preservation": preservation, "compile_edit_fraction": diagnostics["edited_cells"] / (GRID_SIZE * GRID_SIZE), "noop_exact": noop_exact, "accepted_neural_changes": projection["accepted_changes"], "rejected_out_of_scope_changes": projection["rejected_changes"], "added_cells": int(added.sum()), "expected_feature": expected_feature, "new_expected_feature_cells": semantic_cells, "single_toroidal_network": connected})
            teacher = target; neural = compiled
        final_teacher.append(teacher); final_neural.append(neural)
    output.mkdir(parents=True, exist_ok=False); teacher_array = np.stack(final_teacher); neural_array = np.stack(final_neural); np.save(output / "teacher.npy", teacher_array, allow_pickle=False); np.save(output / "neural.npy", neural_array, allow_pickle=False); _render_sheet(final_teacher, final_neural).save(output / "contact_sheet.png")
    mean_iou = float(np.mean([item["occupied_iou_diagnostic"] for item in steps])); mean_categorical = float(np.mean([item["categorical_foreground_accuracy_diagnostic"] for item in steps])); mean_preservation = float(np.mean([item["prior_preservation"] for item in steps])); max_compile = float(max(item["compile_edit_fraction"] for item in steps)); noop_rate = noop_passes / len(steps)
    network_rate = connected_passes / len(steps); semantic_rate = semantic_passes / len(steps); progress_rate = float(np.mean([item["added_cells"] >= 24 for item in steps]))
    gates = {"all_compiled_valid": compiled_valid, "single_accessible_network_rate": network_rate == 1, "action_semantic_rate": semantic_rate >= .85, "meaningful_growth_rate": progress_rate >= .95, "prior_preservation": mean_preservation >= .99, "noop_exact_rate": noop_rate >= .95, "compile_edit_bounded": max_compile <= .05}
    report = {"format": FORMAT, "status": "experimental_ready" if all(gates.values()) else "quality_failed", "source_sha256": source_sha256(), "checkpoint_sha256": file_sha256(checkpoint), "families": 5, "rollout_steps_per_family": 6, "mean_occupied_iou_diagnostic": mean_iou, "mean_categorical_foreground_accuracy_diagnostic": mean_categorical, "single_accessible_network_rate": network_rate, "action_semantic_rate": semantic_rate, "meaningful_growth_rate": progress_rate, "mean_prior_preservation": mean_preservation, "noop_exact_rate": noop_rate, "maximum_compile_edit_fraction": max_compile, "gates": gates, "steps": steps, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target_device)) if target_device.type == "cuda" else 0, "visually_inspected": bool(visually_inspected), "artifacts": {}}
    for name in ("teacher.npy", "neural.npy", "contact_sheet.png"):
        path = output / name; report["artifacts"][name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    (output / "report.json").write_bytes(canonical_json_bytes(report)); return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--visually-inspected", action="store_true"); args = parser.parse_args(argv); print(json.dumps(evaluate(args.checkpoint, args.output, device=args.device, visually_inspected=args.visually_inspected), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
