from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ..config import PROJECT_ROOT
from ..safety import require_disk_floor
from ..creature_stage_developmental.contract import FAMILIES, TISSUES
from ..creature_stage_developmental.development import develop
from ..creature_stage_developmental.genomes import review_genomes
from ..creature_stage_neural_grounded.contract import BODY_SPEED_SCALE, POSITION_SCALE
from ..multifield_style_motion.hashing import artifact_record_from_bytes, canonical_json_bytes, deterministic_npz_bytes, sha256_file
from .contract import source_sha256
from .dataset import ComponentSentinelTeacher
from .training import load_model


FORMAT = "nullvector-neural-grounded-components-evaluation-v1"
EVALUATION_FILE = "forge/creature_stage_neural_grounded_components/evaluation.py"
TISSUE_COLORS = {
    "skin": (244, 111, 126), "bone": (238, 237, 204), "muscle": (255, 69, 113),
    "tendon": (248, 188, 144), "armor": (113, 158, 177), "neural": (72, 225, 246),
    "vascular": (234, 52, 101), "respiratory": (123, 216, 244), "digestive": (241, 169, 63),
    "sensor": (253, 244, 105), "storage": (164, 102, 230), "root": (115, 229, 92),
    "phase": (185, 78, 255), "machine": (177, 195, 207), "weapon": (255, 131, 61),
}
OWNER_COLORS = ((62, 230, 255), (255, 84, 128), (255, 198, 73), (143, 241, 94),
                (194, 100, 255), (255, 137, 64), (104, 173, 255), (255, 239, 112))


def evaluation_source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-component-grounded-evaluation-v1\0")
    digest.update(source_sha256().encode("ascii") + b"\0")
    digest.update((PROJECT_ROOT / EVALUATION_FILE).read_bytes())
    return digest.hexdigest()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (Path("C:/Windows/Fonts/consola.ttf"), Path("C:/Windows/Fonts/arial.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _rollout(checkpoint: Path, *, ema: bool, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    teacher = ComponentSentinelTeacher()
    model, payload = load_model(checkpoint, ema=ema, device=device)
    identities = teacher.split_indices("validation")
    predicted, target, baseline, contact, body_predicted, body_target = [], [], [], [], [], []
    with torch.inference_mode():
        for identity in identities:
            first = teacher.sample(identity, 0)
            state = torch.from_numpy(first["state"].copy())[None].to(device)
            owner = teacher.owner(identity, device)
            identity_predicted, identity_target, identity_baseline = [], [], []
            identity_contact, identity_body_predicted, identity_body_target = [], [], []
            for frame in range(72):
                row = teacher.sample(identity, frame)
                result = model(
                    torch.from_numpy(row["static"].copy())[None].to(device), state,
                    torch.from_numpy(row["dynamic"].copy())[None].to(device), owner,
                    torch.from_numpy(row["mask"].copy())[None].to(device),
                    torch.from_numpy(row["adjacency"].copy())[None].to(device),
                    torch.tensor([row["family"]], dtype=torch.long, device=device),
                    torch.tensor([row["morphotype"]], dtype=torch.long, device=device),
                    torch.tensor([row["motion"]], dtype=torch.long, device=device),
                    torch.tensor([row["phase"]], device=device),
                    torch.from_numpy(row["controls"].copy())[None].to(device),
                )
                current = result.cells[0].float().cpu().numpy()
                identity_predicted.append(current)
                identity_target.append(row["target"])
                identity_baseline.append(state[0].float().cpu().numpy())
                identity_contact.append(row["dynamic"][:, 5] > .5)
                identity_body_predicted.append(float(result.body_velocity[0]))
                identity_body_target.append(float(row["body_target"]))
                state = result.cells.detach()
            predicted.append(np.stack(identity_predicted)); target.append(np.stack(identity_target))
            baseline.append(np.stack(identity_baseline)); contact.append(np.stack(identity_contact))
            body_predicted.append(identity_body_predicted); body_target.append(identity_body_target)
    arrays = {
        "identity": np.asarray(identities, dtype=np.uint8),
        "predicted": np.stack(predicted).astype(np.float32),
        "target": np.stack(target).astype(np.float32),
        "baseline": np.stack(baseline).astype(np.float32),
        "contact": np.stack(contact).astype(np.uint8),
        "body_predicted": np.asarray(body_predicted, dtype=np.float32),
        "body_target": np.asarray(body_target, dtype=np.float32),
        "mask": teacher.arrays["cell_mask"][list(identities)].astype(np.uint8),
        "owner": teacher.arrays["appendage_owner"][list(identities)].astype(np.int16),
    }
    return arrays, payload


def _metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    predicted, target, baseline = arrays["predicted"], arrays["target"], arrays["baseline"]
    mask = arrays["mask"].astype(bool); owners = arrays["owner"]
    active = np.broadcast_to(mask[:, None, :, None], predicted[..., :1].shape)
    xy_active = active.repeat(2, axis=3)
    position = np.abs((predicted[..., :2] - target[..., :2]) * POSITION_SCALE)
    previous = np.abs((baseline[..., :2] - target[..., :2]) * POSITION_SCALE)
    velocity = np.abs((predicted[..., 2:] - target[..., 2:]) * POSITION_SCALE)
    appendage = np.broadcast_to((owners >= 0)[:, None, :, None], active.shape) & active
    contacts = arrays["contact"].astype(bool)[:, :, :, None] & active
    centroid_errors, shape_errors, centroid_jerks, centroid_energy = [], [], [], []
    for organism in range(len(mask)):
        for owner in sorted(set(int(value) for value in owners[organism, mask[organism]]) - {-1}):
            member = (owners[organism] == owner) & mask[organism]
            pred_xy = predicted[organism, :, member, :2] * POSITION_SCALE
            target_xy = target[organism, :, member, :2] * POSITION_SCALE
            pred_centroid = pred_xy.mean(1); target_centroid = target_xy.mean(1)
            centroid_errors.append(np.abs(pred_centroid - target_centroid).reshape(-1))
            shape_errors.append(np.abs((pred_xy - pred_centroid[:, None]) - (target_xy - target_centroid[:, None])).reshape(-1))
            pred_accel = np.roll(pred_centroid, -1, axis=0) - 2 * pred_centroid + np.roll(pred_centroid, 1, axis=0)
            target_accel = np.roll(target_centroid, -1, axis=0) - 2 * target_centroid + np.roll(target_centroid, 1, axis=0)
            centroid_jerks.append(np.abs(pred_accel - target_accel).reshape(-1))
            centroid_energy.append(np.square(pred_centroid - pred_centroid.mean(0)).sum(1).mean())
    pred_energy = float(np.sqrt(np.mean(np.square(predicted[..., :2][xy_active] * POSITION_SCALE))))
    target_energy = float(np.sqrt(np.mean(np.square(target[..., :2][xy_active] * POSITION_SCALE))))
    seam = (predicted[:, 0, :, :2] - predicted[:, -1, :, :2]) - (target[:, 0, :, :2] - target[:, -1, :, :2])
    body_predicted = arrays["body_predicted"] * BODY_SPEED_SCALE
    body_target = arrays["body_target"] * BODY_SPEED_SCALE
    predicted_distance, target_distance = float(body_predicted.sum(1).mean()), float(body_target.sum(1).mean())
    result = {
        "position_mae_px": float(position[xy_active].mean()),
        "copy_previous_mae_px": float(previous[xy_active].mean()),
        "velocity_mae_px": float(velocity[xy_active].mean()),
        "appendage_mae_px": float(position[appendage.repeat(2, axis=3)].mean()),
        "contact_mae_px": float(position[contacts.repeat(2, axis=3)].mean()),
        "owner_centroid_mae_px": float(np.concatenate(centroid_errors).mean()),
        "owner_internal_shape_mae_px": float(np.concatenate(shape_errors).mean()),
        "owner_acceleration_mae_px": float(np.concatenate(centroid_jerks).mean()),
        "owner_motion_energy_px2": float(np.mean(centroid_energy)),
        "energy_ratio": pred_energy / max(target_energy, 1e-8),
        "loop_transition_error_px": float(np.max(np.abs(seam)[np.broadcast_to(mask[:, :, None], seam.shape)] * POSITION_SCALE)),
        "body_velocity_mae_px": float(np.abs(body_predicted - body_target).mean()),
        "predicted_distance_px": predicted_distance,
        "target_distance_px": target_distance,
        "advance_ratio": predicted_distance / max(target_distance, 1e-8),
    }
    result["baseline_improvement"] = (result["copy_previous_mae_px"] - result["position_mae_px"]) / max(result["copy_previous_mae_px"], 1e-8)
    return {name: round(value, 9) for name, value in result.items()}


def _gates(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "finite": all(math.isfinite(value) for value in metrics.values()),
        "beats_copy_previous": metrics["baseline_improvement"] >= .10,
        "position_accuracy": metrics["position_mae_px"] <= .35,
        "velocity_accuracy": metrics["velocity_mae_px"] <= .20,
        "appendage_accuracy": metrics["appendage_mae_px"] <= .45,
        "contact_anchor_accuracy": metrics["contact_mae_px"] <= .35,
        "component_translation_accuracy": metrics["owner_centroid_mae_px"] <= .35,
        "component_shape_coherence": metrics["owner_internal_shape_mae_px"] <= .25,
        "component_acceleration_continuity": metrics["owner_acceleration_mae_px"] <= .18,
        "motion_energy": .70 <= metrics["energy_ratio"] <= 1.30,
        "loop_closure": metrics["loop_transition_error_px"] <= .35,
        "body_velocity_accuracy": metrics["body_velocity_mae_px"] <= .08,
        "grounded_advance": .70 <= metrics["advance_ratio"] <= 1.30,
    }


def _render_frame(teacher: ComponentSentinelTeacher, arrays: dict[str, np.ndarray], frame: int) -> Image.Image:
    panel_w, panel_h, header = 300, 280, 62
    image = Image.new("RGB", (panel_w * 5, header + panel_h), (3, 8, 14))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, header), fill=(6, 17, 25, 255))
    draw.text((18, 10), "NEURAL COMPONENT LOCOMOTION // UNSEEN GRAFTS // CELL + GROUND AUTHORITY", font=_font(20), fill=(220, 243, 248, 255))
    draw.text((image.width - 170, 16), f"PHASE {frame:02d}/71", font=_font(14), fill=(77, 226, 247, 255))
    genomes = review_genomes()
    for column, identity in enumerate(arrays["identity"]):
        identity = int(identity); x0, y0 = column * panel_w, header
        record = teacher.manifest["cycles"][identity]; family = int(teacher.arrays["family"][identity])
        organism = develop(genomes[identity]); count = int(teacher.arrays["cell_mask"][identity].sum())
        rest = teacher.arrays["rest_cells"][identity, :count]
        predicted = rest + arrays["predicted"][column, frame, :count, :2] * POSITION_SCALE
        target = rest + arrays["target"][column, frame, :count, :2] * POSITION_SCALE
        tissue = teacher.arrays["tissue"][identity, :count]; owner = teacher.arrays["appendage_owner"][identity, :count]
        cx, cy, scale = x0 + panel_w * .5, y0 + 151, 4.25
        ground = cy + float(teacher.arrays["ground_y"][identity]) * scale
        draw.rectangle((x0, y0, x0 + panel_w - 1, y0 + panel_h - 1), outline=(26, 65, 78, 255))
        draw.text((x0 + 9, y0 + 8), record["genome_id"].upper(), font=_font(11), fill=(220, 239, 244, 255))
        draw.text((x0 + 9, y0 + 26), f"{FAMILIES[family].upper()} // {' + '.join(record['locomotor_modes'])}", font=_font(8), fill=(100, 218, 240, 255))
        draw.line((x0 + 10, ground, x0 + panel_w - 10, ground), fill=(130, 238, 104, 120), width=1)
        draw.line((cx, y0 + 48, cx, ground), fill=(95, 210, 231, 38), width=1)
        # The true physical skeleton is a faint authority overlay. Predicted cells
        # remain visually dominant, making component drift and collapse obvious.
        nodes = teacher.arrays["nodes_local"][identity, frame]
        for left, right in organism.skeleton_edges:
            a, b = nodes[int(left)], nodes[int(right)]
            draw.line((cx + a[0] * scale, cy + a[1] * scale, cx + b[0] * scale, cy + b[1] * scale), fill=(87, 216, 239, 75), width=1)
        for x, y in target:
            px, py = cx + float(x) * scale, cy + float(y) * scale
            draw.rectangle((px - .7, py - .7, px + .7, py + .7), outline=(214, 250, 255, 62))
        for cell_index, ((x, y), tissue_id) in enumerate(zip(predicted, tissue, strict=True)):
            base = TISSUE_COLORS[TISSUES[int(tissue_id)]]
            if owner[cell_index] >= 0:
                accent = OWNER_COLORS[int(owner[cell_index])]
                color = tuple((2 * base[channel] + accent[channel]) // 3 for channel in range(3))
            else:
                color = base
            px, py = cx + float(x) * scale, cy + float(y) * scale
            draw.rectangle((px - 1.45, py - 1.45, px + 1.45, py + 1.45), fill=(*color, 225))
        active = teacher.arrays["contact_active"][identity, frame].astype(bool)
        anchors = teacher.arrays["contact_anchor_local"][identity, frame]
        for owner_id in np.flatnonzero(active):
            member = owner == owner_id
            if not member.any():
                continue
            tip = predicted[member][np.argmax(predicted[member, 1])]
            anchor = anchors[owner_id]
            tx, ty = cx + float(tip[0]) * scale, cy + float(tip[1]) * scale
            ax, ay = cx + float(anchor[0]) * scale, cy + float(anchor[1]) * scale
            accent = OWNER_COLORS[int(owner_id)]
            draw.line((tx, ty, ax, ay), fill=(*accent, 205), width=2)
            draw.ellipse((ax - 4, ay - 2, ax + 4, ay + 2), fill=(*accent, 235))
        draw.text((x0 + 9, y0 + panel_h - 20), "SOLID predicted  /  GHOST target  /  LINE planted tether", font=_font(7), fill=(138, 168, 176, 255))
    return image


def _render_artifacts(teacher: ComponentSentinelTeacher, arrays: dict[str, np.ndarray], stage: Path) -> dict[str, dict[str, Any]]:
    frames = stage / "frames"; frames.mkdir()
    for frame in range(72):
        _render_frame(teacher, arrays, frame).save(frames / f"frame_{frame:03d}.png", compress_level=9)
    sheet = Image.new("RGB", (1500, 342 * 4), (3, 8, 14))
    for index, frame in enumerate((0, 18, 36, 54)):
        sheet.paste(_render_frame(teacher, arrays, frame), (0, index * 342))
    sheet_path = stage / "component_contact_sheet.png"; sheet.save(sheet_path, compress_level=9)
    mp4 = stage / "component_motion.mp4"; gif = stage / "component_motion.gif"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "18", "-i", str(frames / "frame_%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "18", "-i", str(frames / "frame_%03d.png"), "-vf", "fps=18,scale=1100:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3", str(gif)], check=True)
    shutil.rmtree(frames)
    return {name: artifact_record_from_bytes(path.name, path.read_bytes()) for name, path in (("contact_sheet", sheet_path), ("gif", gif), ("mp4", mp4))}


def evaluate(checkpoint: Path, output: Path, *, device: str = "cuda", visually_inspected: bool = False) -> dict[str, Any]:
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=2 * 1024**3)
    target_device = torch.device(device); candidates = {}
    candidate_arrays = {}
    payload = None
    for name, ema in (("raw", False), ("ema", True)):
        arrays, current_payload = _rollout(checkpoint, ema=ema, device=target_device)
        metrics = _metrics(arrays); gates = _gates(metrics)
        candidates[name] = {"metrics": metrics, "gates": gates, "passed_gate_count": sum(gates.values())}
        candidate_arrays[name] = arrays; payload = current_payload
    def score(name: str) -> tuple[int, float, float, float]:
        item = candidates[name]
        metrics = item["metrics"]
        return item["passed_gate_count"], metrics["baseline_improvement"], -metrics["contact_mae_px"], -metrics["position_mae_px"]
    selected = max(candidates, key=score); arrays = candidate_arrays[selected]
    stage = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"; stage.mkdir(parents=True)
    try:
        archive = deterministic_npz_bytes(arrays); (stage / "heldout_rollout.npz").write_bytes(archive)
        artifacts = _render_artifacts(ComponentSentinelTeacher(), arrays, stage)
        artifacts["rollout"] = artifact_record_from_bytes("heldout_rollout.npz", archive)
        gates = candidates[selected]["gates"]
        report = {
            "format": FORMAT, "status": "passed" if all(gates.values()) else "failed-quality",
            "source_sha256": source_sha256(), "evaluation_source_sha256": evaluation_source_sha256(),
            "checkpoint": {"path": checkpoint.relative_to(PROJECT_ROOT).as_posix(), "bytes": checkpoint.stat().st_size, "sha256": sha256_file(checkpoint), "updates": payload["updates"], "model_state_sha256": payload["model_state_sha256"], "ema_state_sha256": payload["ema_state_sha256"]},
            "scope": {"split": "untouched_grafted_sentinels", "identities": arrays["identity"].tolist(), "families": 5, "frames_per_identity": 72, "prediction_fed": True, "rasterizer_used": False},
            "selection": {"selected": selected, "criterion": "gate_count_then_copy_improvement_then_contact_then_position", "candidates": candidates},
            "metrics": candidates[selected]["metrics"], "gates": gates,
            "visually_inspected": bool(visually_inspected), "promotion_eligible": all(gates.values()) and bool(visually_inspected),
            "artifacts": artifacts,
        }
        report["semantic_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
        (stage / "evaluation_manifest.json").write_bytes(canonical_json_bytes(report)); os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True); raise
    return validate_evaluation(output)


def validate_evaluation(output: Path) -> dict[str, Any]:
    output = Path(output).resolve(); manifest = output / "evaluation_manifest.json"
    if manifest.is_symlink() or not manifest.is_file() or not 0 < manifest.stat().st_size <= 2 * 1024**2:
        raise ValueError("component evaluation manifest missing or oversized")
    raw = manifest.read_bytes(); report = json.loads(raw)
    if raw != canonical_json_bytes(report):
        raise ValueError("component evaluation manifest is not canonical")
    semantic = report.pop("semantic_sha256")
    if semantic != hashlib.sha256(canonical_json_bytes(report)).hexdigest():
        raise ValueError("component evaluation semantic hash drifted")
    report["semantic_sha256"] = semantic
    if report["format"] != FORMAT or report["source_sha256"] != source_sha256() or report["evaluation_source_sha256"] != evaluation_source_sha256():
        raise ValueError("component evaluation source provenance drifted")
    checkpoint = PROJECT_ROOT / report["checkpoint"]["path"]
    if checkpoint.stat().st_size != report["checkpoint"]["bytes"] or sha256_file(checkpoint) != report["checkpoint"]["sha256"]:
        raise ValueError("component evaluation checkpoint drifted")
    for artifact in report["artifacts"].values():
        path = output / artifact["path"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError("component evaluation artifact drifted")
    if report["status"] != ("passed" if all(report["gates"].values()) else "failed-quality"):
        raise ValueError("component evaluation verdict drifted")
    if report["promotion_eligible"] is not (all(report["gates"].values()) and report["visually_inspected"]):
        raise ValueError("component evaluation promotion relationship drifted")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate component-aware grounded neural motion")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visually-inspected", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.output, device=args.device, visually_inspected=args.visually_inspected)
    print(json.dumps({"status": report["status"], "selected": report["selection"]["selected"], "metrics": report["metrics"], "gates": report["gates"], "semantic_sha256": report["semantic_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
