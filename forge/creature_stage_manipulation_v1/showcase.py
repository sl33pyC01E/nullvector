from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from ..creature_stage_developmental.development import develop
from ..creature_stage_developmental.genomes import review_genomes
from ..creature_stage_developmental.contract import FAMILIES
from ..creature_stage_neural_grasper_v1.feeding import FoodClump, feeder_status
from .arena import ManipulationStep, NeuralManipulationArena


WIDTH, HEIGHT, SCALE = 384, 288, 4.0
BG = (4, 10, 16, 255)
INK = (26, 238, 242, 255)
HOT = (255, 82, 133, 255)
FOOD = (144, 255, 55, 255)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _point(value: np.ndarray) -> tuple[int, int]:
    return int(round(WIDTH / 2 + value[0] * SCALE)), int(round(HEIGHT / 2 + 18 + value[1] * SCALE))


def _frame(arena: NeuralManipulationArena, target_id: int, title: str, note: str, step) -> Image.Image:
    image = Image.new("RGBA", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH - 1, HEIGHT - 1), outline=(31, 63, 76, 255))
    draw.line((20, HEIGHT - 43, WIDTH - 20, HEIGHT - 43), fill=(20, 91, 95, 255), width=1)
    draw.text((14, 10), title, fill=INK)
    draw.text((14, 24), note, fill=(129, 153, 167, 255))
    _, plane_y = _point(np.asarray((0.0, arena.ground_plane_y()), np.float64))
    draw.line((30, plane_y, WIDTH - 30, plane_y), fill=(24, 57, 66, 170), width=1)
    status = feeder_status(arena.living)
    appendage = min(int(step.appendage), len(arena.articulation.chain_ids) - 1)
    body_points = arena.articulation.cells().astype(np.float64) + arena.body.position
    for index, raw in enumerate(body_points):
        x, y = _point(raw)
        alive = bool(arena.living.alive_mask[index])
        feeder = bool(status.feeder_mask[index])
        selected_limb = int(arena.organism.appendage_index[index]) == appendage
        color = HOT if feeder and alive else ((126, 255, 196, 255) if selected_limb and alive else (72, 212, 220, 255)) if alive else (50, 42, 49, 255)
        radius = 3 if feeder else 2
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)
    target = arena.targets[target_id]
    kinetics = arena.target_kinetics[target_id]
    tx, ground_y = _point(np.asarray((target.position[0], arena.ground_plane_y()), np.float64))
    visual_position = target.position.copy()
    if not step.attached:
        visual_position[1] -= kinetics.height
    _, ty = _point(visual_position)
    radius = max(3, int(round(target.radius * SCALE)))
    shadow_width = max(3, int(round(radius * (1.15 + min(kinetics.height, 12.0) * .025))))
    shadow_height = max(1, int(round(radius * .30)))
    draw.ellipse((tx - shadow_width, ground_y - shadow_height, tx + shadow_width, ground_y + shadow_height), fill=(1, 4, 5, 220), outline=(31, 82, 86, 220))
    if arena.acquisition_strategy() == "phase_tractor" and kinetics.height > .05:
        feeder = arena.posed_feeder_points().mean(axis=0)
        fx, fy = _point(feeder)
        for band, color in ((-1.0, (128, 76, 255, 90)), (0.0, (58, 230, 255, 155)), (1.0, (255, 76, 208, 90))):
            points = []
            for unit in np.linspace(0, 1, 18):
                px = fx + (tx - fx) * unit
                py = fy + (ty - fy) * unit + np.sin(np.pi * unit) * band * 4
                points.append((int(round(px)), int(round(py))))
            draw.line(points, fill=color, width=1)
    if kinetics.impact_mode == "roll":
        cosine, sine = np.cos(kinetics.angle), np.sin(kinetics.angle)
        corners = []
        for local_x, local_y in ((-radius, -radius), (radius, -radius), (radius, radius), (-radius, radius)):
            corners.append((int(round(tx + local_x * cosine - local_y * sine)), int(round(ty + local_x * sine + local_y * cosine))))
        draw.polygon(corners, fill=FOOD, outline=(224, 255, 175, 255))
        spoke = (int(round(tx + radius * cosine)), int(round(ty + radius * sine)))
        draw.line((tx, ty, *spoke), fill=(30, 90, 96, 255), width=1)
    elif kinetics.impact_mode == "thud":
        draw.ellipse((tx - radius, ty - max(2, radius - 1), tx + radius, ty + max(2, radius - 1)), fill=FOOD, outline=(224, 255, 175, 255), width=1)
    else:
        draw.ellipse((tx - radius, ty - radius, tx + radius, ty + radius), fill=FOOD, outline=(224, 255, 175, 255), width=1)
    reserve = min(1.0, arena.feeding.reserve / arena.feeding.reserve_capacity)
    fullness = min(1.0, arena.feeding.fullness_seconds / arena.feeding.fullness_capacity_seconds)
    draw.text((14, HEIGHT - 36), f"FOOD {target.mass:4.2f}  INTAKE {arena.feeding.consumed_mass:4.2f}", fill=(188, 204, 214, 255))
    draw.rectangle((190, HEIGHT - 34, 350, HEIGHT - 28), outline=(54, 84, 91, 255))
    draw.rectangle((191, HEIGHT - 33, 191 + int(158 * reserve), HEIGHT - 29), fill=FOOD)
    draw.rectangle((190, HEIGHT - 22, 350, HEIGHT - 16), outline=(54, 84, 91, 255))
    draw.rectangle((191, HEIGHT - 21, 191 + int(158 * fullness), HEIGHT - 17), fill=INK)
    state = "DETACHED" if step.detached else "DAMAGED" if step.actuation < .35 else f"AIR {kinetics.height:3.1f}" if kinetics.height > .05 and not step.attached else "THROWN" if step.thrown else "FEEDING" if step.absorbed_mass > 0 else "GRASPED" if step.attached else "REACHING"
    draw.text((14, HEIGHT - 20), state, fill=HOT if state in {"THROWN", "FEEDING"} else INK)
    return image


def _encode(frames: list[Image.Image], destination: Path, fps: int = 20) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for deterministic GIF encoding")
    with tempfile.TemporaryDirectory(prefix="nullvector-grasper-") as temporary:
        root = Path(temporary)
        for index, frame in enumerate(frames):
            frame.save(root / f"frame_{index:04d}.png", optimize=False)
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(root / "frame_%04d.png"), "-filter_complex", "[0:v]split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3", "-loop", "0", str(destination)], check=True)


def _arena(genome_index: int, food_position: tuple[float, float], *, material: str = "biomass", impact_mode: str | None = None) -> tuple[NeuralManipulationArena, int]:
    arena = NeuralManipulationArena(develop(review_genomes()[genome_index]), device="cpu")
    arena.feeding.reserve = 0.0
    arena.feeding.fullness_seconds = 0.0
    target = FoodClump(np.asarray(food_position), np.zeros(2), 1.0, .55, 1.0, (1, 1, 1, 1, 1), material)
    return arena, arena.add_clump(target, cohesion=2.0, impact_mode=impact_mode)


def _ground_arena(genome_index: int, x: float, *, material: str = "biomass", impact_mode: str | None = None) -> tuple[NeuralManipulationArena, int]:
    arena, target_id = _arena(genome_index, (x, 0.0), material=material, impact_mode=impact_mode)
    arena.targets[target_id].position[1] = arena.ground_plane_y()
    return arena, target_id


def _feed_clip() -> list[Image.Image]:
    arena, target_id = _ground_arena(0, 5.5)
    frames: list[Image.Image] = []
    for tick in range(420):
        step = arena.step_family_acquisition(target_id, delta=.05)
        if tick % 3 == 0:
            frames.append(_frame(arena, target_id, "NEURAL GRASP + PHYSICAL FEEDER", "food must contact live mouth cells", step))
        if arena.feeding.consumed_mass >= .38:
            break
    return frames + frames[-12:]


def _throw_clip() -> list[Image.Image]:
    arena, target_id = _ground_arena(0, 5.5, material="phase", impact_mode="bounce")
    frames: list[Image.Image] = []
    attached = False
    for tick in range(360):
        goal = "throw" if attached and tick > 25 else "carry"
        arena.pose_for_acquisition(1.0, delta=.05)
        step = arena.step(target_id, goal=goal, delta=.05, throw_strength=1.0)
        attached |= step.attached
        if tick % 2 == 0:
            frames.append(_frame(arena, target_id, "NEURAL THROW + 2.5D BALLISTICS", "elevation, shadow, recoil, material impact", step))
        if step.thrown:
            for _ in range(150):
                target = arena.targets[target_id]
                arena.body.position += arena.body.velocity * .05
                arena.body.velocity *= np.exp(-.05 * 3.2)
                arena.integrate_free_target(target_id, .05)
                rest = np.asarray(arena.organism.genome.appendages[step.appendage].endpoint, np.float64)
                arena.articulation.solve(step.appendage, rest, .11)
                step = ManipulationStep(step.appendage, False, True, False, float(np.linalg.norm(target.position - arena.body.position)), False, 0.0, arena.feeding.reserve, arena.feeding.fullness_seconds)
                frames.append(_frame(arena, target_id, "NEURAL THROW + 2.5D BALLISTICS", "phase matter bounces; mineral rolls; biomass thuds", step))
            break
    return frames


def _impact_modes_clip() -> list[Image.Image]:
    modes = (("bounce", "phase"), ("roll", "mineral"), ("thud", "biomass"))
    entries = []
    for mode, material in modes:
        arena, target_id = _ground_arena(0, 5.5, material=material, impact_mode=mode)
        entries.append({"arena": arena, "target": target_id, "mode": mode, "attached": False, "released": False, "step": None})
    frames: list[Image.Image] = []
    free_frames = 0
    for tick in range(360):
        tiles = []
        for entry in entries:
            arena = entry["arena"]
            target_id = entry["target"]
            if not entry["released"]:
                goal = "throw" if entry["attached"] and tick > 25 else "carry"
                arena.pose_for_acquisition(1.0, delta=.05)
                step = arena.step(target_id, goal=goal, delta=.05, throw_strength=1.0)
                entry["attached"] = bool(entry["attached"] or step.attached)
                entry["released"] = bool(step.thrown)
            else:
                arena.integrate_free_target(target_id, .05)
                appendage = int(entry["step"].appendage)
                rest = np.asarray(arena.organism.genome.appendages[appendage].endpoint, np.float64)
                arena.articulation.solve(appendage, rest, .11)
                target = arena.targets[target_id]
                step = ManipulationStep(appendage, False, True, False, float(np.linalg.norm(target.position - arena.body.position)), False, 0.0, arena.feeding.reserve, arena.feeding.fullness_seconds)
            entry["step"] = step
            tile = _frame(arena, target_id, f"{entry['mode'].upper()} IMPACT", "same neural throw // different matter", step).resize((256, 192), Image.Resampling.NEAREST)
            tiles.append(tile)
        if all(bool(entry["released"]) for entry in entries):
            free_frames += 1
        if tick % 2 == 0:
            sheet = Image.new("RGBA", (768, 192), BG)
            for index, tile in enumerate(tiles):
                sheet.alpha_composite(tile, (index * 256, 0))
            frames.append(sheet)
        if free_frames >= 145:
            break
    return frames + frames[-10:]


def _feeder_ablation_clip() -> list[Image.Image]:
    arena, target_id = _ground_arena(2, 4.0)
    status = feeder_status(arena.living)
    arena.living.health[status.feeder_mask] = 0
    frames: list[Image.Image] = []
    for tick in range(250):
        step = arena.step_family_acquisition(target_id, delta=.05)
        if tick % 3 == 0:
            frames.append(_frame(arena, target_id, "SEVERED FEEDER FAILS CLOSED", "contact cannot bypass damaged organ route", step))
    return frames


def _severed_grasper_clip() -> list[Image.Image]:
    arena, target_id = _ground_arena(0, 5.5)
    frames: list[Image.Image] = []
    held_step = None
    for tick in range(360):
        arena.pose_for_acquisition(1.0, delta=.05)
        step = arena.step(target_id, goal="carry", delta=.05)
        if tick % 2 == 0:
            frames.append(_frame(arena, target_id, "TRUE SEVER // FREE LIMB", "bone bridge breaks; payload and arm fall", step))
        if step.attached:
            held_step = step
            break
    if held_step is None:
        raise RuntimeError("severed grasper showcase never acquired its payload")
    # Prevent the healthy peer arm from immediately hiding the injury by
    # taking over; this clip isolates the severed chain itself.
    for grasper in arena.grasper_indices():
        if grasper != held_step.appendage:
            arena.damage_appendage(grasper, remaining_health=.04)
    arena.sever_appendage(held_step.appendage, impulse=(.8, -1.4))
    for tick in range(150):
        arena._step_detached_limbs(.05)
        arena.integrate_free_target(target_id, .05)
        target = arena.targets[target_id]
        step = ManipulationStep(
            held_step.appendage, False, False, False,
            float(np.linalg.norm(target.position - arena.body.position)), False, 0.0,
            arena.feeding.reserve, arena.feeding.fullness_seconds, 0.0, True,
        )
        if tick % 2 == 0:
            frames.append(_frame(arena, target_id, "TRUE SEVER // FREE LIMB", "brief residual twitch, then inert tissue", step))
    return frames + frames[-12:]


def _damaged_grasper_clip() -> list[Image.Image]:
    arena, target_id = _ground_arena(0, 5.5)
    for grasper in arena.grasper_indices():
        arena.damage_appendage(grasper, remaining_health=.18)
    frames: list[Image.Image] = []
    for tick in range(220):
        arena.pose_for_acquisition(1.0, delta=.05)
        step = arena.step(target_id, goal="consume", delta=.05)
        if tick % 2 == 0:
            frames.append(_frame(arena, target_id, "DAMAGED MUSCLE + NERVE", "weak twitch; insufficient force to carry", step))
    return frames + frames[-12:]


def _five_family_clip() -> list[Image.Image]:
    positions = (5.5, 0.0, 4.0, 8.0, 8.5)
    materials = ("biomass", "biomass", "mineral", "phase", "charge")
    arenas = [_ground_arena(index, positions[family], material=materials[family]) for family, index in enumerate((0, 2, 4, 6, 8))]
    frames: list[Image.Image] = []
    for tick in range(420):
        tiles: list[Image.Image] = []
        complete = True
        for family, (arena, target_id) in enumerate(arenas):
            step = arena.step_family_acquisition(target_id, delta=.05)
            complete &= arena.feeding.consumed_mass >= .20
            tile = _frame(arena, target_id, FAMILIES[family].upper(), arena.acquisition_strategy().replace("_", " "), step).resize((256, 192), Image.Resampling.NEAREST)
            tiles.append(tile)
        if tick % 3 == 0:
            sheet = Image.new("RGBA", (768, 384), BG)
            for index, tile in enumerate(tiles):
                sheet.alpha_composite(tile, ((index % 3) * 256, (index // 3) * 192))
            draw = ImageDraw.Draw(sheet)
            draw.text((526, 246), "5 DIETS", fill=INK)
            draw.text((526, 263), "5 FEEDER TYPES", fill=FOOD)
            draw.text((526, 280), "1 SHARED NN", fill=HOT)
            frames.append(sheet)
        if complete:
            break
    return frames + frames[-10:]


def build_showcase(destination: Path) -> dict[str, object]:
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    clips = {
        "articulated_inertial_feeding_v6.gif": _feed_clip(),
        "articulated_ballistic_throw_v6.gif": _throw_clip(),
        "articulated_impact_modes_v6.gif": _impact_modes_clip(),
        "articulated_feeder_ablation_v6.gif": _feeder_ablation_clip(),
        "articulated_severed_grasper_v6.gif": _severed_grasper_clip(),
        "articulated_damaged_grasper_v6.gif": _damaged_grasper_clip(),
        "articulated_five_family_feeding_v6.gif": _five_family_clip(),
    }
    artifacts: dict[str, dict[str, object]] = {}
    for name, frames in clips.items():
        path = destination / name
        _encode(frames, path)
        artifacts[name] = {"sha256": _sha(path), "bytes": path.stat().st_size, "frames": len(frames)}
    report = {"format": "nullvector-neural-manipulation-showcase/5.0.0", "artifacts": artifacts}
    (destination / "showcase_report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/creature_stage_manipulation_v1/showcase"))
    args = parser.parse_args()
    print(json.dumps(build_showcase(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
