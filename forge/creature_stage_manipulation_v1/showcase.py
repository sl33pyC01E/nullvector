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
    tx, ty = _point(target.position)
    radius = max(3, int(round(target.radius * SCALE)))
    draw.ellipse((tx - radius, ty - radius, tx + radius, ty + radius), fill=FOOD, outline=(224, 255, 175, 255), width=1)
    chain = arena.articulation.chain(appendage) + arena.body.position
    ex, ey = _point(chain[-1])
    draw.ellipse((ex - 4, ey - 4, ex + 4, ey + 4), outline=INK, width=2)
    reserve = min(1.0, arena.feeding.reserve / arena.feeding.reserve_capacity)
    fullness = min(1.0, arena.feeding.fullness_seconds / arena.feeding.fullness_capacity_seconds)
    draw.text((14, HEIGHT - 36), f"FOOD {target.mass:4.2f}  INTAKE {arena.feeding.consumed_mass:4.2f}", fill=(188, 204, 214, 255))
    draw.rectangle((190, HEIGHT - 34, 350, HEIGHT - 28), outline=(54, 84, 91, 255))
    draw.rectangle((191, HEIGHT - 33, 191 + int(158 * reserve), HEIGHT - 29), fill=FOOD)
    draw.rectangle((190, HEIGHT - 22, 350, HEIGHT - 16), outline=(54, 84, 91, 255))
    draw.rectangle((191, HEIGHT - 21, 191 + int(158 * fullness), HEIGHT - 17), fill=INK)
    state = "THROWN" if step.thrown else "FEEDING" if step.absorbed_mass > 0 else "GRASPED" if step.attached else "REACHING"
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


def _arena(genome_index: int, food_position: tuple[float, float]) -> tuple[NeuralManipulationArena, int]:
    arena = NeuralManipulationArena(develop(review_genomes()[genome_index]), device="cpu")
    arena.feeding.reserve = 0.0
    arena.feeding.fullness_seconds = 0.0
    target = FoodClump(np.asarray(food_position), np.zeros(2), 1.0, .55, 1.0, (1, 1, 1, 1, 1))
    return arena, arena.add_clump(target, cohesion=2.0)


def _feed_clip() -> list[Image.Image]:
    arena, target_id = _arena(0, (12.0, 1.0))
    frames: list[Image.Image] = []
    for tick in range(420):
        step = arena.step(target_id, goal="consume", delta=.05)
        if tick % 3 == 0:
            frames.append(_frame(arena, target_id, "NEURAL GRASP + PHYSICAL FEEDER", "food must contact live mouth cells", step))
        if arena.feeding.consumed_mass >= .38:
            break
    return frames + frames[-12:]


def _throw_clip() -> list[Image.Image]:
    arena, target_id = _arena(0, (12.0, 0.0))
    frames: list[Image.Image] = []
    attached = False
    for tick in range(360):
        goal = "throw" if attached and tick > 25 else "carry"
        step = arena.step(target_id, goal=goal, delta=.05, throw_strength=1.0)
        attached |= step.attached
        if tick % 2 == 0:
            frames.append(_frame(arena, target_id, "NEURAL THROW + BODY RECOIL", "constraint, momentum, ground bracing", step))
        if step.thrown:
            for _ in range(80):
                target = arena.targets[target_id]
                arena.body.position += arena.body.velocity * .05
                target.position += target.velocity * .05
                arena.body.velocity *= np.exp(-.05 * 3.2)
                target.velocity *= np.exp(-.05 * (1.2 + .4 / max(target.mass, .1)))
                rest = np.asarray(arena.organism.genome.appendages[step.appendage].endpoint, np.float64)
                arena.articulation.solve(step.appendage, rest, .11)
                step = ManipulationStep(step.appendage, False, True, False, float(np.linalg.norm(target.position - arena.body.position)), False, 0.0, arena.feeding.reserve, arena.feeding.fullness_seconds)
                frames.append(_frame(arena, target_id, "NEURAL THROW + BODY RECOIL", "released target retains impulse", step))
            break
    return frames


def _blocked_clip() -> list[Image.Image]:
    arena, target_id = _arena(2, (10.0, 0.0))
    status = feeder_status(arena.living)
    arena.living.health[status.feeder_mask] = 0
    frames: list[Image.Image] = []
    for tick in range(250):
        step = arena.step(target_id, goal="consume", delta=.05)
        if tick % 3 == 0:
            frames.append(_frame(arena, target_id, "SEVERED FEEDER FAILS CLOSED", "contact cannot bypass damaged organ route", step))
    return frames


def build_showcase(destination: Path) -> dict[str, object]:
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    clips = {"articulated_neural_feeding_v2.gif": _feed_clip(), "articulated_neural_throw_v2.gif": _throw_clip(), "articulated_severed_feeder_v2.gif": _blocked_clip()}
    artifacts: dict[str, dict[str, object]] = {}
    for name, frames in clips.items():
        path = destination / name
        _encode(frames, path)
        artifacts[name] = {"sha256": _sha(path), "bytes": path.stat().st_size, "frames": len(frames)}
    report = {"format": "nullvector-neural-manipulation-showcase/1.0.0", "artifacts": artifacts}
    (destination / "showcase_report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/creature_stage_manipulation_v1/showcase"))
    args = parser.parse_args()
    print(json.dumps(build_showcase(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
