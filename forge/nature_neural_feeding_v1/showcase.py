from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..creature_stage_developmental.contract import FAMILIES
from ..creature_stage_manipulation_v1.showcase import _encode
from ..creature_stage_neural_grasper_v1.feeding import feeder_status
from ..nature_sim_v2 import NatureWorld, founder_genomes
from .system import NatureNeuralFeedingSystem


SIZE = 40
PIXELS = 640
SCALE = PIXELS / SIZE
FAMILY = ((46, 224, 255), (255, 91, 165), (148, 255, 70), (187, 122, 255), (255, 184, 62))
MATERIAL = {"flora": (145, 255, 74), "biomass": (246, 86, 119), "mineral": (181, 194, 205), "charge": (57, 223, 255), "phase": (191, 114, 255)}


def _world_point(position) -> tuple[int, int]:
    value = np.asarray(position) % SIZE
    return int(round(value[0] * SCALE)), int(round(value[1] * SCALE))


def _render(world: NatureWorld, system: NatureNeuralFeedingSystem) -> Image.Image:
    image = Image.new("RGBA", (PIXELS, PIXELS + 72), (3, 10, 14, 255))
    draw = ImageDraw.Draw(image)
    for coordinate in range(0, PIXELS + 1, int(SCALE * 4)):
        draw.line((coordinate, 0, coordinate, PIXELS), fill=(10, 35, 41, 255))
        draw.line((0, coordinate, PIXELS, coordinate), fill=(10, 35, 41, 255))
    for clump in sorted(system.clumps.values(), key=lambda item: item.clump_id):
        x, y = _world_point(clump.food.position)
        radius = max(3, int(clump.food.radius * SCALE))
        color = MATERIAL[clump.food.material]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 235), outline=(235, 255, 245, 255), width=1)
    for entity in sorted(world.organisms.values(), key=lambda item: (item.position[1], item.entity_id)):
        if not entity.alive:
            continue
        organism = entity.body.organism
        state = system.entities.get(entity.entity_id)
        points = organism.cell_xy.astype(np.float32) if state is None else state.articulation.cells()
        feeder = feeder_status(entity.body).feeder_mask
        active_appendage = None if state is None else state.grasp_appendage
        for index, local in enumerate(points):
            position = entity.position + local * .115
            x, y = _world_point(position)
            selected = active_appendage is not None and int(organism.appendage_index[index]) == active_appendage
            color = (123, 255, 200) if selected else (255, 78, 130) if feeder[index] else FAMILY[entity.family]
            draw.rectangle((x - 2, y - 2, x + 2, y + 2), fill=(*color, 245))
        x, y = _world_point(entity.position)
        draw.text((x - 22, y + 28), FAMILIES[entity.family][:5].upper(), fill=FAMILY[entity.family])
        draw.rectangle((x - 20, y + 40, x + 20, y + 43), fill=(17, 34, 39, 255))
        draw.rectangle((x - 20, y + 40, x - 20 + int(40 * np.clip(entity.energy / 1.2, 0, 1)), y + 43), fill=FAMILY[entity.family])
    draw.rectangle((0, PIXELS, PIXELS, PIXELS + 72), fill=(2, 8, 12, 255))
    live = [entity for entity in world.organisms.values() if entity.alive]
    draw.text((18, PIXELS + 12), "PERSISTENT NEURAL FEEDING ECOLOGY", fill=(46, 225, 255, 255))
    draw.text((18, PIXELS + 31), f"TICK {world.tick_index:04d}  LIVE {len(live):02d}  GRASPS {system.grasps:03d}  ABSORBED {system.absorbed_mass:4.2f}", fill=(186, 209, 217, 255))
    draw.text((18, PIXELS + 49), "TANGIBLE MATTER // CELL APPENDAGES // LIVE FEEDERS // STORED RESERVE", fill=(139, 255, 76, 255))
    return image


def build_showcase(destination: Path) -> dict[str, object]:
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    system = NatureNeuralFeedingSystem(seed=0x45434F, device="cpu")
    world = NatureWorld(seed=0x454350, size=SIZE, max_population=30, feeding_system=system)
    positions = ((8, 9), (20, 8), (32, 10), (12, 27), (29, 27))
    materials = ("biomass", "flora", "flora", "phase", "mineral")
    for family, (genome, position, material) in enumerate(zip(founder_genomes(variants_per_family=1), positions, materials)):
        entity_id = world.add_organism(genome, position, energy=.20)
        world.organisms[entity_id].reserve = 0
        system.add_clump((position[0] + 1.1, position[1]), material=material, mass=1.0, source=f"showcase:{family}")
    frames = [_render(world, system)]
    for tick in range(600):
        world.step(.05, publish=False)
        if tick % 4 == 0:
            frames.append(_render(world, system))
    path = destination / "persistent_neural_feeding_ecology_v1.gif"
    _encode(frames, path, fps=20)
    report = {"format": "nullvector-nature-neural-feeding-showcase/1.0.0", "frames": len(frames), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "world_sha256": world.snapshot().semantic_sha256, "population": world.snapshot().population, "deaths": world.deaths, "absorbed_mass": system.absorbed_mass, "grasps": system.grasps}
    (destination / "showcase_report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("outputs/nature_neural_feeding_v1/showcase")); args = parser.parse_args()
    print(json.dumps(build_showcase(args.output), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
