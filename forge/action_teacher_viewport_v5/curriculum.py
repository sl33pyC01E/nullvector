from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from ..action_teacher_natural_v4.curriculum import _nearest_target
from ..action_teacher_v1.contract import ACTIONS
from ..action_teacher_v1.curriculum_v3 import _resolve
from ..action_teacher_v2 import extract_actor_features
from ..maps import MapConfig, generate_map
from ..maps.io import array_digest
from ..maps.model import Hazard, Terrain
from ..maps.validate import assert_valid
from ..nature_counterfactual_nn import ACTIONS as COUNTERFACTUAL_ACTIONS
from ..nature_sim_v2.demo import NatureDemo, STUDENT_VIEW_HIDDEN
from ..nature_timeline_nn import EVENTS as TIMELINE_EVENTS, extract_world_features
from ..powder_world_v1.contract import MATERIALS
from .contract import DEFAULT_ROOT
from .aesthetic import render_teacher_map_frames
from .recorder import WholeViewportRecorder, validate_trajectory
from .state import extract_organism_tokens, extract_posed_actor_field, extract_spatial_state


SCENARIOS = ("journey", "migration", "feeding", "predation", "injury", "settlement_pan")
THEMES = ("arena", "rooms", "caves", "archipelago", "garden", "anomaly")
ZOOM = {"journey": 38.0, "migration": 27.0, "feeding": 42.0, "predation": 35.0, "injury": 44.0, "settlement_pan": 23.0}


def _record(demo, recorder, action, step):
    if not demo.student_view or any(getattr(demo, name) for name in STUDENT_VIEW_HIDDEN):
        raise RuntimeError("whole-viewport overlay state drifted")
    actor = demo.world.organisms.get(demo.selected)
    if actor is None or not actor.alive:
        raise RuntimeError("whole-viewport fixed actor died")
    control = demo._neural_control().copy()
    frame = demo.capture_clean_target()
    forecast = demo.timeline_forecast
    timeline = np.asarray((forecast.confidence, forecast.population_delta, forecast.resource_delta), np.float32)
    counterfactual = np.asarray([
        [demo.counterfactuals[name].benefit, demo.counterfactuals[name].risk, demo.counterfactuals[name].population_delta, demo.counterfactuals[name].resource_delta]
        for name in COUNTERFACTUAL_ACTIONS
    ], np.float32)
    span = float(demo._world_viewport_rect().width / demo.zoom)
    posed = {item.entity_id: demo._posed_points(item) for item in demo.world.organisms.values() if item.alive}
    organisms, organism_mask = extract_organism_tokens(
        demo.world, camera=demo.camera, span=span, selected=demo.selected, posed_points=posed,
    )
    recorder.append(
        frame=frame,
        spatial=extract_spatial_state(
            demo.world, demo.society, demo.feeding, camera=demo.camera, span=span,
            selected=demo.selected, topology=demo.teacher_topology,
        ), organisms=organisms, organism_mask=organism_mask,
        state=extract_world_features(demo.world, demo.society),
        actor_state=extract_actor_features(demo.world, demo.selected),
        actor_field=extract_posed_actor_field(actor, posed[actor.entity_id]),
        visibility=demo.teacher_visibility.copy(), memory=demo.teacher_memory.copy(),
        control=control, action=action, selected=demo.selected,
        timeline_event=TIMELINE_EVENTS.index(forecast.event), timeline=timeline,
        counterfactual=counterfactual, tick=recorder.start_tick + step + 1, episode_step=step,
    )


def _stable_structure(mask: np.ndarray) -> np.ndarray:
    result = np.asarray(mask, np.bool_).copy()
    while result.any():
        neighbors = sum(np.roll(np.roll(result, dy, 0), dx, 1) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        reduced = result & (neighbors >= 2)
        if np.array_equal(reduced, result):
            return result
        result = reduced
    return result


def _apply_topology(demo: NatureDemo, *, seed: int, scenario: str):
    theme = THEMES[SCENARIOS.index(scenario)]
    topology = generate_map(seed ^ 0x4D41505445414348, theme, MapConfig(width=64, height=64, objective_count=4, spawn_count=32))
    assert_valid(topology)
    world = demo.world
    solid = _stable_structure(topology.walkability == 0)
    if solid.any():
        world.materials.add_structure(solid, structure_id=70_000, material="metal" if theme in ("arena", "anomaly") else "rock")
    free = world.materials.structure_id == 0
    for terrain, material, mass in (
        (Terrain.WATER, "water", .72), (Terrain.GROWTH, "biomass", .24),
        (Terrain.CRYSTAL, "crystal", .78), (Terrain.SAND, "soil", .48),
    ):
        chosen = (topology.terrain == int(terrain)) & free
        world.materials.material[chosen] = MATERIALS.index(material)
        world.materials.mass[chosen] = mass
    for hazard, material in ((Hazard.LAVA, "fire"), (Hazard.SPORES, "smoke")):
        chosen = (topology.hazard == int(hazard)) & free
        world.materials.material[chosen] = MATERIALS.index(material)
        world.materials.mass[chosen] = .32
    world.fields[0] = np.maximum(world.fields[0], (topology.terrain == int(Terrain.WATER)) * .88)
    world.fields[2] = np.maximum(world.fields[2], (topology.terrain == int(Terrain.CRYSTAL)) * .92)
    world.fields[3] = np.maximum(world.fields[3], (topology.hazard == int(Hazard.ARC)) * .82)
    world.fields[4] = np.maximum(world.fields[4], (topology.hazard == int(Hazard.LASER)) * .62)
    world.fields[6] = np.maximum(world.fields[6], (topology.hazard == int(Hazard.LAVA)) * .92)
    world.fields[7] = np.maximum(world.fields[7], (topology.hazard == int(Hazard.SPORES)) * .74)
    world.fields[8] = np.maximum(world.fields[8], (topology.terrain == int(Terrain.GROWTH)) * .92)

    rng = np.random.default_rng(seed ^ 0x535041574E53)
    candidates = list(topology.spawns) + [topology.start, *topology.objectives, topology.exit]
    walkable = np.argwhere(topology.walkability > 0)
    rng.shuffle(walkable)
    candidates.extend((int(x), int(y)) for y, x in walkable)
    locations, used = [], set()
    for point in candidates:
        if point not in used:
            used.add(point)
            locations.append(np.asarray((point[0] + .5, point[1] + .5), np.float64))
    for entity, position in zip(sorted(world.organisms.values(), key=lambda item: item.entity_id), locations):
        entity.position = position.copy()
        entity.velocity *= 0
    for settlement, position in zip(demo.society.settlements.values(), [topology.objectives[0], topology.objectives[-1], topology.exit]):
        settlement.center = (float(position[0]) + .5, float(position[1]) + .5)
    demo.teacher_topology = topology
    demo.teacher_map_art_frames = render_teacher_map_frames(topology)
    demo.teacher_hide_topology_structures = True
    demo.field_cache = None
    demo.field_cache_key = None
    demo.material_cache = None
    demo.material_cache_key = None
    return topology, rng


def _seed_food(demo: NatureDemo, topology, rng, *, scenario: str, actor_family: int):
    count = {"journey": 12, "migration": 16, "feeding": 40, "predation": 14, "injury": 18, "settlement_pan": 20}[scenario]
    materials = ("flora", "biomass", "mineral", "charge", "phase")
    walkable = np.argwhere(topology.walkability > 0)
    if scenario == "feeding":
        actor = demo.world.organisms[demo.selected]
        distances = np.linalg.norm(np.stack((walkable[:, 1], walkable[:, 0]), 1) - actor.position, axis=1)
        local = walkable[distances < 13]
        if len(local):
            walkable = local
    chosen = rng.choice(len(walkable), size=count, replace=len(walkable) < count)
    for ordinal, index in enumerate(chosen):
        y, x = walkable[index]
        demo.feeding.add_clump(
            np.asarray((x, y), np.float64) + rng.uniform(.15, .85, 2),
            material=materials[(ordinal + actor_family) % len(materials)],
            mass=.35 + .12 * (ordinal % 5), source="viewport_macro_teacher",
        )


def _route(topology) -> list[tuple[int, int]]:
    start, goal = topology.start, topology.exit
    queue = deque([start])
    parent = {start: None}
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            break
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            point = (nx, ny)
            if 0 <= nx < topology.config.width and 0 <= ny < topology.config.height and topology.walkability[ny, nx] and point not in parent:
                parent[point] = (x, y)
                queue.append(point)
    if goal not in parent:
        raise RuntimeError("validated topology lost its start-exit route")
    result, current = [], goal
    while current is not None:
        result.append(current)
        current = parent[current]
    return list(reversed(result))


def _arrange_scenario(demo: NatureDemo, actor, topology, scenario: str) -> tuple[int, ...]:
    path = _route(topology)
    actor.position = np.asarray(path[0], np.float64) + .5
    actor.velocity *= 0
    cohort: list[int] = [actor.entity_id]
    others = [item for item in sorted(demo.world.organisms.values(), key=lambda item: item.entity_id) if item.entity_id != actor.entity_id]
    if scenario == "migration":
        for index, entity in enumerate(others[:7], 1):
            entity.position = np.asarray(path[min(index * 2, len(path) - 1)], np.float64) + .5
            entity.velocity *= 0
            cohort.append(entity.entity_id)
    elif scenario == "predation":
        prey = next((item for item in others if item.family != actor.family), others[0])
        prey.position = np.asarray(path[min(9, len(path) - 1)], np.float64) + .5
        prey.velocity *= 0
        cohort.append(prey.entity_id)
    elif scenario == "settlement_pan":
        sockets = [*topology.spawns, *topology.objectives]
        for entity, point in zip(others[:6], sockets):
            entity.position = np.asarray(point, np.float64) + .5
            entity.velocity *= 0
            cohort.append(entity.entity_id)
    return tuple(cohort)


def _target_direction(demo: NatureDemo, actor, topology, scenario: str, step: int) -> np.ndarray:
    route = (topology.start, *topology.objectives, topology.exit)
    if scenario == "feeding" and demo.feeding.clumps:
        target = min(demo.feeding.clumps.values(), key=lambda item: np.linalg.norm(demo.world._delta(actor.position, item.food.position))).food.position
    elif scenario == "predation":
        candidates = [item for item in demo.world.organisms.values() if item.alive and item.entity_id != actor.entity_id and item.family != actor.family]
        target = min(candidates, key=lambda item: np.linalg.norm(demo.world._delta(actor.position, item.position))).position if candidates else np.asarray(topology.exit)
    elif scenario == "settlement_pan" and demo.society.settlements:
        target = np.asarray(next(iter(demo.society.settlements.values())).center)
    elif scenario == "migration":
        target = np.asarray(topology.exit, np.float64) + .5
    else:
        target = np.asarray(route[(step // 96) % len(route)], np.float64) + .5
    direction = demo.world._delta(actor.position, target)
    norm = float(np.linalg.norm(direction))
    if norm > 1e-6:
        direction /= norm
    wander = np.asarray((math.sin(step * .037 + actor.family), math.sin(step * .023 + actor.entity_id * .31))) * .22
    return (direction + wander).astype(np.float32)


def generate(*, root: Path = DEFAULT_ROOT, session_id: str, frames=240, seed=0x56494557504F5254, device="cpu", actor_family=1, scenario="journey"):
    if not 64 <= frames <= 3600:
        raise ValueError("whole-viewport duration drifted")
    if actor_family not in range(5) or scenario not in SCENARIOS:
        raise ValueError("whole-viewport stratum drifted")
    demo = NatureDemo(seed=seed, device=device, showcase=True)
    demo._set_student_view(True)
    topology, rng = _apply_topology(demo, seed=seed, scenario=scenario)
    actor = next(item for item in demo.world.organisms.values() if item.alive and item.family == actor_family)
    actor_id = actor.entity_id
    demo.selected = actor_id
    actor.energy, actor.reserve = 1.5, 1.0
    actor.reproduction_cooldown = max(actor.reproduction_cooldown, frames / 30)
    demo.zoom = ZOOM[scenario]
    demo.camera = actor.position.copy()
    cohort = _arrange_scenario(demo, actor, topology, scenario)
    demo.camera = actor.position.copy()
    _seed_food(demo, topology, rng, scenario=scenario, actor_family=actor_family)
    if scenario == "injury":
        actor.body.impact((0, 0), 4.5, .18)

    recorder = WholeViewportRecorder(root, max_frames=frames + 8)
    recorder.start(session_id, world_seed=demo.world.seed, tick=demo.world.tick_index)
    actions = tuple(name for name in ACTIONS if name != "none")
    counts, failures = {name: 0 for name in ACTIONS}, {name: 0 for name in ACTIONS}
    camera_start, pan_target = demo.camera.copy(), np.asarray(topology.exit, np.float64) + .5
    try:
        warmup, action_period = 12, 9
        for step in range(frames):
            actor = demo.world.organisms.get(actor_id)
            if actor is None or not actor.alive:
                raise RuntimeError("whole-viewport actor died")
            demo.manual = _target_direction(demo, actor, topology, scenario, step)
            if scenario == "migration":
                destination = np.asarray(topology.exit, np.float64) + .5
                for entity_id in cohort[1:]:
                    member = demo.world.organisms.get(entity_id)
                    if member is None or not member.alive:
                        continue
                    direction = demo.world._delta(member.position, destination)
                    norm = float(np.linalg.norm(direction))
                    if norm > 1e-6:
                        member.velocity += direction / norm * (1 / 30) * .85
            emitted = "none"
            if step >= warmup and (step - warmup) % action_period == 0:
                action = actions[((step - warmup) // action_period) % len(actions)]
                target = _nearest_target(demo, actor)
                demo.teacher_aim_override = target.position.copy()
                try:
                    success = _resolve(demo, action, actor_id, target.entity_id, (step - warmup) // action_period)
                except (ValueError, RuntimeError, IndexError):
                    success = False
                if success:
                    emitted = action
                else:
                    failures[action] += 1
            demo.action_latch = emitted
            counts[emitted] += 1
            demo.update(1 / 30)
            demo.selected = actor_id
            if scenario == "settlement_pan":
                progress = step / max(frames - 1, 1)
                eased = progress * progress * (3 - 2 * progress)
                demo.camera = (camera_start + demo.world._delta(camera_start, pan_target) * eased) % demo.world.size
            elif scenario == "migration":
                demo.camera = (demo.camera + demo.world._delta(demo.camera, actor.position) * .065) % demo.world.size
            else:
                demo.camera = actor.position.copy()
            _record(demo, recorder, emitted, step)
            demo.action_latch = "none"
        destination = recorder.finish()
    finally:
        demo.neural_executor.shutdown(wait=True, cancel_futures=True)
        demo.sprite_executor.shutdown(wait=True, cancel_futures=True)
        demo.pg.quit()
    manifest = validate_trajectory(destination)
    report = {
        "format": "nullvector-whole-viewport-curriculum/5.0.0", "session": session_id,
        "frames": manifest["frames"], "seed": seed, "fixed_actor": actor_id,
        "actor_family": actor_family, "scenario": scenario, "map_theme": topology.theme,
        "map_id": topology.map_id, "map_arrays_sha256": array_digest(topology.arrays()),
        "actions": counts, "failures": failures, "warmup": warmup, "action_period": action_period,
        "neural_grasper_fallbacks": demo.feeding.controller_fallbacks,
        "contract": "full map+ecology numeric state -> visual latents -> one full-frame VAE decode; scaffold frames are teacher targets only",
        "trajectory_manifest_sha256": manifest["manifest_sha256"], "trajectory_arrays_sha256": manifest["arrays_sha256"],
    }
    (destination / "curriculum_report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report
