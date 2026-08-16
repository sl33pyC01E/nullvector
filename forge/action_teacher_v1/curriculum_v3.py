from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .contract import ACTIONS
from .recorder import TeacherTrajectoryRecorder, validate_trajectory
from .curriculum_v2 import _ability_actor, _entities
from ..config import PROJECT_ROOT
from ..nature_sim_v2.abilities import entity_abilities, use_ability
from ..nature_sim_v2.demo import NatureDemo
from ..nature_sim_v2.directed_evolution import evolution_offers, metamorphose
from ..nature_sim_v2.forecast_interventions import apply_intervention
from ..nature_sim_v2.social_actions import bond_nearby


TARGETED = {"impact", "scrape", "cut", "beam", "projectile"}


def spatial_schedule(repeats: int) -> tuple[str, ...]:
    if not 2 <= repeats <= 6:
        raise ValueError("spatial teacher repeats must be 2..6")
    return tuple(emitted for _ in range(repeats) for action in ACTIONS for emitted in ("none", action, "none"))


def _bootstrap_four_slot_actor(demo: NatureDemo) -> int:
    actor = next(item for item in demo.world.organisms.values() if item.alive and item.family == 1)
    for family in (2, 4):
        if len(entity_abilities(actor, equipment_damage=demo.adventure.bonus("damage"))) >= 4:
            break
        donor = next(item for item in demo.world.organisms.values() if item.alive and item.family == family)
        demo.world.graft_from(actor.entity_id, donor.entity_id, kind="locomotor")
    if len(entity_abilities(actor, equipment_damage=demo.adventure.bonus("damage"))) < 4:
        raise RuntimeError("spatial teacher could not construct a four-slot action actor")
    demo.runtime.forget(actor.entity_id)
    demo.visible_physics.states.pop(actor.entity_id, None)
    demo.selected = actor.entity_id
    return actor.entity_id


def _actor_for(demo: NatureDemo, action: str):
    entity, _, living = _entities(demo)
    if action.startswith("ability_"):
        index = ("ability_up", "ability_right", "ability_down", "ability_left").index(action)
        candidate = _ability_actor(demo, living, index)
        if candidate is not None:
            entity = candidate
    demo.selected = entity.entity_id
    return entity, living


def _stage(demo: NatureDemo, action: str, event_index: int):
    actor, living = _actor_for(demo, action)
    targets = [item for item in living if item.entity_id != actor.entity_id]
    target = targets[event_index % len(targets)] if targets else actor
    angle = (event_index * 2.399963229728653 + actor.family * 0.71) % math.tau
    radius = (3.0, 6.0, 10.0, 14.0)[(event_index // len(ACTIONS)) % 4]
    if action == "bond":
        kin = next((item for item in targets if item.family == actor.family), target)
        target = kin
        radius = 0.65
    if action in ("heal", "craft", "build", "graft_organ", "graft_locomotor", "intervention", "metamorphosis"):
        radius = 0.0
        target = actor
    if action in ("trade", "service") and demo.society.settlements:
        actor.position = np.asarray(next(iter(demo.society.settlements.values())).center, dtype=np.float64)
        target = actor
        radius = 0.0
    if action == "interact":
        actor.position = demo.adventure.sites[event_index % len(demo.adventure.sites)].position.copy()
        target = actor
        radius = 0.0
    if target.entity_id != actor.entity_id:
        target.position = (actor.position + np.asarray((math.cos(angle), math.sin(angle))) * radius) % demo.world.size
        target.velocity *= 0
    actor.velocity *= 0
    demo.camera = actor.position.copy()
    demo.teacher_aim_override = target.position.copy()
    return actor.entity_id, target.entity_id


def _resolve(demo: NatureDemo, action: str, actor_id: int, target_id: int, event_index: int) -> bool:
    world = demo.world
    actor = world.organisms.get(actor_id)
    target = world.organisms.get(target_id)
    if actor is None or not actor.alive:
        return False
    if target is None or not target.alive:
        target = actor
    demo.selected = actor.entity_id
    demo.camera = actor.position.copy()
    demo.teacher_aim_override = target.position.copy()
    if action in ("none", "inspect"):
        return True
    if action == "impact":
        target.body.impact((0, 0), 3, 0.28)
    elif action == "heal":
        actor.body.impact((0, 0), 2, 0.13)
        actor.body.heal((0, 0), 8, 0.24)
        demo.teacher_aim_override = actor.position.copy()
    elif action == "scrape":
        target.body.impact((0, 0), 2.2, 0.55)
    elif action == "cut":
        angle = event_index * 1.61803398875
        direction = np.asarray((math.cos(angle), math.sin(angle))) * 13
        target.body.cut(tuple(-direction), tuple(direction), width=0.72)
    elif action == "beam":
        world.fire_beam(actor.entity_id, tuple(target.position), energy=5.0, width=0.68)
    elif action == "projectile":
        world.fire_projectile(actor.entity_id, tuple(target.position), speed=18, energy=1.6)
    elif action == "interact":
        demo.adventure.interact(world, actor)
    elif action == "build":
        demo.adventure.inventory.update({"rock": 8, "metal": 8, "biomass": 8})
        demo.adventure.build(world, actor)
    elif action == "craft":
        demo.adventure.inventory.update({"rock": 8, "metal": 8, "biomass": 8, "crystal": 8})
        demo.adventure.craft_selected()
    elif action == "bond":
        target.position = (actor.position + np.asarray((0.45, 0.15))) % world.size
        bond_nearby(world, actor)
    elif action in ("graft_organ", "graft_locomotor"):
        donors = [item for item in world.organisms.values() if item.alive and item.entity_id != actor.entity_id and item.family != actor.family]
        if not donors:
            return False
        donor = donors[event_index % len(donors)]
        world.graft_from(actor.entity_id, donor.entity_id, kind="organ" if action == "graft_organ" else "locomotor")
        demo.runtime.forget(actor.entity_id)
        demo.visible_physics.states.pop(actor.entity_id, None)
    elif action.startswith("ability_"):
        index = ("ability_up", "ability_right", "ability_down", "ability_left").index(action)
        abilities = entity_abilities(actor, equipment_damage=demo.adventure.bonus("damage"))
        if index >= len(abilities):
            return False
        use_ability(world, actor, abilities[index], tuple(target.position), power=0.9)
    elif action == "intervention":
        demo.adventure.inventory.update({"rock": 8, "metal": 8, "biomass": 8, "crystal": 8, "water": 8, "knowledge": 8})
        offer = demo.intervention_offers[event_index % len(demo.intervention_offers)]
        apply_intervention(world, demo.adventure, actor, offer, forecast_event=demo.timeline_forecast.event)
    elif action == "trade":
        if not demo.society.settlements:
            return False
        settlement = next(iter(demo.society.settlements.values()))
        demo.adventure.inventory["biomass"] += 0.2
        settlement.stockpiles["biomass"] = settlement.stockpiles.get("biomass", 0) + 0.2
    elif action == "service":
        if not demo.society.settlements:
            return False
        actor.body.impact((0, 0), 2, 0.12)
        actor.body.heal((0, 0), 8, 0.18)
    elif action == "metamorphosis":
        offer = evolution_offers(actor.genome, epoch=event_index)[event_index % 3]
        metamorphose(actor, offer, seed=(world.seed ^ event_index * 7919) & 0x7FFF_FFFF_FFFF_FFFF)
        demo.runtime.forget(actor.entity_id)
        demo.behavior.cache.pop(actor.entity_id, None)
        demo.visible_physics.states.pop(actor.entity_id, None)
    else:
        return False
    return True


def _emit(demo: NatureDemo, action: str, manual: np.ndarray, counts: dict[str, int]) -> None:
    demo.manual = manual.astype(np.float32)
    demo.action_latch = action
    counts[action] += 1
    demo.update(1 / 30)
    actor = demo.world.organisms.get(demo.selected)
    if actor is not None:
        demo.camera = actor.position.copy()
    demo.draw()


def generate(*, root: Path, session_id: str, repeats: int = 4, seed: int = 0x5350415449414C33, device: str = "cuda"):
    schedule = spatial_schedule(repeats)
    demo = NatureDemo(seed=seed, device=device, showcase=True)
    four_slot_actor = _bootstrap_four_slot_actor(demo)
    frames = len(schedule)
    demo.trajectory = TeacherTrajectoryRecorder(root, stride=1, max_frames=frames + 8)
    demo.trajectory.start(session_id, world_seed=demo.world.seed, tick=demo.world.tick_index)
    counts = {name: 0 for name in ACTIONS}
    failures = {name: 0 for name in ACTIONS}
    aim_bins = [0, 0, 0, 0]
    event_index = 0
    for repeat in range(repeats):
        for action in ACTIONS:
            actor_id, target_id = _stage(demo, action, event_index)
            phase = event_index * 0.61803398875
            manual = np.asarray((math.cos(phase), math.sin(phase * 0.73))) * (0.18 + repeat * 0.16)
            _emit(demo, "none", manual * 0.2, counts)
            actor = demo.world.organisms.get(actor_id)
            target = demo.world.organisms.get(target_id)
            if actor is not None and target is not None:
                magnitude = float(np.linalg.norm(demo.world._delta(actor.position, target.position))) / (demo.world.size * 0.5)
                aim_bins[min(3, int(magnitude * 8))] += 1
            try:
                success = _resolve(demo, action, actor_id, target_id, event_index)
            except (ValueError, RuntimeError, IndexError):
                success = False
            emitted = action if success else "none"
            if not success:
                failures[action] += 1
            _emit(demo, emitted, manual, counts)
            _emit(demo, "none", manual * 0.1, counts)
            event_index += 1
    destination = demo.trajectory.finish()
    demo.neural_executor.shutdown(wait=True, cancel_futures=True)
    demo.pg.quit()
    manifest = validate_trajectory(destination)
    missing = [name for name in ACTIONS if name != "none" and counts[name] < repeats]
    if missing:
        raise RuntimeError("spatial teacher missed actions: " + ",".join(missing))
    report = {
        "format": "nullvector-action-teacher-spatial-curriculum/3.0.0",
        "session": session_id,
        "frames": manifest["frames"],
        "repeats": repeats,
        "seed": seed,
        "actions": counts,
        "failures": failures,
        "aim_distance_bins": aim_bins,
        "protocol": "setup-none -> spatial-action -> settle-none; actor camera-centered",
        "four_slot_actor": four_slot_actor,
        "trajectory_manifest_sha256": manifest["manifest_sha256"],
        "trajectory_arrays_sha256": manifest["arrays_sha256"],
    }
    (destination / "curriculum_v3_report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "outputs/action_teacher_v1")
    parser.add_argument("--session", required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x5350415449414C33)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(generate(root=args.root, session_id=args.session, repeats=args.repeats, seed=args.seed, device=args.device), indent=2))


if __name__ == "__main__":
    main()
