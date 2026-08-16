from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid
import zipfile

import numpy as np

from ..qud_items_v1 import Artifact
from ..qud_quests_v1 import QuestJournal
from ..qud_quests_v1.contract import QuestEntry
from ..qud_society_v1 import SocietyLayer
from ..qud_society_v1.contract import Activity, BuildingPlan, FactionState, HistoryEvent, SettlementState
from ..nature_world_scale_v1 import InfiniteNatureAtlas, RegionKey
from ..nature_world_scale_v1.atlas import RegionSummary
from .adventure import AdventureState, ObjectiveState, WorldSite
from .savegame import load_world, save_world


SESSION_FORMAT = "nullvector-living-nature-session/1.0.0"
MAX_SESSION_BYTES = 896 * 1024**2


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_plain(item) for item in value]
    return value


def _adventure_payload(adventure: AdventureState) -> dict:
    return {
        "seed": adventure.seed,
        "size": adventure.size,
        "rng": _plain(adventure.rng.bit_generator.state),
        "inventory": adventure.inventory,
        "discoveries": sorted(adventure.discoveries),
        "score": adventure.score,
        "last_event": adventure.last_event,
        "recipe_index": adventure.recipe_index,
        "craft_count": adventure.craft_count,
        "sites": [_plain(asdict(item)) for item in adventure.sites],
        "objectives": [_plain(asdict(item)) for item in adventure.objectives],
        "artifacts": [_plain(asdict(item)) for item in adventure.artifacts],
        "equipped": adventure.equipped,
        "buildings": [_plain(asdict(item)) for item in adventure.buildings],
    }


def _restore_adventure(payload: dict) -> AdventureState:
    adventure = AdventureState(seed=int(payload["seed"]), size=int(payload["size"]))
    adventure.rng.bit_generator.state = payload["rng"]
    adventure.inventory = {str(name): float(value) for name, value in payload["inventory"].items()}
    adventure.discoveries = set(payload["discoveries"])
    adventure.score = int(payload["score"])
    adventure.last_event = int(payload["last_event"])
    adventure.recipe_index = int(payload["recipe_index"])
    adventure.craft_count = int(payload["craft_count"])
    adventure.sites = [WorldSite(item["site_id"], item["kind"], np.asarray(item["position"], np.float64), float(item["richness"]), bool(item["discovered"])) for item in payload["sites"]]
    adventure.objectives = [ObjectiveState(**item) for item in payload["objectives"]]
    adventure.artifacts = [Artifact(**{**item, "components": tuple(item["components"]), "effects": tuple(tuple(value) for value in item["effects"])}) for item in payload["artifacts"]]
    adventure.equipped = {str(slot): str(item_id) for slot, item_id in payload["equipped"].items()}
    adventure.buildings = [BuildingPlan(**{**item, "origin": tuple(item["origin"]), "cells": tuple(tuple(value) for value in item["cells"]), "entrances": tuple(tuple(value) for value in item["entrances"])}) for item in payload["buildings"]]
    return adventure


def _society_payload(society: SocietyLayer) -> dict:
    return {
        "seed": society.seed,
        "tick": society.tick,
        "rng": _plain(society.rng.bit_generator.state),
        "factions": [_plain(asdict(item)) for item in sorted(society.factions.values(), key=lambda value: value.faction_id)],
        "settlements": [_plain(asdict(item)) for item in sorted(society.settlements.values(), key=lambda value: value.settlement_id)],
        "history": [_plain(asdict(item)) for item in society.history],
        "activities": [_plain(asdict(item)) for item in sorted(society.activities.values(), key=lambda value: value.activity_id)],
        "assignments": society.assignments,
        "materialized_buildings": sorted(society.materialized_buildings),
    }


def _building(item: dict) -> BuildingPlan:
    return BuildingPlan(**{**item, "origin": tuple(item["origin"]), "cells": tuple(tuple(value) for value in item["cells"]), "entrances": tuple(tuple(value) for value in item["entrances"])})


def _restore_society(payload: dict, world, policy=None) -> SocietyLayer:
    society = SocietyLayer(world, seed=int(payload["seed"]), policy=policy)
    society.tick = int(payload["tick"])
    society.rng.bit_generator.state = payload["rng"]
    for item in payload["factions"]:
        faction = FactionState(item["faction_id"], item["name"], int(item["family"]), item["lineage_id"], tuple(item["cultural_traits"]), set(item["technologies"]), set(item["settlement_ids"]), {str(key): float(value) for key, value in item["relations"].items()}, float(item["knowledge"]), float(item["cohesion"]), item["doctrine"])
        society.factions[faction.faction_id] = faction
    for item in payload["settlements"]:
        settlement = SettlementState(item["settlement_id"], item["faction_id"], tuple(item["center"]), int(item["population"]), float(item["wealth"]), float(item["food"]), float(item["power"]), [_building(value) for value in item["buildings"]], {tuple(value) for value in item["roads"]}, int(item["founded_tick"]), {str(key): float(value) for key, value in item.get("stockpiles", {}).items()}, {str(key): float(value) for key, value in item.get("production", {}).items()}, int(item.get("shortages", 0)), int(item.get("projects_completed", 0)))
        society.settlements[settlement.settlement_id] = settlement
    society.history = [HistoryEvent(int(item["tick"]), item["kind"], tuple(item["actors"]), tuple(item["location"]), item["description"], tuple(tuple(value) for value in item["consequences"])) for item in payload["history"]]
    society.activities = {item["activity_id"]: Activity(item["activity_id"], item["kind"], item["issuer"], tuple(item["location"]), float(item["difficulty"]), tuple(tuple(value) for value in item["reward_materials"]), item["description"]) for item in payload["activities"]}
    society.assignments = {int(key): str(value) for key, value in payload["assignments"].items()}
    society.materialized_buildings = set(payload["materialized_buildings"])
    return society


def _journal_payload(journal: QuestJournal) -> dict:
    return {"entries": [_plain(asdict(item)) for item in sorted(journal.entries.values(), key=lambda value: value.quest_id)], "reputation": journal.reputation, "completed": journal.completed}


def _restore_journal(payload: dict) -> QuestJournal:
    journal = QuestJournal()
    journal.entries = {item["quest_id"]: QuestEntry(**{**item, "rewards": tuple(tuple(value) for value in item["rewards"])}) for item in payload["entries"]}
    journal.reputation = {str(key): float(value) for key, value in payload["reputation"].items()}
    journal.completed = int(payload["completed"])
    return journal


def _atlas_payload(atlas: InfiniteNatureAtlas) -> dict:
    return {
        "seed": atlas.seed,
        "visited": [
            {"key": asdict(key), **{name: _plain(value) for name, value in asdict(summary).items() if name != "key"}}
            for key, summary in sorted(atlas.visited.items(), key=lambda pair: (pair[0].depth, pair[0].y, pair[0].x))
        ],
    }


def _restore_atlas(payload: dict) -> InfiniteNatureAtlas:
    atlas = InfiniteNatureAtlas(seed=int(payload["seed"]))
    for item in payload["visited"]:
        key = RegionKey(**{name: int(value) for name, value in item["key"].items()})
        atlas.visited[key] = RegionSummary(key, int(item["seed"]), item["biome"], float(item["fertility"]), float(item["mineral"]), float(item["phase"]), float(item["danger"]), int(item["ruins"]), tuple(item["population"]), int(item["visits"]), item["world_sha256"])
    return atlas


def save_session(*, world, adventure: AdventureState, society: SocietyLayer, quests: QuestJournal, atlas: InfiniteNatureAtlas, region: RegionKey, selected: int, path: Path) -> dict:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nullvector-session-") as directory:
        world_path = Path(directory) / "world.nvz"
        world_report = save_world(world, world_path)
        metadata = {
            "format": SESSION_FORMAT,
            "region": asdict(region),
            "selected": int(selected),
            "world_sha256": world_report["world_sha256"],
            "adventure": _adventure_payload(adventure),
            "society": _society_payload(society),
            "quests": _journal_payload(quests),
            "atlas": _atlas_payload(atlas),
        }
        metadata_bytes = (json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        temporary = path.parent / ("." + path.name + ".tmp-" + uuid.uuid4().hex)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("session.json", metadata_bytes)
                archive.write(world_path, "world.nvz")
            if temporary.stat().st_size > MAX_SESSION_BYTES:
                raise ValueError("nature session exceeds size bound")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "world_sha256": world_report["world_sha256"], "selected": int(selected)}


def load_session(path: Path, *, motion_policy=None, behavior_policy=None, colony_policy=None, society_policy=None) -> dict:
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size > MAX_SESSION_BYTES:
        raise ValueError("nature session missing or oversized")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if [item.filename for item in infos] != ["session.json", "world.nvz"] or any(item.file_size > MAX_SESSION_BYTES for item in infos):
            raise ValueError("nature session archive contract failed")
        metadata = json.loads(archive.read("session.json").decode())
        world_bytes = archive.read("world.nvz")
    if metadata.get("format") != SESSION_FORMAT:
        raise ValueError("nature session format drifted")
    with tempfile.TemporaryDirectory(prefix="nullvector-session-load-") as directory:
        world_path = Path(directory) / "world.nvz"
        world_path.write_bytes(world_bytes)
        world = load_world(world_path, motion_policy=motion_policy, behavior_policy=behavior_policy, colony_policy=colony_policy)
    if world.snapshot().semantic_sha256 != metadata["world_sha256"]:
        raise ValueError("nature session world binding failed")
    selected = int(metadata["selected"])
    if selected not in world.organisms:
        raise ValueError("nature session selected entity is missing")
    region = RegionKey(**{name: int(value) for name, value in metadata["region"].items()})
    return {
        "world": world,
        "adventure": _restore_adventure(metadata["adventure"]),
        "society": _restore_society(metadata["society"], world, society_policy),
        "quests": _restore_journal(metadata["quests"]),
        "atlas": _restore_atlas(metadata["atlas"]),
        "region": region,
        "selected": selected,
    }
