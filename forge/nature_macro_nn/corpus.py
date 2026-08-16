from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from ..nature_sim_v2 import NatureWorld
from ..nature_sim_v2.state import ColonyState
from ..nature_world_scale_v1 import BIOMES, InfiniteNatureAtlas, RegionKey
from ..qud_society_v1 import SocietyLayer
from .contract import CORPUS_FORMAT, GLOBAL_FEATURES, PATCH_SIZE, STATE_CHANNELS, canonical, corpus_source_sha256
from .state import extract_global_state, extract_patch_state

DISK_FLOOR = 100 * 1024**3
ARRAYS = ("previous", "current", "target", "previous_global", "global_state", "target_global")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256((CORPUS_FORMAT + "\0").encode())
    for name in ARRAYS:
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode() + b"\0" + value.dtype.str.encode() + b"\0" + np.asarray(value.shape, dtype="<i8").tobytes() + value.tobytes())
    return digest.hexdigest()


def _bootstrap(seed: int, index: int):
    world = NatureWorld(seed=seed, size=64, max_population=120)
    atlas = InfiniteNatureAtlas(seed=seed ^ 0x504C414E4554)
    region = RegionKey(index % 7 - 3, index // 7 - 2)
    atlas.terraform(world, region)
    world.biome = BIOMES[index % len(BIOMES)]
    world.seed_founders(variants_per_family=3)
    family = index % 5
    members = [item for item in world.organisms.values() if item.family == family][:3]
    center = np.asarray((16.0 + (index * 11) % 30, 16.0 + (index * 17) % 30), np.float64)
    member_ids = set()
    for ordinal, entity in enumerate(members):
        entity.position = (center + np.asarray((ordinal * .72, -ordinal * .36))) % world.size
        entity.colony_id = 1
        entity.energy = .82
        member_ids.add(entity.entity_id)
    world.colonies[1] = ColonyState(1, family, members[0].genome.lineage_id, member_ids, center.copy())
    world.next_colony_id = 2
    society = SocietyLayer(world, seed=seed ^ 0x534F4349455459)
    society.found_from_colony(1)
    settlement = next(iter(society.settlements.values()))
    settlement.stockpiles.update({"water": 10.0, "flora": 9.0, "biomass": 6.0, "mineral": 16.0, "metal": 4.0, "food": 12.0, "medicine": 2.0, "parts": 9.0, "energy": 8.0, "knowledge": 2.0})
    return world, society


def _advance(world, society, *, ticks: int = 4) -> None:
    for _ in range(ticks):
        world.step(.25, publish=False)
    society.step_history(1)


def generate_world_arrays(index: int, *, steps: int, base_seed: int) -> dict[str, np.ndarray]:
    if not 8 <= steps <= 256:
        raise ValueError("macro corpus step count drifted")
    seed = int(base_seed + index * 0x9E3779B1)
    world, society = _bootstrap(seed, index)
    previous = extract_patch_state(world, society); previous_global = extract_global_state(world, society)
    _advance(world, society)
    current = extract_patch_state(world, society); current_global = extract_global_state(world, society)
    rows = {name: [] for name in ARRAYS}
    for _ in range(steps):
        _advance(world, society)
        target = extract_patch_state(world, society); target_global = extract_global_state(world, society)
        for name, value in (("previous", previous), ("current", current), ("target", target), ("previous_global", previous_global), ("global_state", current_global), ("target_global", target_global)):
            rows[name].append(value)
        previous, current = current, target
        previous_global, current_global = current_global, target_global
    arrays = {name: np.asarray(values, dtype=np.float16 if "global" not in name else np.float32) for name, values in rows.items()}
    return arrays


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    stage = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(stage, **arrays)
    os.replace(stage, path)


def build_corpus(destination: Path, *, worlds: int = 32, steps: int = 48, base_seed: int = 0x4D4143524F) -> dict:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if shutil.disk_usage(destination.parent).free < DISK_FLOOR:
        raise OSError("macro corpus disk floor reached")
    stage = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    (stage / "shards").mkdir(parents=True)
    records = []
    try:
        for index in range(worlds):
            arrays = generate_world_arrays(index, steps=steps, base_seed=base_seed)
            path = stage / "shards" / f"world-{index:04d}.npz"
            _write_npz(path, arrays)
            records.append({"index": index, "seed": int(base_seed + index * 0x9E3779B1), "pairs": steps, "artifact": {"path": f"shards/{path.name}", "bytes": path.stat().st_size, "sha256": _sha(path)}, "semantic_sha256": _semantic(arrays)})
        manifest = {"format": CORPUS_FORMAT, "corpus_source_sha256": corpus_source_sha256(), "worlds": worlds, "pairs": worlds * steps, "steps_per_world": steps, "base_seed": base_seed, "channels": list(STATE_CHANNELS), "global_features": GLOBAL_FEATURES, "shards": records}
        manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
        (stage / "manifest.json").write_bytes(canonical(manifest))
        os.replace(stage, destination)
    except BaseException:
        if stage.exists(): shutil.rmtree(stage)
        raise
    return validate_corpus(destination)


def validate_corpus(root: Path, *, load_arrays: bool = True) -> dict:
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text("utf-8")); provided = manifest.pop("manifest_sha256", None)
    if manifest.get("format") != CORPUS_FORMAT or manifest.get("corpus_source_sha256") != corpus_source_sha256() or provided != hashlib.sha256(canonical(manifest)).hexdigest():
        raise ValueError("macro corpus manifest provenance drifted")
    if manifest.get("channels") != list(STATE_CHANNELS) or manifest.get("global_features") != GLOBAL_FEATURES:
        raise ValueError("macro corpus tensor vocabulary drifted")
    pairs = 0
    for record in manifest.get("shards", ()):
        path = root / record["artifact"]["path"]
        if path.stat().st_size != record["artifact"]["bytes"] or _sha(path) != record["artifact"]["sha256"]:
            raise ValueError("macro corpus shard artifact drifted")
        if load_arrays:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != set(ARRAYS): raise ValueError("macro corpus member closure drifted")
                arrays = {name: archive[name] for name in ARRAYS}
            count = record["pairs"]
            for name in ("previous", "current", "target"):
                if arrays[name].shape != (count, len(STATE_CHANNELS), PATCH_SIZE, PATCH_SIZE) or arrays[name].dtype != np.float16: raise ValueError("macro spatial tensor drifted")
            for name in ("previous_global", "global_state", "target_global"):
                if arrays[name].shape != (count, GLOBAL_FEATURES) or arrays[name].dtype != np.float32: raise ValueError("macro global tensor drifted")
            if _semantic(arrays) != record["semantic_sha256"] or any(not np.isfinite(value).all() for value in arrays.values()): raise ValueError("macro corpus semantics drifted")
        pairs += int(record["pairs"])
    if pairs != manifest["pairs"] or len(manifest["shards"]) != manifest["worlds"]: raise ValueError("macro corpus census drifted")
    return {"passed": True, "worlds": manifest["worlds"], "pairs": pairs, "manifest_sha256": provided}
