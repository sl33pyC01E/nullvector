from __future__ import annotations

from collections import deque
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image, ImageDraw

from ..cellular_symmetry import validate_bank as validate_symmetry_bank
from ..config import PROJECT_ROOT
from ..maps.io import array_digest, file_sha256, load_map_pack
from ..maps.model import Hazard, THEMES, Terrain
from ..maps.validate import validate_pack
from ..multifield_style_motion.hashing import canonical_json_bytes, sha256_bytes
from ..safety import require_disk_floor
from .contract import (
    DEFAULT_MAP_ROOT,
    DEFAULT_ORGANISM_SOURCE,
    DEFAULT_OUTPUT,
    FAMILIES,
    FIELD_FORMAT,
    FIELD_NAMES,
    FORMAT,
    MAP_FORMAT,
    RESOURCE_NAMES,
    SCHEMA_PATH,
    source_sha256,
)


ARRAY_NAMES = (*FIELD_NAMES, "family_suitability", "resource_type")
THEME_BASE = {
    "arena": (0.38, 0.28, 0.72, 0.58, 0.08, 0.30),
    "rooms": (0.44, 0.31, 0.48, 0.51, 0.10, 0.37),
    "caves": (0.48, 0.57, 0.18, 0.42, 0.13, 0.58),
    "archipelago": (0.52, 0.88, 0.80, 0.64, 0.06, 0.28),
    "garden": (0.86, 0.82, 0.76, 0.57, 0.09, 0.34),
    "anomaly": (0.32, 0.38, 0.45, 0.66, 0.62, 0.91),
}
FIELD_COLORS = {
    "nutrient": (245, 169, 63), "moisture": (42, 177, 255), "light": (255, 244, 144),
    "toxicity": (198, 74, 241), "energy": (60, 255, 211), "biomass": (74, 236, 111),
}
FAMILY_COLORS = ((63, 224, 255), (255, 135, 70), (110, 255, 86), (219, 83, 255), (170, 190, 225))


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Ecology source must remain inside project root: {resolved}")
    return resolved.relative_to(PROJECT_ROOT).as_posix()


def _splitmix(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def _noise(shape: tuple[int, int], seed: int, channel: int) -> np.ndarray:
    height, width = shape
    values = np.empty(shape, dtype=np.float32)
    salt = _splitmix(seed ^ (channel * 0xD1B54A32D192ED03))
    for y in range(height):
        for x in range(width):
            mixed = _splitmix(salt ^ (x * 0x9E3779B1) ^ (y * 0x85EBCA77))
            values[y, x] = np.float32((mixed & 0xFFFFFF) / 16777215.0)
    # Three integer-neighborhood diffusion passes create coherent ecological patches.
    for _ in range(3):
        values = np.asarray(
            (values * 4.0 + np.roll(values, 1, 0) + np.roll(values, -1, 0)
             + np.roll(values, 1, 1) + np.roll(values, -1, 1)) / 8.0,
            dtype=np.float32,
        )
    return values


def _distance(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    limit = height + width
    distances = np.full((height, width), limit, dtype=np.int16)
    queue: deque[tuple[int, int]] = deque()
    for y, x in np.argwhere(mask):
        distances[y, x] = 0
        queue.append((int(x), int(y)))
    while queue:
        x, y = queue.popleft(); next_distance = int(distances[y, x]) + 1
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and next_distance < int(distances[ny, nx]):
                distances[ny, nx] = next_distance; queue.append((nx, ny))
    return distances.astype(np.float32)


def _proximity(mask: np.ndarray, radius: float) -> np.ndarray:
    return np.asarray(np.clip(1.0 - _distance(mask) / radius, 0.0, 1.0), dtype=np.float32)


def _clip(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.clip(values, 0.0, 1.0), dtype=np.float32)


def _derive_fields(data) -> dict[str, np.ndarray]:
    terrain = data.terrain; hazard = data.hazard; walk = data.walkability != 0
    safe = walk & (hazard == int(Hazard.NONE))
    base = THEME_BASE[data.theme]
    noise_a = _noise(data.shape, int(data.seed), 1) - 0.5
    noise_b = _noise(data.shape, int(data.seed), 2) - 0.5
    water = terrain == int(Terrain.WATER); growth = terrain == int(Terrain.GROWTH)
    crystal = terrain == int(Terrain.CRYSTAL); sand = terrain == int(Terrain.SAND)
    walls = ~walk; spores = hazard == int(Hazard.SPORES); lava = hazard == int(Hazard.LAVA)
    arc = hazard == int(Hazard.ARC); lasers = hazard == int(Hazard.LASER)
    water_near = _proximity(water, 9.0); growth_near = _proximity(growth, 7.0)
    crystal_near = _proximity(crystal, 8.0); wall_near = _proximity(walls, 5.0)
    hazard_near = _proximity(hazard != 0, 6.0)
    elev = data.elevation.astype(np.float32)
    elev = (elev - float(elev.min())) / max(float(elev.max() - elev.min()), 1.0)
    nutrient = _clip(base[0] + 0.32 * growth_near + 0.13 * water_near + 0.14 * noise_a - 0.22 * hazard_near)
    moisture = _clip(base[1] + 0.45 * water_near + 0.15 * growth_near + 0.10 * noise_b - 0.12 * lava)
    light = _clip(base[2] + 0.18 * elev - 0.27 * wall_near + 0.12 * noise_a + 0.16 * crystal_near)
    temperature = _clip(base[3] + 0.36 * lava + 0.08 * lasers + 0.08 * noise_b - 0.12 * water_near)
    toxicity = _clip(base[4] + 0.55 * spores + 0.33 * lava + 0.25 * arc + 0.14 * hazard_near + (0.16 * noise_a if data.theme == "anomaly" else 0.0))
    energy = _clip(base[5] + 0.42 * crystal_near + 0.48 * arc + 0.17 * lasers + 0.10 * noise_b)
    biomass = _clip((nutrient * 0.42 + moisture * 0.30 + light * 0.20 + growth_near * 0.20) * (1.0 - toxicity * 0.55))
    # Blocked cells retain environmental fields but cannot be occupied or harvested.
    occupiable = safe.astype(np.float32)
    balanced_temp = 1.0 - np.abs(temperature - 0.52) * 1.7
    suitability = np.stack((
        _clip((0.34 * nutrient + 0.22 * moisture + 0.18 * energy + 0.18 * balanced_temp + 0.08 * light) * (1.0 - 0.78 * toxicity)),
        _clip((0.46 * nutrient + 0.30 * moisture + 0.15 * balanced_temp + 0.09 * light) * (1.0 - 0.92 * toxicity)),
        _clip((0.34 * light + 0.31 * moisture + 0.28 * nutrient + 0.18 * growth_near) * (1.0 - 0.65 * toxicity)),
        _clip(0.40 * energy + 0.32 * toxicity + 0.15 * temperature + 0.13 * noise_a + 0.10),
        _clip((0.50 * energy + 0.24 * crystal_near + 0.16 * (1.0 - moisture) + 0.10 * balanced_temp) * (1.0 - 0.24 * toxicity)),
    ), axis=0).astype(np.float32)
    suitability *= occupiable[None, :, :]
    return {
        "nutrient": nutrient, "moisture": moisture, "light": light, "temperature": temperature,
        "toxicity": toxicity, "energy": energy, "biomass": biomass,
        "family_suitability": np.ascontiguousarray(suitability, dtype=np.float32),
    }


def _resource_nodes(data, fields: dict[str, np.ndarray]) -> tuple[np.ndarray, list[dict[str, object]]]:
    safe = (data.walkability != 0) & (data.hazard == 0) & (data.decoration_forbidden == 0)
    resource_type = np.zeros(data.shape, dtype=np.uint8)
    nodes: list[dict[str, object]] = []
    chosen: list[tuple[int, int]] = []
    signals = (
        fields["nutrient"] * 0.62 + fields["biomass"] * 0.38,
        fields["moisture"] * 0.52 + fields["nutrient"] * 0.48,
        fields["light"] * 0.48 + fields["biomass"] * 0.52,
        fields["energy"] * 0.55 + fields["toxicity"] * 0.45,
        fields["energy"] * 0.67 + (1.0 - fields["moisture"]) * 0.33,
    )
    for family_id, signal in enumerate(signals):
        candidates = []
        for y, x in np.argwhere(safe):
            tie = _splitmix(int(data.seed) ^ family_id * 0x9E37 ^ int(x) * 0x85EB ^ int(y) * 0xC2B2) & 0xFFFF
            candidates.append((float(signal[y, x]), int(tie), int(x), int(y)))
        candidates.sort(reverse=True)
        accepted = 0
        for value, _, x, y in candidates:
            if any(abs(x - ox) + abs(y - oy) < 3 for ox, oy in chosen):
                continue
            chosen.append((x, y)); accepted += 1; resource_type[y, x] = family_id + 1
            nodes.append({
                "id": len(nodes), "family": FAMILIES[family_id], "family_id": family_id,
                "kind": RESOURCE_NAMES[family_id + 1], "resource_type": family_id + 1,
                "position": [x, y], "capacity": round(20.0 + value * 80.0, 6),
                "regrowth_per_second": round(0.04 + float(fields["biomass"][y, x]) * 0.34, 6),
                "habitat_score": round(float(fields["family_suitability"][family_id, y, x]), 7),
            })
            if accepted == 4:
                break
        if accepted < 3:
            raise ValueError(f"Ecology could not place three distinct nodes for {FAMILIES[family_id]} on {data.map_id}")
    return resource_type, nodes


def _stats(array: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    values = array[mask] if mask is not None else array.reshape(-1)
    return {"minimum": round(float(values.min()), 7), "mean": round(float(values.mean()), 7), "maximum": round(float(values.max()), 7)}


def _compile_map(pack: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    validation = validate_pack(pack)
    if not validation.get("passed"):
        raise ValueError(f"Map pack is not valid: {pack}")
    data = load_map_pack(pack)
    manifest_path = pack / "manifest.json"; source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fields = _derive_fields(data); resource_type, nodes = _resource_nodes(data, fields); fields["resource_type"] = resource_type
    safe = (data.walkability != 0) & (data.hazard == 0)
    arrays_sha = array_digest(fields)
    record: dict[str, object] = {
        "format": MAP_FORMAT, "map_id": data.map_id, "theme": data.theme,
        "dimensions": {"width": data.config.width, "height": data.config.height},
        "source": {
            "manifest": _relative(manifest_path), "manifest_sha256": file_sha256(manifest_path),
            "semantic_array_sha256": source_manifest["semantic_array_sha256"],
            "topology_mask_sha256": source_manifest["semantics"]["topology_masks"]["combined_sha256"],
        },
        "ecology_array_sha256": arrays_sha,
        "statistics": {name: _stats(fields[name], safe) for name in FIELD_NAMES},
        "family_habitats": [
            {"family": name, "family_id": index, **_stats(fields["family_suitability"][index], safe)}
            for index, name in enumerate(FAMILIES)
        ],
        "resource_nodes": nodes,
        "gates": {
            "source_map_v2_valid": True, "mission_topology_untouched": True,
            "resources_outside_forbidden_mask": bool(all(data.decoration_forbidden[y, x] == 0 for x, y in (node["position"] for node in nodes))),
            "resources_safe_and_walkable": bool(all(data.walkability[y, x] != 0 and data.hazard[y, x] == 0 for x, y in (node["position"] for node in nodes))),
            "every_family_has_three_resources": all(sum(node["family_id"] == family for node in nodes) >= 3 for family in range(len(FAMILIES))),
            "fields_finite_and_bounded": all(np.isfinite(fields[name]).all() and float(fields[name].min()) >= 0.0 and float(fields[name].max()) <= 1.0 for name in (*FIELD_NAMES, "family_suitability")),
        },
    }
    return fields, record


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = BytesIO(); np.savez_compressed(buffer, format=np.asarray([FIELD_FORMAT]), **{name: arrays[name] for name in ARRAY_NAMES})
    return buffer.getvalue()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        expected = {"format", *ARRAY_NAMES}
        if set(archive.files) != expected or archive["format"].shape != (1,) or str(archive["format"][0]) != FIELD_FORMAT:
            raise ValueError("Ecology archive member contract differs")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in ARRAY_NAMES}
    shape = arrays["nutrient"].shape
    if any(arrays[name].shape != shape or arrays[name].dtype != np.float32 for name in FIELD_NAMES):
        raise ValueError("Ecology scalar field shape/dtype differs")
    if arrays["family_suitability"].shape != (5, *shape) or arrays["family_suitability"].dtype != np.float32:
        raise ValueError("Ecology family field shape/dtype differs")
    if arrays["resource_type"].shape != shape or arrays["resource_type"].dtype != np.uint8 or int(arrays["resource_type"].max()) > 5:
        raise ValueError("Ecology resource field contract differs")
    return arrays


def _heat(values: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    strength = np.clip(values, 0.0, 1.0)[..., None]
    base = np.asarray((4, 10, 18), dtype=np.float32)
    rgb = base + strength * (np.asarray(color, dtype=np.float32) - base)
    return Image.fromarray(np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8))


def _overview(data, fields: dict[str, np.ndarray], nodes: list[dict[str, object]], scale: int = 3) -> Image.Image:
    terrain_palette = np.asarray(((2, 7, 14), (22, 36, 50), (8, 14, 23), (14, 61, 89), (93, 72, 46), (28, 84, 45), (32, 91, 116), (5, 4, 12), (100, 82, 44)), dtype=np.uint8)
    image = terrain_palette[np.clip(data.terrain, 0, len(terrain_palette) - 1)]
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for node in nodes:
        x, y = node["position"]; color = FAMILY_COLORS[int(node["family_id"])]
        draw.point((x, y), fill=color)
    return image.resize((data.config.width * scale, data.config.height * scale), Image.Resampling.NEAREST)


def _contact_sheet(compiled: list[tuple[object, dict[str, np.ndarray], dict[str, object]]]) -> bytes:
    scale = 3; cell = 48 * scale; header = 18; columns = 8
    sheet = Image.new("RGB", (columns * cell, len(compiled) * (cell + header)), (3, 8, 15)); draw = ImageDraw.Draw(sheet)
    labels = ("HABITAT", "NUTRIENT", "MOISTURE", "LIGHT", "TOXICITY", "ENERGY", "BIOMASS", "FAMILY MAX")
    for row, (data, fields, record) in enumerate(compiled):
        y0 = row * (cell + header); draw.text((4, y0 + 3), f"{record['theme'].upper()} // {record['map_id']}", fill=(130, 224, 240))
        panels = [_overview(data, fields, record["resource_nodes"], scale)]
        for name in ("nutrient", "moisture", "light", "toxicity", "energy", "biomass"):
            panels.append(_heat(fields[name], FIELD_COLORS[name]).resize((cell, cell), Image.Resampling.NEAREST))
        family = np.argmax(fields["family_suitability"], axis=0); intensity = np.max(fields["family_suitability"], axis=0)[..., None]
        family_rgb = np.asarray(FAMILY_COLORS, dtype=np.float32)[family] * intensity + np.asarray((3, 8, 15)) * (1.0 - intensity)
        panels.append(Image.fromarray(np.asarray(family_rgb, dtype=np.uint8)).resize((cell, cell), Image.Resampling.NEAREST))
        for column, panel in enumerate(panels):
            x0 = column * cell; sheet.paste(panel, (x0, y0 + header)); draw.text((x0 + 3, y0 + header + 3), labels[column], fill=(225, 246, 250))
    buffer = BytesIO(); sheet.save(buffer, format="PNG", optimize=False, compress_level=9); return buffer.getvalue()


def _discover_maps(root: Path) -> list[Path]:
    root = Path(root).resolve(); found: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "manifest.json").is_file():
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8")); theme = manifest.get("theme")
            if theme in THEMES:
                if theme in found: raise ValueError(f"Multiple ecology map sources for theme {theme}")
                found[theme] = path
    if set(found) != set(THEMES): raise ValueError(f"Ecology requires exactly six themes, found {sorted(found)}")
    return [found[theme] for theme in THEMES]


def _build_files(map_root: Path, organism_manifest: Path = DEFAULT_ORGANISM_SOURCE) -> tuple[dict[str, bytes], dict[str, object]]:
    organism_manifest = Path(organism_manifest).resolve(); organism_validation = validate_symmetry_bank(organism_manifest)
    organism = json.loads(organism_manifest.read_text(encoding="utf-8"))
    files: dict[str, bytes] = {}; records = []; compiled = []
    for pack in _discover_maps(map_root):
        data = load_map_pack(pack); fields, record = _compile_map(pack)
        relative = f"maps/{data.theme}/ecology_fields.npz"; payload = _npz_bytes(fields); files[relative] = payload
        record["artifact"] = {"path": relative, "bytes": len(payload), "sha256": sha256_bytes(payload)}
        records.append(record); compiled.append((data, fields, record))
    contact = _contact_sheet(compiled); files["cellular_ecology_contact_sheet.png"] = contact
    total_nodes = sum(len(record["resource_nodes"]) for record in records)
    manifest: dict[str, object] = {
        "format": FORMAT, "status": "ready", "quality_tier": "deterministic-topology-safe-living-habitat-v1",
        "compiler": {"source_sha256": source_sha256(), "rng": "coordinate-splitmix64-plus-fixed-neighbor-diffusion-v1"},
        "organism_source": {
            "manifest": _relative(organism_manifest), "manifest_sha256": file_sha256(organism_manifest),
            "semantic_sha256": organism["semantic_sha256"], "sample_count": organism_validation["sample_count"],
            "symmetry_policy": "family-aware-soft-bilateral-additive-v1",
        },
        "family_vocab": [{"id": index, "name": name} for index, name in enumerate(FAMILIES)],
        "resource_vocab": [{"id": index, "name": name} for index, name in enumerate(RESOURCE_NAMES)],
        "field_vocab": list(FIELD_NAMES), "map_count": len(records), "resource_node_count": total_nodes,
        "maps": records,
        "contact_sheet": {"path": "cellular_ecology_contact_sheet.png", "bytes": len(contact), "sha256": sha256_bytes(contact)},
        "runtime_contract": {
            "pixels_are_resource_cells": True, "resources_regrow": True, "organisms_consume_local_resources": True,
            "family_specific_niches": True, "mission_topology_is_immutable": True,
            "environmental_damage_sources": ["toxicity", "temperature", "hazard"],
            "reproduction_requires_energy_and_local_carrying_capacity": True,
        },
        "gates": {
            "six_topology_v2_maps": len(records) == 6, "all_source_maps_valid": True,
            "all_ecology_fields_exact": True, "all_resource_nodes_legal": all(all(record["gates"].values()) for record in records),
            "all_families_have_niches": all(any(node["family_id"] == family for node in record["resource_nodes"]) for record in records for family in range(5)),
            "soft_symmetry_source_preserved": True,
        },
    }
    semantic = dict(manifest); manifest["semantic_sha256"] = sha256_bytes(canonical_json_bytes(semantic))
    files["cellular_ecology_manifest.json"] = canonical_json_bytes(manifest)
    return files, manifest


def _publish(destination: Path, files: Mapping[str, bytes]) -> None:
    destination = Path(destination).resolve()
    if destination.exists(): raise FileExistsError(destination)
    require_disk_floor(destination.parent, planned_bytes=sum(map(len, files.values())) + 256 * 1024**2)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"; staging.mkdir(parents=True)
    try:
        for relative, payload in sorted(files.items()):
            target = staging.joinpath(*PurePosixPath(relative).parts); target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
            with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, target)
        os.replace(staging, destination)
    except BaseException:
        # Preserve staging evidence; never publish a partial bank.
        raise


def build_bank(map_root: Path = DEFAULT_MAP_ROOT, destination: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    first, manifest = _build_files(map_root); second, _ = _build_files(map_root)
    if first != second: raise ValueError("Ecology build is not byte deterministic")
    if not all(manifest["gates"].values()): raise ValueError("Ecology bank gates failed")
    _publish(destination, first); validation = validate_bank(Path(destination) / "cellular_ecology_manifest.json")
    return {"passed": True, "destination": str(Path(destination).resolve()), "map_count": 6, "resource_node_count": manifest["resource_node_count"], "semantic_sha256": manifest["semantic_sha256"], "manifest_sha256": file_sha256(Path(destination) / "cellular_ecology_manifest.json"), "validation": validation}


def _artifact(root: Path, record: Mapping[str, object]) -> Path:
    relative = str(record["path"]); pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative: raise ValueError("Unsafe ecology artifact path")
    path = root.joinpath(*pure.parts).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError("Ecology artifact missing/outside bank")
    if path.stat().st_size != int(record["bytes"]) or file_sha256(path) != record["sha256"]: raise ValueError("Ecology artifact hash/size differs")
    return path


def validate_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve(); raw = manifest_path.read_bytes(); manifest = json.loads(raw)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8")); errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda error: list(error.absolute_path))
    if errors: raise ValueError(f"Ecology schema validation failed: {errors[0].message}")
    if raw != canonical_json_bytes(manifest): raise ValueError("Ecology manifest is not canonical JSON")
    semantic = {key: value for key, value in manifest.items() if key != "semantic_sha256"}
    if manifest["semantic_sha256"] != sha256_bytes(canonical_json_bytes(semantic)): raise ValueError("Ecology semantic hash differs")
    if manifest["compiler"]["source_sha256"] != source_sha256(): raise ValueError("Ecology source provenance is stale")
    organism_path = PROJECT_ROOT.joinpath(*PurePosixPath(manifest["organism_source"]["manifest"]).parts)
    if file_sha256(organism_path) != manifest["organism_source"]["manifest_sha256"]: raise ValueError("Ecology organism source differs")
    validate_symmetry_bank(organism_path)
    root = manifest_path.parent; actual_themes = []
    for record in manifest["maps"]:
        source_path = PROJECT_ROOT.joinpath(*PurePosixPath(record["source"]["manifest"]).parts); pack = source_path.parent
        if file_sha256(source_path) != record["source"]["manifest_sha256"]: raise ValueError("Ecology map source manifest differs")
        expected_fields, expected_record = _compile_map(pack); fields = _load_npz(_artifact(root, record["artifact"]))
        for name in ARRAY_NAMES:
            if not np.array_equal(fields[name], expected_fields[name]): raise ValueError(f"Ecology field replay differs: {record['theme']} {name}")
        expected_record["artifact"] = record["artifact"]
        if expected_record != record: raise ValueError(f"Ecology map record replay differs: {record['theme']}")
        actual_themes.append(record["theme"])
    if actual_themes != list(THEMES) or not all(manifest["gates"].values()): raise ValueError("Ecology census/gates differ")
    _artifact(root, manifest["contact_sheet"])
    return {"passed": True, "map_count": 6, "resource_node_count": manifest["resource_node_count"], "semantic_sha256": manifest["semantic_sha256"], "manifest_sha256": file_sha256(manifest_path), "contact_sheet_sha256": manifest["contact_sheet"]["sha256"]}


def replay_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve(); validation = validate_bank(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")); map_root = PROJECT_ROOT / PurePosixPath(manifest["maps"][0]["source"]["manifest"]).parent.parent
    expected, expected_manifest = _build_files(map_root)
    if expected_manifest["semantic_sha256"] != manifest["semantic_sha256"]: raise ValueError("Ecology semantic replay differs")
    root = manifest_path.parent
    for relative, payload in expected.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.read_bytes() != payload: raise ValueError(f"Ecology byte replay differs: {relative}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(expected): raise ValueError("Ecology output closure differs")
    return {**validation, "exact_replay": True, "artifact_count": len(expected), "artifact_bytes": sum(map(len, expected.values()))}
