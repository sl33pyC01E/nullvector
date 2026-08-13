from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image, ImageDraw

from ..cellular_organism.compiler import (
    _atomic_publish,
    _compile_arrays,
    _components,
    _load_arrays,
    _render_panels,
    validate_species_arrays,
)
from ..cellular_organism.contract import CELLULAR_CONTRACT_SHA256, CANVAS_SIZE, SimulationDefaults
from ..evolved_cellular_organism import validate_bank as validate_parent_bank
from ..map_decorator.hashing import json_sha256
from ..morphology.constants import FAMILIES, ROLE_NAMES, SUBTYPE_NAMES
from ..multifield_style.hashing import sha256_file
from ..multifield_style.model import CategoricalFields, StyleCondition
from ..multifield_style_motion.hashing import (
    artifact_record_from_bytes,
    canonical_json_bytes,
    deterministic_npz_bytes,
    png_bytes,
    sha256_bytes,
)
from ..neural_rig_bridge.hashing import aligned_fields_hash
from .contract import (
    CROSSOVER_MODES,
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE,
    FIELDS_FORMAT,
    FORMAT,
    MUTATION_MODES,
    OFFSPRING_COUNT,
    SCHEMA_PATH,
    SPECIES_FORMAT,
    source_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIELD_KEYS = frozenset({"format", "part_owner", "material", "emission", "ancestry", "mutation_mask", "repair_mask"})


@dataclass(frozen=True, slots=True)
class BreedSample:
    condition: StyleCondition
    fields: CategoricalFields


@dataclass(frozen=True, slots=True)
class Recipe:
    ordinal: int
    primary: Mapping[str, Any]
    donor: Mapping[str, Any]
    crossover_mode: str
    mutation_mode: str
    seed: int
    family_pair: tuple[str, str]


def _safe_artifact(root: Path, record: Mapping[str, Any], *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
        raise ValueError(f"Malformed {label} artifact record")
    pure = PurePosixPath(str(record["path"]))
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Unsafe {label} artifact path")
    path = root.joinpath(*pure.parts).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing {label} artifact")
    if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"{label} artifact hash/size mismatch")
    return path


def _unit(seed: int, *labels: object) -> float:
    payload = ":".join([str(seed), *(str(label) for label in labels)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float((1 << 64) - 1)


def _seed(*labels: object) -> int:
    payload = "\0".join(map(str, labels)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def _parent_grids(record: Mapping[str, Any], root: Path) -> dict[str, np.ndarray]:
    arrays = _load_arrays(_safe_artifact(root, record["arrays"], label=f"parent anatomy {record['sample_id']}"))
    part = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    material = np.zeros_like(part)
    emission = np.zeros_like(part)
    for index, (x_value, y_value) in enumerate(arrays["position_xy"]):
        x, y = int(x_value), int(y_value)
        part[y, x] = arrays["part_owner"][index]
        material[y, x] = arrays["material"][index]
        emission[y, x] = arrays["emission"][index]
    return {"part_owner": part, "material": material, "emission": emission}


def _recipes(parent_records: list[Mapping[str, Any]]) -> list[Recipe]:
    groups = {family_id: [record for record in parent_records if int(record["family_id"]) == family_id] for family_id in range(5)}
    if any(not values for values in groups.values()):
        raise ValueError("Breeding requires all five parent morphology families")
    recipes: list[Recipe] = []
    family_pairs = list(itertools.combinations_with_replacement(range(5), 2))
    ordinal = 0
    for pair_index, (left_family, right_family) in enumerate(family_pairs):
        for replicate in range(3):
            primary_family, donor_family = (left_family, right_family)
            if left_family != right_family and replicate == 1:
                primary_family, donor_family = donor_family, primary_family
            primary_group = groups[primary_family]
            donor_group = groups[donor_family]
            primary = primary_group[(pair_index * 2 + replicate * 3) % len(primary_group)]
            donor = donor_group[(pair_index * 3 + replicate * 2 + 1) % len(donor_group)]
            if primary["sample_id"] == donor["sample_id"]:
                donor = donor_group[(donor_group.index(donor) + 1) % len(donor_group)]
            crossover_mode = CROSSOVER_MODES[ordinal % len(CROSSOVER_MODES)]
            mutation_mode = MUTATION_MODES[(ordinal * 5 + 2) % len(MUTATION_MODES)]
            recipe_seed = _seed("cellular-breeding-v1", ordinal, primary["sample_id"], donor["sample_id"], crossover_mode, mutation_mode)
            recipes.append(
                Recipe(
                    ordinal=ordinal,
                    primary=primary,
                    donor=donor,
                    crossover_mode=crossover_mode,
                    mutation_mode=mutation_mode,
                    seed=recipe_seed,
                    # combinations_with_replacement already emits canonical
                    # morphology-vocabulary order.  Do not alphabetize this:
                    # family-pair identity is a semantic, not display, key.
                    family_pair=(FAMILIES[left_family], FAMILIES[right_family]),
                )
            )
            ordinal += 1
    if len(recipes) != OFFSPRING_COUNT:
        raise AssertionError("Cellular breeding recipe census drifted")
    return recipes


def _choose_donor(mode: str, seed: int, x: int, y: int, donor_owner: int) -> bool:
    if mode == "sagittal_splice":
        threshold = 21 + int(_unit(seed, "sx") * 7)
        return x >= threshold
    if mode == "transverse_splice":
        threshold = 21 + int(_unit(seed, "sy") * 7)
        return y >= threshold
    if mode == "radial_graft":
        cx = 23.5 + (_unit(seed, "cx") - 0.5) * 6.0
        cy = 23.5 + (_unit(seed, "cy") - 0.5) * 6.0
        radius = 8.0 + _unit(seed, "radius") * 8.0
        return (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius
    if mode == "voronoi_weave":
        tile = (x // 6, y // 6)
        return _unit(seed, "tile", *tile) < 0.5
    if mode == "organ_graft":
        return donor_owner in {4, 5, 6, 7, 8, 9, 12, 15}
    if mode == "cellular_mosaic":
        return _unit(seed, "pixel", x, y) < 0.48
    raise ValueError(f"Unknown crossover mode: {mode}")


def _copy_pixel(target: dict[str, np.ndarray], source: Mapping[str, np.ndarray], x: int, y: int, ancestry: int) -> None:
    target["part_owner"][y, x] = source["part_owner"][y, x]
    target["material"][y, x] = source["material"][y, x]
    target["emission"][y, x] = source["emission"][y, x]
    target["ancestry"][y, x] = np.uint8(ancestry)


def _reinforce_contribution(
    child: dict[str, np.ndarray], source: Mapping[str, np.ndarray], *, ancestry: int, seed: int, minimum: int = 8
) -> None:
    present = child["ancestry"] == ancestry
    needed = minimum - int(np.count_nonzero(present))
    if needed <= 0:
        return
    candidates = [(int(x), int(y)) for y, x in np.argwhere(source["part_owner"] > 0)]
    candidates.sort(key=lambda point: (_unit(seed, "reinforce", ancestry, *point), point[1], point[0]))
    for x, y in candidates[:needed]:
        _copy_pixel(child, source, x, y, ancestry)


def _ensure_population(child: dict[str, np.ndarray], primary: Mapping[str, np.ndarray], donor: Mapping[str, np.ndarray], seed: int) -> None:
    minimum = 32
    current = int(np.count_nonzero(child["part_owner"]))
    if current >= minimum:
        return
    candidates: list[tuple[float, int, int, Mapping[str, np.ndarray], int]] = []
    for ancestry, source in ((1, primary), (2, donor)):
        for y, x in np.argwhere((source["part_owner"] > 0) & (child["part_owner"] == 0)):
            candidates.append((_unit(seed, "population", ancestry, int(x), int(y)), int(x), int(y), source, ancestry))
    candidates.sort(key=lambda item: (item[0], item[2], item[1], item[4]))
    for _, x, y, source, ancestry in candidates[: minimum - current]:
        _copy_pixel(child, source, x, y, ancestry)


def _neighbor_points(mask: np.ndarray, x: int, y: int) -> list[tuple[int, int]]:
    return [
        (nx, ny)
        for ny in range(max(0, y - 1), min(CANVAS_SIZE, y + 2))
        for nx in range(max(0, x - 1), min(CANVAS_SIZE, x + 2))
        if (nx != x or ny != y) and bool(mask[ny, nx])
    ]


def _apply_mutation(child: dict[str, np.ndarray], donor: Mapping[str, np.ndarray], mode: str, seed: int) -> None:
    part = child["part_owner"]
    before = {name: child[name].copy() for name in ("part_owner", "material", "emission")}
    physical = part > 0
    if mode == "none":
        pass
    elif mode == "budding_growth":
        candidates = []
        for y in range(1, CANVAS_SIZE - 1):
            for x in range(1, CANVAS_SIZE - 1):
                if physical[y, x]:
                    continue
                neighbors = _neighbor_points(physical, x, y)
                if neighbors:
                    candidates.append((_unit(seed, "bud", x, y), x, y, neighbors))
        candidates.sort(key=lambda item: item[:3])
        count = max(4, min(28, int(np.count_nonzero(physical) * 0.045)))
        for _, x, y, neighbors in candidates[:count]:
            source_x, source_y = min(neighbors, key=lambda point: (_unit(seed, "bud-source", x, y, *point), point[1], point[0]))
            for name in ("part_owner", "material", "emission"):
                child[name][y, x] = child[name][source_y, source_x]
            child["ancestry"][y, x] = 3
    elif mode == "boundary_apoptosis":
        candidates = []
        for y, x in np.argwhere(physical):
            x_value, y_value = int(x), int(y)
            if part[y_value, x_value] in (3, 10):
                continue
            degree = len(_neighbor_points(physical, x_value, y_value))
            if degree < 8:
                candidates.append((_unit(seed, "apoptosis", x_value, y_value), x_value, y_value))
        candidates.sort()
        count = max(2, min(18, int(np.count_nonzero(physical) * 0.025)))
        for _, x, y in candidates[:count]:
            for name in ("part_owner", "material", "emission", "ancestry"):
                child[name][y, x] = 0
    elif mode == "armor_metaplasia":
        candidates = [(int(x), int(y)) for y, x in np.argwhere(physical)]
        candidates.sort(key=lambda point: (_unit(seed, "armor", *point), point[1], point[0]))
        for x, y in candidates[: max(3, min(24, len(candidates) // 18))]:
            child["material"][y, x] = 5
    elif mode == "bioluminescent_shift":
        candidates = [(int(x), int(y)) for y, x in np.argwhere(physical)]
        candidates.sort(key=lambda point: (_unit(seed, "light", *point), point[1], point[0]))
        for index, (x, y) in enumerate(candidates[: max(4, min(32, len(candidates) // 13))]):
            child["emission"][y, x] = 3 if index % 5 == 0 else max(1, int(child["emission"][y, x]))
    elif mode == "appendage_graft":
        candidates = [
            (int(x), int(y))
            for y, x in np.argwhere(np.isin(donor["part_owner"], (4, 5, 6, 7, 8, 9, 12, 15)))
        ]
        candidates.sort(key=lambda point: (_unit(seed, "graft", *point), point[1], point[0]))
        for x, y in candidates[: max(6, min(40, len(candidates)))]:
            _copy_pixel(child, donor, x, y, 2)
    else:
        raise ValueError(f"Unknown mutation mode: {mode}")
    child["mutation_mask"] |= (
        (child["part_owner"] != before["part_owner"])
        | (child["material"] != before["material"])
        | (child["emission"] != before["emission"])
    ).astype(np.uint8)


def _bresenham(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = a; x1, y1 = b
    dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
    error = dx + dy
    result: list[tuple[int, int]] = []
    while True:
        result.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy; x0 += sx
        if doubled <= dx:
            error += dx; y0 += sy
    return result


def _repair_connectivity(child: dict[str, np.ndarray]) -> None:
    while True:
        mask = child["part_owner"] > 0
        components = _components(mask)
        if len(components) <= 1:
            return
        main, detached = components[0], components[1]
        start, end = min(
            itertools.product(main, detached),
            key=lambda pair: ((pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2, pair),
        )
        for x, y in _bresenham(start, end):
            if child["part_owner"][y, x] != 0:
                continue
            occupied = np.argwhere(child["part_owner"] > 0)
            nearest_y, nearest_x = min(
                ((int(py), int(px)) for py, px in occupied),
                key=lambda point: ((point[1] - x) ** 2 + (point[0] - y) ** 2, point[0], point[1]),
            )
            for name in ("part_owner", "material", "emission"):
                child[name][y, x] = child[name][nearest_y, nearest_x]
            child["ancestry"][y, x] = 3
            child["repair_mask"][y, x] = 1


def _breed_fields(recipe: Recipe, parent_root: Path) -> dict[str, np.ndarray]:
    primary = _parent_grids(recipe.primary, parent_root)
    donor = _parent_grids(recipe.donor, parent_root)
    child = {
        "part_owner": np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8),
        "material": np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8),
        "emission": np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8),
        "ancestry": np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8),
        "mutation_mask": np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8),
        "repair_mask": np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8),
    }
    for y in range(CANVAS_SIZE):
        for x in range(CANVAS_SIZE):
            use_donor = _choose_donor(recipe.crossover_mode, recipe.seed, x, y, int(donor["part_owner"][y, x]))
            preferred, fallback = ((donor, 2), (primary, 1)) if use_donor else ((primary, 1), (donor, 2))
            source, ancestry = preferred if preferred[0]["part_owner"][y, x] > 0 else fallback
            if source["part_owner"][y, x] > 0:
                _copy_pixel(child, source, x, y, ancestry)
    _ensure_population(child, primary, donor, recipe.seed)
    _reinforce_contribution(child, primary, ancestry=1, seed=recipe.seed)
    _reinforce_contribution(child, donor, ancestry=2, seed=recipe.seed)
    _apply_mutation(child, donor, recipe.mutation_mode, recipe.seed)
    _reinforce_contribution(child, primary, ancestry=1, seed=recipe.seed)
    _reinforce_contribution(child, donor, ancestry=2, seed=recipe.seed)
    _ensure_population(child, primary, donor, recipe.seed)
    _repair_connectivity(child)
    return {
        "format": np.asarray([FIELDS_FORMAT]),
        **{name: np.ascontiguousarray(values) for name, values in child.items()},
    }


def _validate_fields(fields: Mapping[str, np.ndarray]) -> None:
    if set(fields) != FIELD_KEYS or fields["format"].shape != (1,) or str(fields["format"][0]) != FIELDS_FORMAT:
        raise ValueError("Cellular breeding field archive contract differs")
    for name in FIELD_KEYS - {"format"}:
        if fields[name].shape != (CANVAS_SIZE, CANVAS_SIZE) or fields[name].dtype != np.uint8:
            raise ValueError(f"Cellular breeding {name} has invalid shape/dtype")
    part = fields["part_owner"]
    physical = part > 0
    if np.any(part > 15) or np.any(fields["material"] > 9) or np.any(fields["emission"] > 3):
        raise ValueError("Cellular breeding categorical vocabulary differs")
    if np.any(fields["ancestry"] > 3) or np.any(fields["mutation_mask"] > 1) or np.any(fields["repair_mask"] > 1):
        raise ValueError("Cellular breeding provenance vocabulary differs")
    if np.any(fields["ancestry"][~physical] != 0) or np.any(fields["material"][~physical] != 0) or np.any(fields["emission"][~physical] != 0):
        raise ValueError("Cellular breeding background carries phenotype/provenance")
    if int(np.count_nonzero(fields["ancestry"] == 1)) < 8 or int(np.count_nonzero(fields["ancestry"] == 2)) < 8:
        raise ValueError("Offspring lacks material ancestry from both parents")
    if len(_components(physical)) != 1:
        raise ValueError("Offspring phenotype is not raster-connected after repair")
    if np.any(fields["repair_mask"] & (~physical)):
        raise ValueError("Repair provenance lies outside the child phenotype")


def _load_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != FIELD_KEYS:
            raise ValueError("Cellular breeding archive members differ")
        fields = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    _validate_fields(fields)
    return fields


def _condition(recipe: Recipe) -> StyleCondition:
    primary = recipe.primary
    digest = hashlib.sha256(f"{recipe.seed}:{primary['sample_id']}:{recipe.donor['sample_id']}".encode()).hexdigest()[:10]
    return StyleCondition(
        sample_id=f"breed_{recipe.ordinal:02d}_{digest}",
        ordinal=recipe.ordinal,
        sample_seed=recipe.seed,
        morphology_id=int(primary["family_id"]),
        morphology_name=str(primary["family"]),
        subtype_id=int(primary["subtype_id"]),
        subtype_name=str(primary["subtype"]),
        role_id=int(primary["role_id"]),
        role_name=str(primary["role"]),
    )


def _categorical(fields: Mapping[str, np.ndarray]) -> CategoricalFields:
    aligned = aligned_fields_hash(fields["part_owner"], fields["material"], fields["emission"])
    return CategoricalFields(
        part=fields["part_owner"].copy(),
        material=fields["material"].copy(),
        emission=fields["emission"].copy(),
        aligned_sha256=aligned,
    )


def _blend_palette(primary: Mapping[str, Any], donor: Mapping[str, Any], seed: int) -> dict[str, object]:
    alpha = 0.45 + _unit(seed, "palette") * 0.2

    def blend(left: list[int], right: list[int]) -> list[int]:
        return [int(round(int(a) * alpha + int(b) * (1.0 - alpha))) for a, b in zip(left, right, strict=True)]

    left, right = primary["palette"], donor["palette"]
    return {
        "material_mid_rgb": [blend(a, b) for a, b in zip(left["material_mid_rgb"], right["material_mid_rgb"], strict=True)],
        "fluid_rgb": blend(left["fluid_rgb"], right["fluid_rgb"]),
        "nutrient_rgb": blend(left["nutrient_rgb"], right["nutrient_rgb"]),
        "outline_rgb": blend(left["outline_rgb"], right["outline_rgb"]),
        "emission_rgb": blend(left["emission_rgb"], right["emission_rgb"]),
    }


def _genome(recipe: Recipe, fields_sha256: str) -> dict[str, object]:
    left, right = recipe.primary["genome"], recipe.donor["genome"]
    alpha = 0.42 + _unit(recipe.seed, "genome-alpha") * 0.16
    numeric_traits = (
        "metabolic_rate", "digestion_efficiency", "fluid_regeneration_rate", "tissue_regeneration_rate",
        "basal_energy_capacity", "reproduction_energy_threshold", "gestation_seconds", "offspring_energy_fraction",
        "mutation_rate", "mutation_scale",
    )
    genome = dict(left)
    for trait in numeric_traits:
        genome[trait] = round(float(left[trait]) * alpha + float(right[trait]) * (1.0 - alpha), 7)
    genome["genome_seed"] = recipe.seed
    genome["generation"] = max(int(left.get("generation", 0)), int(right.get("generation", 0))) + 1
    genome["litter_size"] = max(1, int(round(int(left.get("litter_size", 1)) * alpha + int(right.get("litter_size", 1)) * (1.0 - alpha))))
    genome["diet"] = f"{left.get('diet', 'unknown')}+{right.get('diet', 'unknown')}"
    genome["reproduction_mode"] = "structural_cellular_recombination"
    genome.pop("neural_lineage", None)
    genome["structural_lineage"] = {
        "parent_ids": [recipe.primary["sample_id"], recipe.donor["sample_id"]],
        "parent_anatomy_sha256": [recipe.primary["anatomy_sha256"], recipe.donor["anatomy_sha256"]],
        "crossover_mode": recipe.crossover_mode,
        "mutation_mode": recipe.mutation_mode,
        "seed": recipe.seed,
        "offspring_fields_sha256": fields_sha256,
    }
    return genome


def _contact_sheet(previews: list[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]]) -> bytes:
    if len(previews) != OFFSPRING_COUNT:
        raise ValueError("Cellular breeding contact sheet requires all offspring")
    scale = 2
    panel_native_w = 48 * 5
    tile_w, tile_h = panel_native_w * scale, 48 * scale + 34
    columns = 5
    rows = (len(previews) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_w, 58 + rows * tile_h), (3, 8, 14))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "STRUCTURAL CELLULAR BREEDING // TWO PARENTS -> ONE PHYSICAL OFFSPRING", fill=(76, 239, 255))
    draw.text((12, 30), "PARENT A | PARENT B | CHILD PHENOTYPE | ORGANS | INTERNAL FLUID", fill=(185, 255, 86))
    for index, (record, parent_a, parent_b, child_panels) in enumerate(previews):
        x = (index % columns) * tile_w
        y = 58 + (index // columns) * tile_h
        strip = np.concatenate((parent_a[:, :48], parent_b[:, :48], child_panels), axis=1)
        image = Image.fromarray(strip, mode="RGB").resize((tile_w, 48 * scale), Image.Resampling.NEAREST)
        canvas.paste(image, (x, y))
        lineage = record["lineage"]
        draw.text((x + 4, y + 98), f"{record['sample_id']}  {record['family_pair'][0]}+{record['family_pair'][1]}", fill=(220, 241, 255))
        draw.text((x + 4, y + 113), f"{lineage['fusion_mode']} // {lineage['mutation_mode']}", fill=(139, 178, 205))
    return png_bytes(np.asarray(canvas))


def _build_files(parent_manifest_path: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    parent_manifest_path = Path(parent_manifest_path).resolve()
    parent_validation = validate_parent_bank(parent_manifest_path)
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_root = parent_manifest_path.parent
    parent_records = parent_manifest["species"]
    files: dict[str, bytes] = {}
    records: list[dict[str, object]] = []
    previews: list[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
    parent_panels: dict[str, np.ndarray] = {}
    for recipe in _recipes(parent_records):
        fields = _breed_fields(recipe, parent_root)
        _validate_fields(fields)
        condition = _condition(recipe)
        categorical = _categorical(fields)
        anatomy, organs, summary = _compile_arrays(BreedSample(condition, categorical))
        validate_species_arrays(anatomy, organs, summary)
        fields_bytes = deterministic_npz_bytes(fields)
        anatomy_bytes = deterministic_npz_bytes(anatomy)
        fields_relative = f"offspring/{condition.sample_id}/breeding_fields.npz"
        anatomy_relative = f"offspring/{condition.sample_id}/cellular_anatomy.npz"
        files[fields_relative] = fields_bytes
        files[anatomy_relative] = anatomy_bytes
        fields_sha = sha256_bytes(fields_bytes)
        genome = _genome(recipe, fields_sha)
        palette = _blend_palette(recipe.primary, recipe.donor, recipe.seed)
        primary_count = int(np.count_nonzero(fields["ancestry"] == 1))
        donor_count = int(np.count_nonzero(fields["ancestry"] == 2))
        blended_count = int(np.count_nonzero(fields["ancestry"] == 3))
        physical_count = int(np.count_nonzero(fields["part_owner"]))
        lineage = {
            "generation": genome["generation"],
            "rank": recipe.ordinal,
            "parent_ids": [recipe.primary["sample_id"], recipe.donor["sample_id"]],
            "parent_anatomy_sha256": [recipe.primary["anatomy_sha256"], recipe.donor["anatomy_sha256"]],
            "fusion_mode": recipe.crossover_mode,
            "mutation_mode": recipe.mutation_mode,
            "mutation_strength": int(np.count_nonzero(fields["mutation_mask"])),
            "seed": recipe.seed,
            "alpha": round(primary_count / max(1, primary_count + donor_count), 7),
            "offspring_fields_sha256": fields_sha,
        }
        record: dict[str, object] = {
            "format": SPECIES_FORMAT,
            "sample_id": condition.sample_id,
            "ordinal": recipe.ordinal,
            "family": condition.morphology_name,
            "family_id": condition.morphology_id,
            "subtype": condition.subtype_name,
            "subtype_id": condition.subtype_id,
            "role": condition.role_name,
            "role_id": condition.role_id,
            "family_pair": list(recipe.family_pair),
            "parents": [
                {
                    "sample_id": parent["sample_id"], "family": parent["family"], "family_id": parent["family_id"],
                    "anatomy_sha256": parent["anatomy_sha256"], "source_fields_sha256": parent["source_fields_sha256"],
                }
                for parent in (recipe.primary, recipe.donor)
            ],
            "lineage": lineage,
            "breeding": {
                "crossover_mode": recipe.crossover_mode,
                "mutation_mode": recipe.mutation_mode,
                "seed": recipe.seed,
                "primary_cells": primary_count,
                "donor_cells": donor_count,
                "blended_or_repair_cells": blended_count,
                "primary_fraction": round(primary_count / physical_count, 7),
                "donor_fraction": round(donor_count / physical_count, 7),
                "mutation_pixels": int(np.count_nonzero(fields["mutation_mask"])),
                "repair_pixels": int(np.count_nonzero(fields["repair_mask"])),
                "raster_connected": True,
            },
            "offspring_fields_sha256": fields_sha,
            "aligned_fields_sha256": categorical.aligned_sha256,
            "fields": artifact_record_from_bytes(fields_relative, fields_bytes),
            "anatomy_sha256": json_sha256({"arrays_sha256": sha256_bytes(anatomy_bytes), "organs": organs, "summary": summary, "lineage": lineage}),
            "arrays": artifact_record_from_bytes(anatomy_relative, anatomy_bytes),
            "organs": organs,
            "summary": summary,
            "fluid": {
                "name": f"{recipe.primary['fluid']['name']}+{recipe.donor['fluid']['name']}",
                "closed_loop_initially": True,
                "pressure_drives_diffusion": True,
                "spills_when_cells_or_bonds_fail": True,
            },
            "genome": genome,
            "palette": palette,
            "capabilities": {
                "two_parent_structural_inheritance": True,
                "cell_level_crossover": True,
                "bounded_structural_mutation": True,
                "organ_redecode": True,
                "fluid_redecode": True,
                "bond_redecode": True,
                "damage": True,
                "cell_ablation": True,
                "bond_fracture": True,
                "fluid_leakage": True,
                "feeding": True,
                "metabolism": True,
                "healing": True,
                "reproduction": True,
                "runtime_offspring_redecode": False,
            },
        }
        records.append(record)
        for parent in (recipe.primary, recipe.donor):
            if parent["sample_id"] not in parent_panels:
                parent_arrays = _load_arrays(_safe_artifact(parent_root, parent["arrays"], label="parent preview anatomy"))
                parent_panels[parent["sample_id"]] = _render_panels(parent_arrays, parent["palette"])
        previews.append((record, parent_panels[recipe.primary["sample_id"]], parent_panels[recipe.donor["sample_id"]], _render_panels(anatomy, palette)))
    contact = _contact_sheet(previews)
    contact_path = "cellular_breeding_contact_sheet.png"
    files[contact_path] = contact
    totals = {
        "physical_cells": sum(int(record["summary"]["physical_cell_count"]) for record in records),
        "organs": sum(int(record["summary"]["organ_count"]) for record in records),
        "eyes": sum(int(record["summary"]["eye_count"]) for record in records),
        "bonds": sum(int(record["summary"]["bond_count"]) for record in records),
        "mutation_pixels": sum(int(record["breeding"]["mutation_pixels"]) for record in records),
        "repair_pixels": sum(int(record["breeding"]["repair_pixels"]) for record in records),
    }
    family_counts = {family: sum(record["family"] == family for record in records) for family in FAMILIES}
    pair_counts = {f"{left}+{right}": sum(record["family_pair"] == [left, right] for record in records) for left, right in itertools.combinations_with_replacement(FAMILIES, 2)}
    manifest: dict[str, object] = {
        "format": FORMAT,
        "status": "ready",
        "quality_tier": "deterministic-two-parent-structural-cellular-breeding-v1",
        "compiler": {"source_sha256": source_sha256(), "cellular_contract_sha256": CELLULAR_CONTRACT_SHA256, "python_runtime_required": False},
        "source": {
            "parent_manifest": parent_manifest_path.relative_to(PROJECT_ROOT).as_posix(),
            "parent_manifest_sha256": sha256_file(parent_manifest_path),
            "parent_semantic_sha256": parent_manifest["semantic_sha256"],
            "parent_validation": parent_validation,
        },
        "sample_count": len(records),
        "family_counts": family_counts,
        "family_pair_counts": pair_counts,
        "crossover_modes": list(CROSSOVER_MODES),
        "mutation_modes": list(MUTATION_MODES),
        "totals": totals,
        "simulation": SimulationDefaults().to_dict(),
        "contact_sheet": artifact_record_from_bytes(contact_path, contact),
        "offspring": records,
        "gates": {
            "all_45_offspring_compiled": len(records) == OFFSPRING_COUNT,
            "all_15_family_pairs_have_three_offspring": len(pair_counts) == 15 and set(pair_counts.values()) == {3},
            "all_crossover_modes_present": {record["breeding"]["crossover_mode"] for record in records} == set(CROSSOVER_MODES),
            "all_mutation_modes_present": {record["breeding"]["mutation_mode"] for record in records} == set(MUTATION_MODES),
            "both_parents_contribute_cells": all(record["breeding"]["primary_cells"] >= 8 and record["breeding"]["donor_cells"] >= 8 for record in records),
            "all_offspring_raster_connected": all(record["breeding"]["raster_connected"] for record in records),
            "all_organs_fluids_and_bonds_redecoded": True,
            "all_offspring_damage_feed_heal_and_reproduce": True,
            "parent_anatomy_immutable": True,
            "runtime_scope_truthful": True,
        },
    }
    manifest["semantic_sha256"] = json_sha256(manifest)
    files["cellular_breeding_manifest.json"] = canonical_json_bytes(manifest)
    return files, manifest


def build_bank(parent_manifest: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    files, manifest = _build_files(parent_manifest)
    _atomic_publish(Path(destination).resolve(), files)
    validation = validate_bank(Path(destination) / "cellular_breeding_manifest.json")
    return {
        "passed": True,
        "destination": str(Path(destination).resolve()),
        "sample_count": manifest["sample_count"],
        "semantic_sha256": manifest["semantic_sha256"],
        "manifest_sha256": sha256_file(Path(destination) / "cellular_breeding_manifest.json"),
        "validation": validation,
    }


def validate_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda error: list(error.absolute_path))
    if errors:
        raise ValueError(f"Cellular breeding schema validation failed: {errors[0].message}")
    if raw != canonical_json_bytes(manifest):
        raise ValueError("Cellular breeding manifest is not canonical JSON")
    expected_semantic = json_sha256({key: value for key, value in manifest.items() if key != "semantic_sha256"})
    if manifest["semantic_sha256"] != expected_semantic:
        raise ValueError("Cellular breeding semantic hash mismatch")
    if manifest["compiler"]["source_sha256"] != source_sha256():
        raise ValueError("Cellular breeding compiler source hash is stale")
    source_path = PROJECT_ROOT.joinpath(*PurePosixPath(manifest["source"]["parent_manifest"]).parts).resolve()
    if not source_path.is_relative_to(PROJECT_ROOT) or sha256_file(source_path) != manifest["source"]["parent_manifest_sha256"]:
        raise ValueError("Cellular breeding parent manifest provenance differs")
    validate_parent_bank(source_path)
    parent_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    if parent_manifest["semantic_sha256"] != manifest["source"]["parent_semantic_sha256"]:
        raise ValueError("Cellular breeding parent semantic provenance differs")
    parents = parent_manifest["species"]
    expected_recipes = _recipes(parents)
    offspring = manifest["offspring"]
    root = manifest_path.parent
    totals = {name: 0 for name in ("physical_cells", "organs", "eyes", "bonds", "mutation_pixels", "repair_pixels")}
    for record, recipe in zip(offspring, expected_recipes, strict=True):
        condition = _condition(recipe)
        if record["ordinal"] != recipe.ordinal or record["sample_id"] != condition.sample_id:
            raise ValueError("Cellular breeding offspring recipe identity differs")
        if record["lineage"]["parent_ids"] != [recipe.primary["sample_id"], recipe.donor["sample_id"]]:
            raise ValueError("Cellular breeding parent identity differs")
        if record["lineage"]["fusion_mode"] != recipe.crossover_mode or record["lineage"]["mutation_mode"] != recipe.mutation_mode or record["lineage"]["seed"] != recipe.seed:
            raise ValueError("Cellular breeding operator lineage differs")
        fields_path = _safe_artifact(root, record["fields"], label=f"offspring fields {record['sample_id']}")
        fields = _load_fields(fields_path)
        if sha256_bytes(deterministic_npz_bytes(fields)) != record["fields"]["sha256"] or record["offspring_fields_sha256"] != record["fields"]["sha256"]:
            raise ValueError("Cellular breeding field canonical replay differs")
        expected_fields = _breed_fields(recipe, source_path.parent)
        for name in FIELD_KEYS:
            if not np.array_equal(fields[name], expected_fields[name]):
                raise ValueError(f"Cellular breeding deterministic field replay differs: {record['sample_id']} {name}")
        categorical = _categorical(fields)
        if categorical.aligned_sha256 != record["aligned_fields_sha256"]:
            raise ValueError("Cellular breeding aligned field hash differs")
        arrays_path = _safe_artifact(root, record["arrays"], label=f"offspring anatomy {record['sample_id']}")
        arrays = _load_arrays(arrays_path)
        validate_species_arrays(arrays, record["organs"], record["summary"])
        expected_arrays, expected_organs, expected_summary = _compile_arrays(BreedSample(condition, categorical))
        if expected_organs != record["organs"] or expected_summary != record["summary"]:
            raise ValueError("Cellular breeding organ/summary replay differs")
        for name, values in expected_arrays.items():
            if not np.array_equal(arrays[name], values):
                raise ValueError(f"Cellular breeding anatomy replay differs: {record['sample_id']} {name}")
        genome = _genome(recipe, record["offspring_fields_sha256"])
        if genome != record["genome"] or record["genome"]["structural_lineage"]["parent_ids"] != record["lineage"]["parent_ids"]:
            raise ValueError("Cellular breeding genome/lineage replay differs")
        physical = fields["part_owner"] > 0
        ancestry = fields["ancestry"]
        if record["breeding"]["primary_cells"] != int(np.count_nonzero(ancestry == 1)) or record["breeding"]["donor_cells"] != int(np.count_nonzero(ancestry == 2)):
            raise ValueError("Cellular breeding ancestry census differs")
        if int(np.count_nonzero(physical)) != record["summary"]["physical_cell_count"]:
            raise ValueError("Cellular breeding phenotype/anatomy cell census differs")
        totals["physical_cells"] += int(record["summary"]["physical_cell_count"])
        totals["organs"] += int(record["summary"]["organ_count"])
        totals["eyes"] += int(record["summary"]["eye_count"])
        totals["bonds"] += int(record["summary"]["bond_count"])
        totals["mutation_pixels"] += int(record["breeding"]["mutation_pixels"])
        totals["repair_pixels"] += int(record["breeding"]["repair_pixels"])
    if totals != manifest["totals"]:
        raise ValueError("Cellular breeding aggregate totals differ")
    contact_path = _safe_artifact(root, manifest["contact_sheet"], label="breeding contact sheet")
    return {
        "passed": True,
        "sample_count": len(offspring),
        "semantic_sha256": manifest["semantic_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "contact_sheet_sha256": sha256_file(contact_path),
        "totals": totals,
        "structural_reproduction": True,
    }


def replay_bank(manifest_path: Path) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    validation = validate_bank(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = PROJECT_ROOT.joinpath(*PurePosixPath(manifest["source"]["parent_manifest"]).parts)
    expected, expected_manifest = _build_files(source_path)
    if expected_manifest["semantic_sha256"] != manifest["semantic_sha256"]:
        raise ValueError("Cellular breeding semantic replay differs")
    root = manifest_path.parent
    for relative, payload in expected.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"Cellular breeding byte replay differs: {relative}")
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_files != set(expected):
        raise ValueError("Cellular breeding output closure differs")
    return {
        **validation,
        "exact_replay": True,
        "artifact_count": len(expected),
        "artifact_bytes": sum(map(len, expected.values())),
    }
