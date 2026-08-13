from __future__ import annotations

"""Build the native Godot NeuralWorkshop asset bundle.

The workshop is intentionally a presentation/runtime consumer.  It compiles
the frozen neural static bank into compact layer atlases, copies the validated
topology-v2 map atlas contract, and exposes a deliberately fail-closed seam for
the derived neural-motion presentation bank.  Godot receives PNG and JSON
only; Python, NumPy, model checkpoints, and semantic archives remain build-time
dependencies.
"""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np
from PIL import Image
from jsonschema import Draft202012Validator

from .forge_lab_sync import (
    ASSET_INDEX_FORMAT as FORGE_LAB_INDEX_FORMAT,
    EXPECTED_FAMILIES,
    MAP_LAYERS as FORGE_LAB_MAP_LAYERS,
    MAP_THEMES,
    validate_synced_assets as validate_forge_lab_assets,
)
from .morphology import FACING_NAMES, MOTION_NAMES
from .morphology.constants import ROLE_NAMES
from .morphology.motion import DEFAULT_FRAME_COUNTS, LOOPING_MOTIONS
from .maps import load_map_pack


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATIC_SOURCE = (
    PROJECT_ROOT / "outputs" / "multifield_style" / "final_best_stratified80_v3"
)
DEFAULT_MAP_INDEX = PROJECT_ROOT / "game" / "generated" / "v2" / "asset_index.json"
DEFAULT_TOPOLOGY_SOURCE = PROJECT_ROOT / "outputs" / "maps_v2_forge_lab"
DEFAULT_NEURAL_MOTION_SOURCE = (
    PROJECT_ROOT / "outputs" / "multifield_style_neural_motion"
)
DEFAULT_DESTINATION = (
    PROJECT_ROOT / "game" / "generated" / "neural_workshop" / "v1"
)

INDEX_FORMAT = "nullvector-neural-workshop-assets-v1"
STATIC_BANK_FORMAT = "nullvector-multifield-style-bank-v1"
NEURAL_MOTION_BANK_FORMAT = "nullvector-multifield-style-neural-motion-bank-v1"
NEURAL_MOTION_IDENTITY_FORMAT = (
    "nullvector-multifield-style-neural-motion-identity-v1"
)
NEURAL_MOTION_REPLAY_FORMAT = (
    "nullvector-multifield-style-neural-motion-replay-v1"
)
STATIC_LAYERS = (
    "base",
    "outline",
    "emission_core",
    "aura",
    "bloom_r1",
    "bloom_r2",
    "composite",
)
TOPOLOGY_LAYERS = (
    "protected_backbone",
    "required_clearance",
    "decoration_forbidden",
    "walkability",
    "hazard_semantic",
    "zones",
    "nav_cost",
)
MAP_LAYERS = (*FORGE_LAB_MAP_LAYERS, *TOPOLOGY_LAYERS)
STATIC_COLUMNS = 16
STATIC_CELL_SIZE = 48
STATIC_SAMPLE_COUNT = 80
EXPECTED_SUBTYPE_COUNT = 20
NEURAL_MOTION_IDENTITY_COUNT = 5
NEURAL_MOTION_CLIP_COUNT = 520
NEURAL_MOTION_FRAME_COUNT = 4720
NEURAL_MOTION_COLUMNS = 16
NEURAL_MOTION_ROWS = 59
NEURAL_MOTION_ATLAS_SIZE = (
    NEURAL_MOTION_COLUMNS * STATIC_CELL_SIZE,
    NEURAL_MOTION_ROWS * STATIC_CELL_SIZE,
)
NEURAL_MOTION_BANK_SCHEMA = "multifield_style_neural_motion_bank.schema.json"
NEURAL_MOTION_IDENTITY_SCHEMA = "multifield_style_neural_motion_identity.schema.json"
NEURAL_MOTION_REPLAY_SCHEMA = "multifield_style_neural_motion_replay.schema.json"
EXPECTED_BRIDGE_SOURCE_SHA256 = (
    "46372e031c91d0202d0e55a8422385978c5157f76d83ed20adef9ed3e7250305"
)
EXPECTED_NEURAL_MOTION_REPRESENTATIVES = (
    ("humanoid", "0000_f0_s00_r0_v00", 0),
    ("animalian", "0016_f1_s04_r0_v00", 16),
    ("plantlike", "0032_f2_s08_r0_v00", 32),
    ("anomaly", "0048_f3_s12_r0_v00", 48),
    ("machine", "0064_f4_s16_r0_v00", 64),
)
EXPECTED_CENSUS_FAMILY_COUNTS = (
    {"family": "humanoid", "sample_count": 16, "bindable_count": 15, "rejected_count": 1},
    {"family": "animalian", "sample_count": 16, "bindable_count": 16, "rejected_count": 0},
    {"family": "plantlike", "sample_count": 16, "bindable_count": 13, "rejected_count": 3},
    {"family": "anomaly", "sample_count": 16, "bindable_count": 13, "rejected_count": 3},
    {"family": "machine", "sample_count": 16, "bindable_count": 13, "rejected_count": 3},
)
EXPECTED_CENSUS_CATEGORIES = (
    {"category": "anchor_on_background", "count": 3},
    {"category": "plant_topology", "count": 1},
    {"category": "required_owner_absence", "count": 3},
    {"category": "safety_margin", "count": 3},
)
EXPECTED_CENSUS_REJECTIONS = (
    ("humanoid", 12, "0012_f0_s02_r6_v00", "anchor_on_background"),
    ("plantlike", 8, "0040_f2_s08_r4_v00", "anchor_on_background"),
    ("plantlike", 12, "0044_f2_s10_r6_v00", "anchor_on_background"),
    ("plantlike", 15, "0047_f2_s11_r7_v01", "plant_topology"),
    ("anomaly", 7, "0055_f3_s15_r3_v01", "required_owner_absence"),
    ("anomaly", 8, "0056_f3_s12_r4_v00", "required_owner_absence"),
    ("anomaly", 9, "0057_f3_s12_r4_v01", "required_owner_absence"),
    ("machine", 3, "0067_f4_s17_r1_v01", "safety_margin"),
    ("machine", 8, "0072_f4_s16_r4_v00", "safety_margin"),
    ("machine", 15, "0079_f4_s19_r7_v01", "safety_margin"),
)
EXPECTED_BANK_GATES = (
    "actual_neural_samples_animated",
    "full_five_family_matrix_compiled",
    "raw_generation_provenance_exact",
    "static_style_parent_exact",
    "binding_source_exact",
    "motion_program_source_exact",
    "no_procedural_pixel_substitution",
    "categorical_authority_preserved",
    "rig_and_socket_authority_preserved",
    "motion_events_preserved",
    "palette_identity_invariant",
    "no_temporal_palette_flicker",
    "loop_endpoint_coherence",
    "outline_and_bloom_bounds_exact",
    "emission_pulse_support_bounded",
    "all_artifacts_hash_bound",
    "all_80_binding_census_exact",
    "exact_replay_ready",
    "ffmpeg_showcase_encoded",
)
EXPECTED_IDENTITY_GATES = (
    "raw_neural_fields_exact",
    "binding_exact",
    "motion_clips_valid",
    "source_tuples_preserved",
    "procedural_pixel_substitution_absent",
    "categorical_fields_unchanged_by_presentation",
    "rig_and_socket_authority_preserved",
    "motion_events_preserved",
    "palette_matches_static_neural_parent",
    "palette_identity_invariant",
    "no_temporal_palette_flicker",
    "loop_endpoints_exact",
    "outline_radius_1_exact",
    "bloom_radius_1_exact",
    "bloom_radius_2_exact",
    "effect_rings_unclipped",
    "emission_pulse_support_bounded",
)
SOURCE_IDENTITY_MANIFEST_SEMANTICS = (
    "verbatim-upstream-audit-copy; embedded artifact paths are "
    "upstream-root-relative and are not runtime-resolvable"
)
MIN_FREE_BYTES = 100 * 1024**3
PLANNED_BYTES = 256 * 1024**2
RUNTIME_EXTENSIONS = (".json", ".png")
PRESERVED_GAME_FILES = {
    "project.godot": "7397c7c032be468b94e072aa31ebaba42250342d5eae49fc9aa9972bffe245da",
    "Arena.tscn": "c5f8b961297b43f683d40dff831cb576d89539c222f0a5a9abab3d29a1f67490",
    "scripts/arena_game.gd": "0a5d2964cf9869bc292afada98ade98460cb25bcade1e50c4cd3a9766e419b22",
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_VARIANT_RE = re.compile(r"_v(?P<variant>[0-9]{2})$")


class SourceContractError(ValueError):
    """Raised when an upstream artifact cannot enter the runtime bundle."""


@dataclass(frozen=True, slots=True)
class SyncResult:
    destination: Path
    index_path: Path
    index: dict[str, Any]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


@lru_cache(maxsize=3)
def _manifest_validator(filename: str) -> Draft202012Validator:
    allowed = {
        NEURAL_MOTION_BANK_SCHEMA,
        NEURAL_MOTION_IDENTITY_SCHEMA,
        NEURAL_MOTION_REPLAY_SCHEMA,
    }
    if filename not in allowed:
        raise ValueError(f"Unsupported NeuralWorkshop source schema: {filename!r}")
    schema_path = PROJECT_ROOT / "shared" / "schema" / filename
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate_manifest_schema(
    payload: dict[str, Any],
    filename: str,
    *,
    label: str,
) -> None:
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SourceContractError(f"{label} is not finite JSON: {error}") from error
    errors = sorted(
        _manifest_validator(filename).iter_errors(payload),
        key=lambda error: tuple(str(value) for value in error.absolute_path),
    )
    if errors:
        rendered: list[str] = []
        for error in errors[:8]:
            location = "/".join(map(str, error.absolute_path)) or "<root>"
            rendered.append(f"{location}: {error.message}")
        suffix = f" (+{len(errors) - 8} more)" if len(errors) > 8 else ""
        raise SourceContractError(
            f"{label} failed {filename}: " + "; ".join(rendered) + suffix
        )


def _require_canonical_manifest(path: Path, payload: dict[str, Any], *, label: str) -> None:
    if path.read_bytes() != _canonical_json(payload):
        raise SourceContractError(f"{label} is not canonical JSON")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceContractError(f"Invalid JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SourceContractError(f"Expected a JSON object in {path}")
    return payload


def _write_if_changed(destination: Path, payload: bytes) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.read_bytes() == payload:
        return False
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return True


def _write_json(destination: Path, payload: Any) -> bool:
    return _write_if_changed(destination, _canonical_json(payload))


def _copy_if_changed(source: Path, destination: Path) -> bool:
    if destination.is_file() and _sha256_file(source) == _sha256_file(destination):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    return True


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _write_png(destination: Path, image: Image.Image) -> dict[str, Any]:
    payload = _png_bytes(image)
    _write_if_changed(destination, payload)
    return {
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _guard_disk(destination: Path) -> dict[str, int | bool]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(destination.parent)
    projected = usage.free - PLANNED_BYTES
    if projected < MIN_FREE_BYTES:
        raise RuntimeError(
            "NeuralWorkshop sync would breach the 100 GiB free-space floor: "
            f"free={usage.free}, planned={PLANNED_BYTES}, projected={projected}"
        )
    return {
        "guard_passed": True,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "planned_bytes": PLANNED_BYTES,
    }


def _safe_relative(value: Any, *, label: str) -> str:
    text = str(value)
    pure = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or pure.is_absolute()
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise SourceContractError(f"Unsafe {label} path: {text!r}")
    return pure.as_posix()


def _resolve_artifact(root: Path, value: Any, *, label: str) -> Path:
    relative = _safe_relative(value, label=label)
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise SourceContractError(f"{label} escapes its source root: {relative!r}")
    if not candidate.is_file():
        raise SourceContractError(f"Missing {label}: {candidate}")
    return candidate


def _require_sha(value: Any, *, label: str) -> str:
    text = str(value)
    if _SHA_RE.fullmatch(text) is None:
        raise SourceContractError(f"Invalid SHA-256 for {label}: {text!r}")
    return text


def _validate_artifact(
    root: Path,
    record: dict[str, Any],
    *,
    label: str,
) -> Path:
    if not isinstance(record, dict):
        raise SourceContractError(f"{label} artifact record is not an object")
    path = _resolve_artifact(root, record.get("path", ""), label=label)
    expected_bytes = int(record.get("bytes", -1))
    expected_sha = _require_sha(record.get("sha256", ""), label=label)
    if path.stat().st_size != expected_bytes:
        raise SourceContractError(
            f"Byte count mismatch for {label}: {path.stat().st_size} != {expected_bytes}"
        )
    observed_sha = _sha256_file(path)
    if observed_sha != expected_sha:
        raise SourceContractError(
            f"SHA-256 mismatch for {label}: {observed_sha} != {expected_sha}"
        )
    return path


def _destination_relative(destination: Path, path: Path) -> str:
    return path.resolve().relative_to(destination.resolve()).as_posix()


def _version_token(source_sha256: str) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "source_sha256": source_sha256,
                "workshop_sync_source_sha256": _sha256_file(Path(__file__)),
                "contract": INDEX_FORMAT,
            }
        )
    )[:16]


def _inventory_record(destination: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _destination_relative(destination, path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _copy_recorded(
    source: Path,
    target: Path,
    destination: Path,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    _copy_if_changed(source, target)
    record = _inventory_record(destination, target)
    inventory.append(record)
    return record


def _family_catalog() -> list[dict[str, Any]]:
    return [
        {"id": index, "name": family}
        for index, family in enumerate(EXPECTED_FAMILIES)
    ]


def _validate_static_source(manifest: dict[str, Any], source_root: Path) -> list[dict[str, Any]]:
    if manifest.get("format") != STATIC_BANK_FORMAT:
        raise SourceContractError("Unsupported static neural presentation bank format")
    if manifest.get("status") != "ready":
        raise SourceContractError("Static neural presentation bank is not ready")
    if manifest.get("all_metrics_passed") is not True:
        raise SourceContractError("Static neural presentation bank has failed metrics")
    authority = dict(manifest.get("authority", {}))
    expected_authority = {
        "raw_categorical_fields_remain_authoritative": True,
        "presentation_is_derived_only": True,
        "rig_authority_modified": False,
        "collision_authority_modified": False,
        "aura_is_effect_not_body": True,
    }
    for key, expected in expected_authority.items():
        if authority.get(key) is not expected:
            raise SourceContractError(f"Static neural authority gate failed: {key}")

    samples = list(manifest.get("samples", []))
    if int(manifest.get("sample_count", -1)) != STATIC_SAMPLE_COUNT:
        raise SourceContractError("Static neural source sample count is not 80")
    if len(samples) != STATIC_SAMPLE_COUNT:
        raise SourceContractError("Static neural source does not contain 80 sample records")

    seen_ids: set[str] = set()
    coverage: dict[tuple[str, str], int] = {}
    subtype_ids: set[int] = set()
    role_names: set[str] = set()
    for ordinal, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise SourceContractError(f"Static sample {ordinal} is not an object")
        condition = dict(sample.get("condition", {}))
        presentation = dict(sample.get("presentation", {}))
        sample_id = str(condition.get("sample_id", ""))
        if not sample_id or sample_id in seen_ids:
            raise SourceContractError(f"Duplicate or empty static sample id {sample_id!r}")
        seen_ids.add(sample_id)
        if int(condition.get("ordinal", -1)) != ordinal:
            raise SourceContractError(f"Static sample ordinal drift for {sample_id}")
        family_id = int(condition.get("morphology_id", -1))
        family = str(condition.get("morphology_name", ""))
        if not 0 <= family_id < len(EXPECTED_FAMILIES):
            raise SourceContractError(f"Invalid family id for {sample_id}")
        if family != EXPECTED_FAMILIES[family_id]:
            raise SourceContractError(f"Family id/name mismatch for {sample_id}")
        subtype_id = int(condition.get("subtype_id", -1))
        subtype_name = str(condition.get("subtype_name", ""))
        expected_subtype_name = f"{family}_{subtype_id - family_id * 4}"
        if not family_id * 4 <= subtype_id < family_id * 4 + 4:
            raise SourceContractError(f"Subtype id is outside family range for {sample_id}")
        if subtype_name != expected_subtype_name:
            raise SourceContractError(f"Subtype id/name mismatch for {sample_id}")
        role_id = int(condition.get("role_id", -1))
        role_name = str(condition.get("role_name", ""))
        if not 0 <= role_id < len(ROLE_NAMES) or role_name != ROLE_NAMES[role_id]:
            raise SourceContractError(f"Role id/name mismatch for {sample_id}")
        variant_match = _VARIANT_RE.search(sample_id)
        if variant_match is None or int(variant_match.group("variant")) not in (0, 1):
            raise SourceContractError(f"Static sample has invalid variant suffix: {sample_id}")
        if presentation.get("metrics_passed") is not True:
            raise SourceContractError(f"Static presentation metrics failed for {sample_id}")
        if presentation.get("layer_format") != "nullvector-multifield-style-layers-v1":
            raise SourceContractError(f"Static layer contract drift for {sample_id}")
        artifacts = dict(presentation.get("artifacts", {}))
        for layer in STATIC_LAYERS:
            path = _validate_artifact(
                source_root,
                dict(artifacts.get(layer, {})),
                label=f"static {sample_id}/{layer}",
            )
            with Image.open(path) as image:
                if image.size != (STATIC_CELL_SIZE, STATIC_CELL_SIZE):
                    raise SourceContractError(
                        f"Static layer has invalid dimensions for {sample_id}/{layer}: {image.size}"
                    )
                if image.mode != "RGBA":
                    raise SourceContractError(
                        f"Static layer has invalid mode for {sample_id}/{layer}: {image.mode}"
                    )
                image.verify()
        coverage[(family, role_name)] = coverage.get((family, role_name), 0) + 1
        subtype_ids.add(subtype_id)
        role_names.add(role_name)

    expected_coverage = {
        (family, role): 2 for family in EXPECTED_FAMILIES for role in ROLE_NAMES
    }
    if coverage != expected_coverage:
        raise SourceContractError("Static bank is not balanced at two variants per family/role")
    if subtype_ids != set(range(EXPECTED_SUBTYPE_COUNT)):
        raise SourceContractError("Static bank does not cover all 20 subtypes")
    if role_names != set(ROLE_NAMES):
        raise SourceContractError("Static bank does not cover all eight roles")
    return samples


def _sync_static_bank(
    source_root: Path,
    destination: Path,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = source_root / "style_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _load_json(manifest_path)
    samples = _validate_static_source(manifest, source_root)
    manifest_sha = _sha256_file(manifest_path)
    version_root = destination / "static" / _version_token(manifest_sha)
    copied_manifest = _copy_recorded(
        manifest_path,
        version_root / "source_manifest.json",
        destination,
        inventory,
    )

    rows = (len(samples) + STATIC_COLUMNS - 1) // STATIC_COLUMNS
    atlas_entries: list[dict[str, Any]] = []
    for layer in STATIC_LAYERS:
        atlas = Image.new(
            "RGBA",
            (STATIC_COLUMNS * STATIC_CELL_SIZE, rows * STATIC_CELL_SIZE),
            (0, 0, 0, 0),
        )
        for cell, sample in enumerate(samples):
            record = dict(sample["presentation"]["artifacts"][layer])
            path = _validate_artifact(
                source_root,
                record,
                label=f"static {sample['condition']['sample_id']}/{layer}",
            )
            with Image.open(path) as source:
                frame = source.copy()
            atlas.paste(
                frame,
                (
                    (cell % STATIC_COLUMNS) * STATIC_CELL_SIZE,
                    (cell // STATIC_COLUMNS) * STATIC_CELL_SIZE,
                ),
            )
        atlas_path = version_root / f"{layer}_atlas.png"
        artifact = _write_png(atlas_path, atlas)
        artifact.update(
            {
                "path": _destination_relative(destination, atlas_path),
                "layer": layer,
                "size": list(atlas.size),
                "columns": STATIC_COLUMNS,
                "rows": rows,
                "cell_size": STATIC_CELL_SIZE,
            }
        )
        inventory.append(
            {
                "path": artifact["path"],
                "bytes": artifact["bytes"],
                "sha256": artifact["sha256"],
            }
        )
        atlas_entries.append(artifact)

    identities: list[dict[str, Any]] = []
    for cell, sample in enumerate(samples):
        condition = dict(sample["condition"])
        presentation = dict(sample["presentation"])
        source = dict(sample["source"])
        variant_match = _VARIANT_RE.search(str(condition["sample_id"]))
        assert variant_match is not None
        identities.append(
            {
                "cell": cell,
                "sample_id": condition["sample_id"],
                "ordinal": condition["ordinal"],
                "sample_seed": condition["sample_seed"],
                "family_id": condition["morphology_id"],
                "family": condition["morphology_name"],
                "subtype_id": condition["subtype_id"],
                "subtype": condition["subtype_name"],
                "role_id": condition["role_id"],
                "role": condition["role_name"],
                "variant": int(variant_match.group("variant")),
                "raw_fields_sha256": source["raw_fields_sha256"],
                "compiled_fields_sha256": source["compiled_fields_sha256"],
                "metrics_sha256": presentation["artifacts"]["metrics"]["sha256"],
                "palette_sha256": presentation["artifacts"]["palette"]["sha256"],
                "source_layer_sha256": {
                    layer: presentation["artifacts"][layer]["sha256"]
                    for layer in STATIC_LAYERS
                },
            }
        )

    subtype_catalog = [
        {
            "id": subtype_id,
            "name": f"{EXPECTED_FAMILIES[subtype_id // 4]}_{subtype_id % 4}",
            "family": EXPECTED_FAMILIES[subtype_id // 4],
        }
        for subtype_id in range(EXPECTED_SUBTYPE_COUNT)
    ]
    return {
        "status": "ready",
        "source_format": STATIC_BANK_FORMAT,
        "source_manifest": copied_manifest,
        "source_compiler": dict(manifest["compiler"]),
        "source_parent": dict(manifest["parent"]),
        "authority": dict(manifest["authority"]),
        "layers": list(STATIC_LAYERS),
        "families": _family_catalog(),
        "subtypes": subtype_catalog,
        "roles": [
            {"id": role_id, "name": role_name}
            for role_id, role_name in enumerate(ROLE_NAMES)
        ],
        "identity_count": len(identities),
        "identities": identities,
        "atlases": atlas_entries,
        "layout": {
            "cell_size": STATIC_CELL_SIZE,
            "columns": STATIC_COLUMNS,
            "rows": rows,
            "native_scale_options": [1, 4],
            "pixel_filter": "nearest",
        },
    }


def _topology_manifest_for_map(topology_root: Path, map_id: str) -> Path:
    direct = topology_root / map_id / "manifest.json"
    if direct.is_file():
        return direct
    matches = sorted(topology_root.glob(f"{map_id}/manifest.json"))
    if len(matches) != 1:
        raise SourceContractError(
            f"Expected exactly one topology-v2 manifest for {map_id}, observed {len(matches)}"
        )
    return matches[0]


def _validate_topology_manifest(
    path: Path,
    *,
    map_id: str,
    theme: str,
    semantic_sha256: str,
) -> dict[str, Any]:
    manifest = _load_json(path)
    if manifest.get("schema_version") != "2.0.0":
        raise SourceContractError(f"Topology source is not schema v2 for {map_id}")
    if manifest.get("map_id") != map_id or manifest.get("theme") != theme:
        raise SourceContractError(f"Topology source identity mismatch for {map_id}")
    if manifest.get("semantic_array_sha256") != semantic_sha256:
        raise SourceContractError(f"Topology semantic hash mismatch for {map_id}")
    topology = dict(manifest.get("topology", {}))
    invariants = list(topology.get("invariants", []))
    if not invariants or any(entry.get("passed") is not True for entry in invariants):
        raise SourceContractError(f"Topology-v2 invariants failed for {map_id}")
    invariant_names = {str(entry.get("name", "")) for entry in invariants}
    required = {
        "topology.protected_backbone_connected",
        "topology.required_hazard_free_connected",
        "topology.agent_radius_one_connected",
        "safety.required_clearance_hazard_free",
        "safety.decoration_forbidden_exact_union",
    }
    missing = sorted(required - invariant_names)
    if missing:
        raise SourceContractError(
            f"Topology-v2 source lacks workshop safety invariants for {map_id}: {missing}"
        )
    return manifest


def _rgb_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(array, dtype=np.uint8))


def _mask_layer(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    output[:] = (3, 7, 15)
    output[np.asarray(mask) != 0] = color
    return _rgb_image(output)


def _topology_debug_layers(map_data: Any) -> list[tuple[str, Image.Image]]:
    arrays = map_data.arrays()
    shape = arrays["walkability"].shape
    layers: list[tuple[str, Image.Image]] = [
        ("protected_backbone", _mask_layer(arrays["protected_backbone"], (47, 236, 255))),
        ("required_clearance", _mask_layer(arrays["required_clearance"], (186, 255, 87))),
        ("decoration_forbidden", _mask_layer(arrays["decoration_forbidden"], (255, 63, 180))),
        ("walkability", _mask_layer(arrays["walkability"], (60, 220, 170))),
    ]

    hazard = np.asarray(arrays["hazard"], dtype=np.int64)
    hazard_rgb = np.zeros((*shape, 3), dtype=np.uint8)
    hazard_rgb[:] = (3, 7, 15)
    hazard_palette = np.asarray(
        [
            (3, 7, 15),
            (255, 76, 126),
            (255, 178, 62),
            (163, 81, 255),
            (48, 233, 255),
        ],
        dtype=np.uint8,
    )
    hazard_rgb[:] = hazard_palette[np.clip(hazard, 0, len(hazard_palette) - 1)]
    layers.append(("hazard_semantic", _rgb_image(hazard_rgb)))

    zones = np.asarray(arrays["zone"], dtype=np.int64)
    zone_rgb = np.zeros((*shape, 3), dtype=np.uint8)
    zone_rgb[:] = (3, 7, 15)
    valid_zones = zones >= 0
    # Integer coordinate hashing gives stable, high-separation debug colors
    # without introducing random state into the runtime contract.
    zone_values = zones[valid_zones].astype(np.uint64)
    zone_rgb[valid_zones, 0] = ((zone_values * 73 + 53) % 196 + 40).astype(np.uint8)
    zone_rgb[valid_zones, 1] = ((zone_values * 151 + 97) % 196 + 40).astype(np.uint8)
    zone_rgb[valid_zones, 2] = ((zone_values * 211 + 149) % 196 + 40).astype(np.uint8)
    layers.append(("zones", _rgb_image(zone_rgb)))

    nav = np.asarray(arrays["nav_cost"], dtype=np.float32)
    nav_rgb = np.zeros((*shape, 3), dtype=np.uint8)
    nav_rgb[:] = (3, 7, 15)
    traversable = nav > 0
    if bool(traversable.any()):
        values = nav[traversable]
        minimum = float(values.min())
        maximum = float(values.max())
        normalized = np.zeros_like(values, dtype=np.float32)
        if maximum > minimum:
            normalized = (values - minimum) / (maximum - minimum)
        nav_rgb[traversable, 0] = np.rint(38 + normalized * 217).astype(np.uint8)
        nav_rgb[traversable, 1] = np.rint(236 - normalized * 160).astype(np.uint8)
        nav_rgb[traversable, 2] = np.rint(255 - normalized * 60).astype(np.uint8)
    layers.append(("nav_cost", _rgb_image(nav_rgb)))
    if tuple(name for name, _ in layers) != TOPOLOGY_LAYERS:
        raise AssertionError("Topology debug layer packing order drifted")
    return layers


def _sync_map_bank(
    source_index_path: Path,
    topology_root: Path,
    destination: Path,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    validation_errors = validate_forge_lab_assets(source_index_path)
    if validation_errors:
        raise SourceContractError(
            "ForgeLab map source index failed validation: " + "; ".join(validation_errors)
        )
    source_index = _load_json(source_index_path)
    if source_index.get("format") != FORGE_LAB_INDEX_FORMAT:
        raise SourceContractError("Unsupported ForgeLab source index format")
    maps = dict(source_index.get("maps", {}))
    if maps.get("status") != "ready":
        raise SourceContractError("ForgeLab map source is not ready")
    if tuple(maps.get("themes", [])) != tuple(MAP_THEMES):
        raise SourceContractError("Map theme order drifted from the public contract")
    if tuple(maps.get("layers", [])) != tuple(FORGE_LAB_MAP_LAYERS):
        raise SourceContractError("Map layer order drifted from the public contract")
    source_root = source_index_path.parent
    source_index_sha = _sha256_file(source_index_path)
    version_root = destination / "maps" / _version_token(source_index_sha)
    copied_source_index = _copy_recorded(
        source_index_path,
        version_root / "source_index.json",
        destination,
        inventory,
    )

    output_entries: list[dict[str, Any]] = []
    source_entries = list(maps.get("maps", []))
    if len(source_entries) != len(MAP_THEMES):
        raise SourceContractError("ForgeLab map source does not contain six themes")
    for expected_theme, entry_value in zip(MAP_THEMES, source_entries, strict=True):
        entry = dict(entry_value)
        theme = str(entry.get("theme", ""))
        if theme != expected_theme:
            raise SourceContractError(f"Map entry order mismatch at {expected_theme}")
        map_id = str(entry.get("map_id", ""))
        semantic_sha = _require_sha(
            entry.get("source_semantic_sha256", ""),
            label=f"map semantic {theme}",
        )
        topology_path = _topology_manifest_for_map(topology_root, map_id)
        topology_manifest = _validate_topology_manifest(
            topology_path,
            map_id=map_id,
            theme=theme,
            semantic_sha256=semantic_sha,
        )

        atlas_source = _resolve_artifact(
            source_root,
            entry.get("atlas", ""),
            label=f"map atlas {theme}",
        )
        atlas_sha = _require_sha(entry.get("atlas_sha256", ""), label=f"map atlas {theme}")
        if _sha256_file(atlas_source) != atlas_sha:
            raise SourceContractError(f"Map atlas source hash mismatch for {theme}")
        with Image.open(atlas_source) as image:
            if list(image.size) != list(entry.get("atlas_size", [])):
                raise SourceContractError(f"Map atlas dimensions drifted for {theme}")
            image.verify()
        art_manifest_source = _resolve_artifact(
            source_root,
            entry.get("source_manifest", ""),
            label=f"map art manifest {theme}",
        )
        art_manifest_sha = _require_sha(
            entry.get("source_manifest_sha256", ""),
            label=f"map art manifest {theme}",
        )
        if _sha256_file(art_manifest_source) != art_manifest_sha:
            raise SourceContractError(f"Map art manifest hash mismatch for {theme}")

        art_manifest_record = _copy_recorded(
            art_manifest_source,
            version_root / "art" / f"{theme}.json",
            destination,
            inventory,
        )
        topology_record = _copy_recorded(
            topology_path,
            version_root / "topology" / f"{theme}.json",
            destination,
            inventory,
        )
        layers = [dict(layer) for layer in entry.get("layers", [])]
        if tuple(layer.get("name") for layer in layers) != tuple(FORGE_LAB_MAP_LAYERS):
            raise SourceContractError(f"Map layer matrix drifted for {theme}")
        cell_size = int(entry.get("cell_size", 0))
        columns = int(entry.get("columns", 0))
        source_rows = int(entry.get("rows", 0))
        if cell_size != 384 or columns != 4 or source_rows < 1:
            raise SourceContractError(f"Map atlas grid contract drifted for {theme}")
        for layer in layers:
            start = int(layer.get("start_cell", -1))
            count = int(layer.get("frame_count", 0))
            if start < 0 or count < 1 or start + count > columns * source_rows:
                raise SourceContractError(f"Map layer region is invalid for {theme}/{layer.get('name')}")
        map_data = load_map_pack(topology_path.parent, verify_hashes=True)
        if map_data.map_id != map_id or map_data.theme != theme:
            raise SourceContractError(f"Loaded topology-v2 map identity drifted for {theme}")
        topology_layers = _topology_debug_layers(map_data)
        first_topology_cell = columns * source_rows
        for offset, (name, _image) in enumerate(topology_layers):
            layers.append(
                {
                    "name": name,
                    "start_cell": first_topology_cell + offset,
                    "frame_count": 1,
                    "fps": 0,
                }
            )
        if tuple(layer["name"] for layer in layers) != MAP_LAYERS:
            raise AssertionError("Workshop map layer packing order drifted")
        atlas_cell_count = first_topology_cell + len(topology_layers)
        rows = (atlas_cell_count + columns - 1) // columns
        atlas_size = (columns * cell_size, rows * cell_size)
        compiled_atlas = Image.new("RGB", atlas_size, (3, 7, 15))
        with Image.open(atlas_source) as source_atlas:
            compiled_atlas.paste(source_atlas.convert("RGB"), (0, 0))
        for offset, (_name, image) in enumerate(topology_layers):
            cell = first_topology_cell + offset
            compiled_atlas.paste(
                image.resize((cell_size, cell_size), Image.Resampling.NEAREST),
                ((cell % columns) * cell_size, (cell // columns) * cell_size),
            )
        atlas_path = version_root / "atlases" / f"{theme}.png"
        atlas_record = _write_png(atlas_path, compiled_atlas)
        atlas_record["path"] = _destination_relative(destination, atlas_path)
        inventory.append(dict(atlas_record))
        output_entries.append(
            {
                "theme": theme,
                "map_id": map_id,
                "art_id": entry["art_id"],
                "seed": entry["seed"],
                "source_semantic_sha256": semantic_sha,
                "renderer": dict(entry["renderer"]),
                "statistics": dict(entry["statistics"]),
                "atlas": atlas_record,
                "atlas_size": list(atlas_size),
                "source_atlas_sha256": atlas_sha,
                "source_atlas_size": list(entry["atlas_size"]),
                "cell_size": cell_size,
                "columns": columns,
                "rows": rows,
                "layers": layers,
                "art_manifest": art_manifest_record,
                "topology_manifest": topology_record,
                "topology_contract": {
                    "schema_version": topology_manifest["schema_version"],
                    "semantic_array_sha256": topology_manifest["semantic_array_sha256"],
                    "protected_backbone_segments": topology_manifest["topology"][
                        "protected_backbone_segments"
                    ],
                    "start_exit_path_length": topology_manifest["topology"][
                        "start_exit_path_length"
                    ],
                    "invariant_count": len(topology_manifest["topology"]["invariants"]),
                    "all_invariants_passed": True,
                },
            }
        )
    return {
        "status": "ready",
        "source_format": FORGE_LAB_INDEX_FORMAT,
        "source_index": copied_source_index,
        "renderer_source_sha256": maps["renderer_source_sha256"],
        "themes": list(MAP_THEMES),
        "layers": list(MAP_LAYERS),
        "map_count": len(output_entries),
        "maps": output_entries,
        "topology_schema_version": "2.0.0",
        "pixel_filter": "nearest",
    }


def _staged_motion_contract(status: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "status": status,
        "available": False,
        "neural_output": False,
        "reasons": reasons,
        "source_root": "outputs/multifield_style_neural_motion",
        "expected": {
            "bank_format": NEURAL_MOTION_BANK_FORMAT,
            "identity_format": NEURAL_MOTION_IDENTITY_FORMAT,
            "replay_format": NEURAL_MOTION_REPLAY_FORMAT,
            "bank_status": "ready",
            "verification_status": "passed",
            "neural_output": True,
            "identity_count": NEURAL_MOTION_IDENTITY_COUNT,
            "clip_count": NEURAL_MOTION_CLIP_COUNT,
            "frame_count": NEURAL_MOTION_FRAME_COUNT,
            "source_sample_count": STATIC_SAMPLE_COUNT,
            "bindable_count": 70,
            "rejected_count": 10,
            "representatives": [
                {
                    "family": family,
                    "sample_id": sample_id,
                    "static_cell": cell,
                }
                for family, sample_id, cell in EXPECTED_NEURAL_MOTION_REPRESENTATIVES
            ],
        },
        "layers": list(STATIC_LAYERS),
        "motions": list(MOTION_NAMES),
        "facings": list(FACING_NAMES),
        "identities": [],
        "clip_count": 0,
        "frame_count": 0,
        "fail_closed": True,
    }


def _bank_identity_records(bank: dict[str, Any]) -> list[dict[str, Any]]:
    raw = bank.get("identities")
    if not isinstance(raw, list):
        # The public bank contract may call these family records.  Supporting
        # both names is explicit, while every entry below is still validated
        # against the frozen identity manifest contract.
        raw = bank.get("families")
    if not isinstance(raw, list):
        raise SourceContractError("Neural motion bank has no identity manifest records")
    records = [dict(value) for value in raw if isinstance(value, dict)]
    if len(records) != len(raw):
        raise SourceContractError("Neural motion identity registry contains a non-object")
    return records


def _validate_neural_motion_census(
    census: dict[str, Any],
    static_identities: list[dict[str, Any]],
) -> None:
    if len(static_identities) != STATIC_SAMPLE_COUNT or any(
        not isinstance(identity, dict) for identity in static_identities
    ):
        raise SourceContractError("Static bank is unavailable for neural motion census binding")
    if (
        census.get("format") != "nullvector-neural-rig-binding-census-v1"
        or census.get("scope") != "all-80-immutable-production-samples"
        or int(census.get("sample_count", -1)) != STATIC_SAMPLE_COUNT
        or int(census.get("bindable_count", -1)) != 70
        or int(census.get("rejected_count", -1)) != 10
    ):
        raise SourceContractError("Neural motion all-80 source census headline drifted")
    family_counts = tuple(dict(value) for value in census.get("family_counts", []))
    if family_counts != EXPECTED_CENSUS_FAMILY_COUNTS:
        raise SourceContractError("Neural motion source census family counts drifted")
    categories = tuple(dict(value) for value in census.get("rejection_categories", []))
    if categories != EXPECTED_CENSUS_CATEGORIES:
        raise SourceContractError("Neural motion source census rejection categories drifted")

    rejections = [dict(value) for value in census.get("rejections", [])]
    observed_rejections = tuple(
        (
            str(value.get("family", "")),
            int(value.get("candidate_ordinal_within_family", -1)),
            str(value.get("sample_id", "")),
            str(value.get("category", "")),
        )
        for value in rejections
    )
    if observed_rejections != EXPECTED_CENSUS_REJECTIONS:
        raise SourceContractError("Neural motion source census rejection registry drifted")
    if any(not str(value.get("reason", "")).strip() for value in rejections):
        raise SourceContractError("Neural motion source census contains an empty rejection reason")

    observed_category_counts = {
        category["category"]: sum(
            value.get("category") == category["category"] for value in rejections
        )
        for category in EXPECTED_CENSUS_CATEGORIES
    }
    expected_category_counts = {
        value["category"]: value["count"] for value in EXPECTED_CENSUS_CATEGORIES
    }
    if observed_category_counts != expected_category_counts:
        raise SourceContractError("Neural motion source census category evidence disagrees with counts")

    static_by_id = {
        str(identity.get("sample_id", "")): identity for identity in static_identities
    }
    if len(static_by_id) != STATIC_SAMPLE_COUNT:
        raise SourceContractError("Static bank is unavailable for neural motion census binding")
    for family, ordinal, sample_id, _category in observed_rejections:
        identity = static_by_id.get(sample_id)
        expected_global_ordinal = EXPECTED_FAMILIES.index(family) * 16 + ordinal
        if (
            identity is None
            or identity.get("family") != family
            or int(identity.get("ordinal", -1)) != expected_global_ordinal
            or int(identity.get("cell", -1)) != expected_global_ordinal
        ):
            raise SourceContractError(
                f"Neural motion census rejection is not bound to static identity {sample_id}"
            )

    animation_scope = dict(census.get("animation_bank_scope", {}))
    expected_scope = {
        "selected_identity_count": NEURAL_MOTION_IDENTITY_COUNT,
        "all_80_animated": False,
        "policy": "first-bank-ordered-full-matrix-valid-identity-per-family-v1",
        "binding_census_does_not_imply_animation": True,
    }
    if animation_scope != expected_scope:
        raise SourceContractError("Neural motion animation scope semantics drifted")


def _validate_neural_motion_bank(
    bank_path: Path,
    bank: dict[str, Any],
    static_identities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _validate_manifest_schema(
        bank,
        NEURAL_MOTION_BANK_SCHEMA,
        label="Neural motion bank manifest",
    )
    _require_canonical_manifest(bank_path, bank, label="Neural motion bank manifest")
    compiler = dict(bank.get("compiler", {}))
    if compiler.get("bridge_source_sha256") != EXPECTED_BRIDGE_SOURCE_SHA256:
        raise SourceContractError("Neural motion bank bridge source is not the frozen production bridge")
    if dict(bank.get("gates", {})) != {name: True for name in EXPECTED_BANK_GATES}:
        raise SourceContractError("Neural motion bank gate registry drifted")
    matrix = dict(bank.get("matrix", {}))
    if (
        matrix.get("families") != list(EXPECTED_FAMILIES)
        or matrix.get("motions") != list(MOTION_NAMES)
        or matrix.get("facings") != list(FACING_NAMES)
        or matrix.get("layers") != list(STATIC_LAYERS)
        or int(matrix.get("identity_count", -1)) != NEURAL_MOTION_IDENTITY_COUNT
        or int(matrix.get("clips_per_identity", -1)) != 104
        or int(matrix.get("frames_per_identity", -1)) != 944
        or int(matrix.get("clip_count", -1)) != NEURAL_MOTION_CLIP_COUNT
        or int(matrix.get("frame_count", -1)) != NEURAL_MOTION_FRAME_COUNT
    ):
        raise SourceContractError("Neural motion bank matrix drifted")
    _validate_neural_motion_census(dict(bank.get("source_census", {})), static_identities)
    identity_records = _bank_identity_records(bank)
    observed_representatives = tuple(
        (str(record.get("family", "")), str(record.get("sample_id", "")))
        for record in identity_records
    )
    expected_representatives = tuple(
        (family, sample_id)
        for family, sample_id, _cell in EXPECTED_NEURAL_MOTION_REPRESENTATIVES
    )
    if observed_representatives != expected_representatives:
        raise SourceContractError("Neural motion representative identity registry drifted")
    return identity_records


def _validate_neural_motion_replay(
    verification_path: Path,
    verification: dict[str, Any],
    *,
    bank_path: Path,
    bank: dict[str, Any],
) -> None:
    _validate_manifest_schema(
        verification,
        NEURAL_MOTION_REPLAY_SCHEMA,
        label="Neural motion replay report",
    )
    _require_canonical_manifest(
        verification_path,
        verification,
        label="Neural motion replay report",
    )
    bank_sha = _sha256_file(bank_path)
    try:
        expected_manifest_path = bank_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise SourceContractError(
            "Authoritative neural motion bank must be inside the project root"
        ) from error
    expected_manifest_record = {
        "path": expected_manifest_path,
        "bytes": bank_path.stat().st_size,
        "sha256": bank_sha,
    }
    if dict(verification.get("manifest", {})) != expected_manifest_record:
        raise SourceContractError("Neural motion replay report targets a different bank artifact")
    if verification.get("compiler_source_sha256") != bank["compiler"]["source_sha256"]:
        raise SourceContractError("Neural motion replay compiler does not match the bank compiler")
    expected_results = tuple(
        (family, sample_id, True, 12, 4)
        for family, sample_id, _cell in EXPECTED_NEURAL_MOTION_REPRESENTATIVES
    )
    identity_results = [dict(value) for value in verification.get("identity_results", [])]
    observed_results = tuple(
        (
            str(value.get("family", "")),
            str(value.get("sample_id", "")),
            value.get("exact"),
            int(value.get("artifact_count", -1)),
            int(value.get("shard_count", -1)),
        )
        for value in identity_results
    )
    if observed_results != expected_results:
        raise SourceContractError("Neural motion replay identity proof registry drifted")
    if sum(int(value.get("artifact_count", 0)) for value in identity_results) + 3 != 63:
        raise SourceContractError("Neural motion replay artifact proof does not close at 63 artifacts")
    if sum(int(value.get("bytes_compared", 0)) for value in identity_results) >= int(
        verification.get("bytes_compared", -1)
    ):
        # The report total must additionally include the three exact showcase artifacts.
        raise SourceContractError("Neural motion replay byte proof omits showcase artifacts")


def _identity_manifest_record(entry: dict[str, Any]) -> dict[str, Any]:
    for key in ("identity_manifest", "manifest"):
        record = entry.get(key)
        if isinstance(record, dict):
            return dict(record)
    raise SourceContractError("Neural motion identity registry lacks a manifest artifact")


def _validate_neural_motion_identity(
    source_root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    expected_family: str,
    expected_sample_id: str,
    expected_compiler: dict[str, Any],
) -> dict[str, Any]:
    _validate_manifest_schema(
        manifest,
        NEURAL_MOTION_IDENTITY_SCHEMA,
        label=f"Neural motion identity manifest {expected_family}",
    )
    _require_canonical_manifest(
        manifest_path,
        manifest,
        label=f"Neural motion identity manifest {expected_family}",
    )
    if manifest.get("format") != NEURAL_MOTION_IDENTITY_FORMAT:
        raise SourceContractError(f"Unsupported neural motion identity format for {expected_family}")
    if manifest.get("status") != "ready" or manifest.get("neural_output") is not True:
        raise SourceContractError(f"Neural motion identity is not authoritative for {expected_family}")
    family = str(manifest.get("family", ""))
    if family != expected_family:
        raise SourceContractError(f"Neural motion family mismatch: {family!r} != {expected_family!r}")
    sample_id = str(manifest.get("sample_id", ""))
    condition = dict(manifest.get("condition", {}))
    if (
        sample_id != expected_sample_id
        or condition.get("sample_id") != sample_id
        or dict(manifest.get("compiler", {})) != expected_compiler
    ):
        raise SourceContractError(f"Neural motion sample identity mismatch for {family}")
    family_id = int(condition.get("morphology_id", -1))
    if family_id != EXPECTED_FAMILIES.index(family):
        raise SourceContractError(f"Neural motion family id mismatch for {family}")
    authority = dict(manifest.get("authority", {}))
    authority_gates = {
        "raw_neural_fields_are_source_authority": True,
        "binding_and_motion_program_are_derived_authority": True,
        "presentation_is_derived_only": True,
        "procedural_pixel_substitution": False,
        "collision_authority_modified": False,
        "aura_is_effect_not_body": True,
    }
    for key, expected in authority_gates.items():
        if authority.get(key) is not expected:
            raise SourceContractError(f"Neural motion authority gate failed for {family}: {key}")
    gates = dict(manifest.get("gates", {}))
    if gates != {name: True for name in EXPECTED_IDENTITY_GATES}:
        raise SourceContractError(f"Neural motion identity gate failed for {family}")
    layout = dict(manifest.get("layout", {}))
    if (
        int(layout.get("cell_size", -1)) != STATIC_CELL_SIZE
        or int(layout.get("columns", -1)) != NEURAL_MOTION_COLUMNS
        or int(layout.get("rows", -1)) != NEURAL_MOTION_ROWS
        or int(layout.get("frame_count", -1)) != 944
        or tuple(layout.get("layer_order", [])) != STATIC_LAYERS
    ):
        raise SourceContractError(f"Neural motion atlas layout drifted for {family}")
    if int(manifest.get("clip_count", -1)) != 104 or int(manifest.get("frame_count", -1)) != 944:
        raise SourceContractError(f"Neural motion counts drifted for {family}")

    artifacts = dict(manifest.get("artifacts", {}))
    layer_artifacts = dict(artifacts.get("layers", {}))
    validated_layers: dict[str, Path] = {}
    for layer in STATIC_LAYERS:
        path = _validate_artifact(
            source_root,
            dict(layer_artifacts.get(layer, {})),
            label=f"neural motion {family}/{layer}",
        )
        with Image.open(path) as image:
            if image.size != NEURAL_MOTION_ATLAS_SIZE or image.mode != "RGBA":
                raise SourceContractError(
                    f"Neural motion atlas image contract drifted for {family}/{layer}: "
                    f"{image.size} {image.mode}"
                )
            image.verify()
        validated_layers[layer] = path
    validated_support: dict[str, Path] = {}
    for key in ("palette", "binding_manifest", "motion_manifests", "frame_index"):
        validated_support[key] = _validate_artifact(
            source_root,
            dict(artifacts.get(key, {})),
            label=f"neural motion {family}/{key}",
        )

    clips = [dict(value) for value in manifest.get("clips", []) if isinstance(value, dict)]
    if len(clips) != 104:
        raise SourceContractError(f"Neural motion clip registry count drifted for {family}")
    expected_keys = {
        (motion, facing) for motion in MOTION_NAMES for facing in FACING_NAMES
    }
    expected_sequence = [
        (motion, facing) for motion in MOTION_NAMES for facing in FACING_NAMES
    ]
    observed_keys: set[tuple[str, str]] = set()
    observed_sequence: list[tuple[str, str]] = []
    occupied_cells: set[int] = set()
    next_start_cell = 0
    for clip in clips:
        key = (str(clip.get("motion", "")), str(clip.get("facing", "")))
        if key in observed_keys:
            raise SourceContractError(f"Duplicate neural motion clip for {family}/{key}")
        observed_keys.add(key)
        observed_sequence.append(key)
        start_cell = int(clip.get("start_cell", -1))
        frame_count = int(clip.get("frame_count", 0))
        expected_frame_count = DEFAULT_FRAME_COUNTS.get(key[0], -1)
        if (
            start_cell != next_start_cell
            or frame_count != expected_frame_count
            or clip.get("loop") is not (key[0] in LOOPING_MOTIONS)
            or start_cell + frame_count > 944
        ):
            raise SourceContractError(f"Invalid neural motion clip region for {family}/{key}")
        next_start_cell += frame_count
        cells = set(range(start_cell, start_cell + frame_count))
        if occupied_cells & cells:
            raise SourceContractError(f"Overlapping neural motion clip region for {family}/{key}")
        occupied_cells.update(cells)
        _require_sha(clip.get("source_clip_sha256", ""), label=f"source clip {family}/{key}")
        _require_sha(clip.get("derived_clip_sha256", ""), label=f"derived clip {family}/{key}")
        clip_gates = dict(clip.get("gates", {}))
        if not clip_gates or any(value is not True for value in clip_gates.values()):
            raise SourceContractError(f"Neural motion clip gate failed for {family}/{key}")
    if (
        observed_keys != expected_keys
        or observed_sequence != expected_sequence
        or occupied_cells != set(range(944))
        or next_start_cell != 944
    ):
        raise SourceContractError(f"Neural motion matrix/cell coverage drifted for {family}")
    return {
        "family": family,
        "sample_id": sample_id,
        "condition": condition,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "layer_paths": validated_layers,
        "support_paths": validated_support,
        "clips": clips,
    }


def _try_sync_neural_motion(
    source_root: Path,
    destination: Path,
    inventory: list[dict[str, Any]],
    static_identities: list[dict[str, Any]],
) -> dict[str, Any]:
    bank_path = source_root / "motion_style_neural_manifest.json"
    verification_path = source_root / "verification_report.json"
    missing = [
        path.name for path in (bank_path, verification_path) if not path.is_file()
    ]
    if missing:
        return _staged_motion_contract(
            "staged",
            ["Awaiting authoritative neural-motion artifacts: " + ", ".join(missing)],
        )
    try:
        bank = _load_json(bank_path)
        verification = _load_json(verification_path)
        identity_records = _validate_neural_motion_bank(
            bank_path,
            bank,
            static_identities,
        )
        _validate_neural_motion_replay(
            verification_path,
            verification,
            bank_path=bank_path,
            bank=bank,
        )
        bank_sha = _sha256_file(bank_path)
        static_by_id = {
            str(identity.get("sample_id", "")): identity for identity in static_identities
        }
        validated: list[dict[str, Any]] = []
        for (expected_family, expected_sample_id, expected_cell), entry in zip(
            EXPECTED_NEURAL_MOTION_REPRESENTATIVES,
            identity_records,
            strict=True,
        ):
            if str(entry.get("family", "")) != expected_family:
                raise SourceContractError("Neural motion family registry order drifted")
            manifest_record = _identity_manifest_record(entry)
            identity_path = _validate_artifact(
                source_root,
                manifest_record,
                label=f"neural motion identity manifest {expected_family}",
            )
            identity_manifest = _load_json(identity_path)
            validated.append(
                _validate_neural_motion_identity(
                    source_root,
                    identity_path,
                    identity_manifest,
                    expected_family=expected_family,
                    expected_sample_id=expected_sample_id,
                    expected_compiler=dict(bank["compiler"]),
                )
            )
            static_identity = static_by_id.get(expected_sample_id)
            if (
                static_identity is None
                or int(static_identity.get("cell", -1)) != expected_cell
                or int(static_identity.get("ordinal", -1)) != expected_cell
                or static_identity.get("family") != expected_family
                or identity_manifest["source"]["raw_fields_sha256"]
                != static_identity.get("raw_fields_sha256")
                or identity_manifest["source"]["compiled_fields_sha256"]
                != static_identity.get("compiled_fields_sha256")
                or identity_manifest["source"]["static_palette_sha256"]
                != static_identity.get("palette_sha256")
            ):
                raise SourceContractError(
                    f"Neural motion representative is not the exact static identity {expected_sample_id}"
                )
            if (
                entry.get("binding_sha256") != identity_manifest["source"]["binding_sha256"]
                or entry.get("raw_fields_sha256")
                != identity_manifest["source"]["raw_fields_sha256"]
                or entry.get("static_palette_sha256")
                != identity_manifest["source"]["static_palette_sha256"]
                or int(entry.get("clip_count", -1)) != 104
                or int(entry.get("frame_count", -1)) != 944
            ):
                raise SourceContractError(
                    f"Neural motion bank summary disagrees with identity manifest for {expected_family}"
                )
        if int(bank.get("identity_count", -1)) != NEURAL_MOTION_IDENTITY_COUNT:
            raise SourceContractError("Neural motion bank identity count drifted")
        if int(bank.get("clip_count", -1)) != NEURAL_MOTION_CLIP_COUNT:
            raise SourceContractError("Neural motion bank clip count drifted")
        if int(bank.get("frame_count", -1)) != NEURAL_MOTION_FRAME_COUNT:
            raise SourceContractError("Neural motion bank frame count drifted")

        version_root = destination / "motion" / _version_token(bank_sha)
        bank_record = _copy_recorded(
            bank_path,
            version_root / "source_manifest.json",
            destination,
            inventory,
        )
        verification_record = _copy_recorded(
            verification_path,
            version_root / "verification_report.json",
            destination,
            inventory,
        )
        output_identities: list[dict[str, Any]] = []
        representative_cells = {
            family: cell
            for family, _sample_id, cell in EXPECTED_NEURAL_MOTION_REPRESENTATIVES
        }
        for identity in validated:
            family = identity["family"]
            identity_record = _copy_recorded(
                identity["manifest_path"],
                version_root / family / "identity_manifest.json",
                destination,
                inventory,
            )
            output_layers: dict[str, dict[str, Any]] = {}
            for layer in STATIC_LAYERS:
                output_layers[layer] = _copy_recorded(
                    identity["layer_paths"][layer],
                    version_root / family / f"{layer}.png",
                    destination,
                    inventory,
                )
            output_identities.append(
                {
                    "family": family,
                    "sample_id": identity["sample_id"],
                    "representative_static_cell": representative_cells[family],
                    "representative_static_sample_id": identity["sample_id"],
                    "representative_semantics": (
                        "one exact animated representative of the 16 static identities in this family; "
                        "the 70 bindable census does not imply that all 80 identities are animated"
                    ),
                    "condition": identity["condition"],
                    "source_identity_manifest_audit_copy": identity_record,
                    "source_identity_manifest_semantics": SOURCE_IDENTITY_MANIFEST_SEMANTICS,
                    "layers": output_layers,
                    "layout": dict(identity["manifest"]["layout"]),
                    "clips": identity["clips"],
                    "source": dict(identity["manifest"]["source"]),
                }
            )
        return {
            "status": "ready",
            "available": True,
            "neural_output": True,
            "fail_closed": True,
            "source_manifest": bank_record,
            "verification_report": verification_record,
            "source_census": dict(bank["source_census"]),
            "gates": dict(bank["gates"]),
            "representative_policy": (
                "first-bank-ordered-full-matrix-valid-identity-per-family-v1"
            ),
            "layers": list(STATIC_LAYERS),
            "motions": list(MOTION_NAMES),
            "facings": list(FACING_NAMES),
            "identity_count": len(output_identities),
            "identities": output_identities,
            "clip_count": sum(len(identity["clips"]) for identity in output_identities),
            "frame_count": sum(
                int(identity["layout"]["frame_count"])
                for identity in output_identities
            ),
        }
    except (KeyError, TypeError, ValueError, OSError, SourceContractError) as error:
        return _staged_motion_contract(
            "rejected",
            [f"Neural-motion source rejected fail-closed: {error}"],
        )


def _preservation_contract() -> dict[str, Any]:
    game_root = PROJECT_ROOT / "game"
    files: list[dict[str, Any]] = []
    for relative, expected_sha in PRESERVED_GAME_FILES.items():
        path = game_root / PurePosixPath(relative)
        if not path.is_file():
            raise SourceContractError(f"Preserved Arena file is missing: {relative}")
        observed_sha = _sha256_file(path)
        if observed_sha != expected_sha:
            raise SourceContractError(
                f"Preserved Arena file changed before workshop sync: {relative} "
                f"{observed_sha} != {expected_sha}"
            )
        files.append({"path": f"res://{relative}", "sha256": observed_sha})
    project_text = (game_root / "project.godot").read_text(encoding="utf-8")
    if 'run/main_scene="res://Arena.tscn"' not in project_text:
        raise SourceContractError("Godot project main scene is no longer Arena.tscn")
    return {
        "main_scene": "res://Arena.tscn",
        "files": files,
        "baseline_preserved": True,
    }


def _bundle_id(static: dict[str, Any], maps: dict[str, Any], motion: dict[str, Any]) -> str:
    motion_source = motion.get("source_manifest", {})
    payload = {
        "contract": INDEX_FORMAT,
        "static_source_sha256": static["source_manifest"]["sha256"],
        "map_source_sha256": maps["source_index"]["sha256"],
        "motion_status": motion["status"],
        "motion_source_sha256": (
            motion_source.get("sha256", "staged")
            if isinstance(motion_source, dict)
            else "staged"
        ),
        "workshop_sync_source_sha256": _sha256_file(Path(__file__)),
    }
    return _sha256_bytes(_canonical_json(payload))


def sync_neural_workshop_assets(
    *,
    destination: Path = DEFAULT_DESTINATION,
    static_source: Path = DEFAULT_STATIC_SOURCE,
    map_index: Path = DEFAULT_MAP_INDEX,
    topology_source: Path = DEFAULT_TOPOLOGY_SOURCE,
    neural_motion_source: Path = DEFAULT_NEURAL_MOTION_SOURCE,
    require_neural_motion_ready: bool = False,
) -> SyncResult:
    destination = Path(destination)
    disk_guard = _guard_disk(destination)
    preservation = _preservation_contract()
    inventory: list[dict[str, Any]] = []
    static = _sync_static_bank(Path(static_source), destination, inventory)
    maps = _sync_map_bank(Path(map_index), Path(topology_source), destination, inventory)
    motion = _try_sync_neural_motion(
        Path(neural_motion_source),
        destination,
        inventory,
        list(static["identities"]),
    )
    if require_neural_motion_ready and motion.get("status") != "ready":
        reasons = "; ".join(str(value) for value in motion.get("reasons", []))
        raise SourceContractError(
            "Neural motion was required but the source did not pass the ready contract: "
            + reasons
        )
    inventory.sort(key=lambda record: str(record["path"]))
    paths = [str(record["path"]) for record in inventory]
    if len(paths) != len(set(paths)):
        raise AssertionError("NeuralWorkshop runtime inventory contains duplicate paths")
    bundle_id = _bundle_id(static, maps, motion)
    index = {
        "format": INDEX_FORMAT,
        "schema_version": "1.0.0",
        "status": "ready",
        "bundle_id": bundle_id,
        "engine": "Godot 4.3",
        "pixel_filter": "nearest",
        "native_scale_options": [1, 4],
        "python_runtime_required": False,
        "runtime_asset_extensions": list(RUNTIME_EXTENSIONS),
        "coordinate_system": (
            "sprite atlases cell-major left-to-right/top-to-bottom; "
            "maps top-left origin; source semantic arrays [y,x]"
        ),
        "generator": {
            "id": "nullvector-native-neural-workshop-sync",
            "version": "1.0.0",
            "source_sha256": _sha256_file(Path(__file__)),
            "deterministic": True,
        },
        "disk_budget": disk_guard,
        "preservation": preservation,
        "capabilities": {
            "neural_identity_browser": True,
            "seven_layer_static_preview": True,
            "neural_motion_preview": motion["status"] == "ready",
            "topology_v2_map_browser": True,
            "runtime_python": False,
        },
        "static": static,
        "motion": motion,
        "maps": maps,
        "inventory": inventory,
        "asset_count": len(inventory),
        "errors": [],
    }
    index_path = destination / "asset_index.json"
    _write_json(index_path, index)
    errors = validate_synced_assets(index_path)
    if errors:
        raise AssertionError("Generated NeuralWorkshop bundle failed validation: " + "; ".join(errors))
    return SyncResult(destination=destination, index_path=index_path, index=index)


def _validate_inventory(index: dict[str, Any], root: Path, errors: list[str]) -> None:
    inventory = list(index.get("inventory", []))
    if int(index.get("asset_count", -1)) != len(inventory):
        errors.append("asset count")
    observed_paths: set[str] = set()
    inventory_by_path: dict[str, dict[str, Any]] = {}
    for ordinal, value in enumerate(inventory):
        if not isinstance(value, dict):
            errors.append(f"inventory record {ordinal}")
            continue
        try:
            relative = _safe_relative(value.get("path", ""), label="runtime inventory")
        except SourceContractError:
            errors.append(f"inventory path {ordinal}")
            continue
        if relative in observed_paths:
            errors.append(f"duplicate inventory {relative}")
            continue
        observed_paths.add(relative)
        inventory_by_path[relative] = dict(value)
        path = root / PurePosixPath(relative)
        if path.suffix not in RUNTIME_EXTENSIONS:
            errors.append(f"runtime extension {relative}")
        if not path.is_file():
            errors.append(f"missing inventory {relative}")
            continue
        if path.stat().st_size != int(value.get("bytes", -1)):
            errors.append(f"bytes inventory {relative}")
        if _sha256_file(path) != str(value.get("sha256", "")):
            errors.append(f"hash inventory {relative}")
    referenced = _runtime_artifact_records(index)
    referenced_paths: set[str] = set()
    for ordinal, record in enumerate(referenced):
        try:
            relative = _safe_relative(record.get("path", ""), label="referenced runtime artifact")
        except SourceContractError:
            errors.append(f"referenced artifact path {ordinal}")
            continue
        if relative in referenced_paths:
            errors.append(f"duplicate referenced artifact {relative}")
            continue
        referenced_paths.add(relative)
        inventory_record = inventory_by_path.get(relative)
        if inventory_record is None:
            errors.append(f"untracked referenced artifact {relative}")
            continue
        if (
            int(record.get("bytes", -1)) != int(inventory_record.get("bytes", -2))
            or str(record.get("sha256", "")) != str(inventory_record.get("sha256", "missing"))
        ):
            errors.append(f"referenced artifact record mismatch {relative}")
    if referenced_paths != observed_paths:
        extras = sorted(observed_paths - referenced_paths)
        missing = sorted(referenced_paths - observed_paths)
        errors.append(f"inventory/reference closure extras={extras} missing={missing}")


def _runtime_artifact_records(index: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    static = dict(index.get("static", {}))
    if isinstance(static.get("source_manifest"), dict):
        records.append(dict(static["source_manifest"]))
    records.extend(dict(value) for value in static.get("atlases", []) if isinstance(value, dict))
    maps = dict(index.get("maps", {}))
    if isinstance(maps.get("source_index"), dict):
        records.append(dict(maps["source_index"]))
    for value in maps.get("maps", []):
        if not isinstance(value, dict):
            continue
        for key in ("atlas", "art_manifest", "topology_manifest"):
            if isinstance(value.get(key), dict):
                records.append(dict(value[key]))
    motion = dict(index.get("motion", {}))
    if motion.get("status") == "ready":
        for key in ("source_manifest", "verification_report"):
            if isinstance(motion.get(key), dict):
                records.append(dict(motion[key]))
        for value in motion.get("identities", []):
            if not isinstance(value, dict):
                continue
            if isinstance(value.get("source_identity_manifest_audit_copy"), dict):
                records.append(dict(value["source_identity_manifest_audit_copy"]))
            layers = value.get("layers", {})
            if isinstance(layers, dict):
                records.extend(
                    dict(layers[layer])
                    for layer in STATIC_LAYERS
                    if isinstance(layers.get(layer), dict)
                )
    return records


def _validate_preservation(index: dict[str, Any], errors: list[str]) -> None:
    preservation = dict(index.get("preservation", {}))
    if preservation.get("main_scene") != "res://Arena.tscn":
        errors.append("preserved main scene")
    if preservation.get("baseline_preserved") is not True:
        errors.append("preservation gate")
    records = list(preservation.get("files", []))
    expected_paths = {f"res://{path}" for path in PRESERVED_GAME_FILES}
    if {str(record.get("path", "")) for record in records} != expected_paths:
        errors.append("preservation inventory")
    for record in records:
        res_path = str(record.get("path", ""))
        relative = res_path.removeprefix("res://")
        path = PROJECT_ROOT / "game" / PurePosixPath(relative)
        expected = PRESERVED_GAME_FILES.get(relative)
        if expected is None or not path.is_file() or _sha256_file(path) != expected:
            errors.append(f"preservation hash {relative}")


def _validate_static_runtime(index: dict[str, Any], root: Path, errors: list[str]) -> None:
    static = dict(index.get("static", {}))
    if static.get("status") != "ready" or static.get("source_format") != STATIC_BANK_FORMAT:
        errors.append("static status/format")
    if tuple(static.get("layers", [])) != STATIC_LAYERS:
        errors.append("static layers")
    identities = list(static.get("identities", []))
    if int(static.get("identity_count", -1)) != STATIC_SAMPLE_COUNT or len(identities) != STATIC_SAMPLE_COUNT:
        errors.append("static identity count")
    if len({str(identity.get("sample_id", "")) for identity in identities}) != STATIC_SAMPLE_COUNT:
        errors.append("static identity uniqueness")
    if [int(identity.get("cell", -1)) for identity in identities] != list(range(STATIC_SAMPLE_COUNT)):
        errors.append("static cell order")
    atlases = list(static.get("atlases", []))
    if len(atlases) != len(STATIC_LAYERS):
        errors.append("static atlas count")
    for expected_layer, atlas in zip(STATIC_LAYERS, atlases):
        if atlas.get("layer") != expected_layer:
            errors.append(f"static atlas layer {expected_layer}")
        path = root / PurePosixPath(str(atlas.get("path", "")))
        if not path.is_file():
            errors.append(f"static atlas missing {expected_layer}")
            continue
        try:
            with Image.open(path) as image:
                if image.size != (STATIC_COLUMNS * STATIC_CELL_SIZE, 5 * STATIC_CELL_SIZE):
                    errors.append(f"static atlas size {expected_layer}")
                image.verify()
        except OSError:
            errors.append(f"static atlas decode {expected_layer}")


def _validate_map_runtime(index: dict[str, Any], root: Path, errors: list[str]) -> None:
    maps = dict(index.get("maps", {}))
    if maps.get("status") != "ready" or maps.get("topology_schema_version") != "2.0.0":
        errors.append("map status/topology version")
    if tuple(maps.get("themes", [])) != tuple(MAP_THEMES):
        errors.append("map themes")
    if tuple(maps.get("layers", [])) != tuple(MAP_LAYERS):
        errors.append("map layers")
    entries = list(maps.get("maps", []))
    if len(entries) != len(MAP_THEMES) or int(maps.get("map_count", -1)) != len(MAP_THEMES):
        errors.append("map count")
    for expected_theme, entry in zip(MAP_THEMES, entries):
        if entry.get("theme") != expected_theme:
            errors.append(f"map theme order {expected_theme}")
        if tuple(layer.get("name") for layer in entry.get("layers", [])) != tuple(MAP_LAYERS):
            errors.append(f"map layer matrix {expected_theme}")
        topology = dict(entry.get("topology_contract", {}))
        if topology.get("schema_version") != "2.0.0" or topology.get("all_invariants_passed") is not True:
            errors.append(f"map topology {expected_theme}")
        atlas_record = dict(entry.get("atlas", {}))
        path = root / PurePosixPath(str(atlas_record.get("path", "")))
        if not path.is_file():
            errors.append(f"map atlas missing {expected_theme}")
            continue
        try:
            with Image.open(path) as image:
                if list(image.size) != list(entry.get("atlas_size", [])):
                    errors.append(f"map atlas size {expected_theme}")
                image.verify()
        except OSError:
            errors.append(f"map atlas decode {expected_theme}")


def _validate_motion_runtime(index: dict[str, Any], root: Path, errors: list[str]) -> None:
    motion = dict(index.get("motion", {}))
    status = str(motion.get("status", ""))
    if status not in ("staged", "rejected", "ready"):
        errors.append("motion status")
        return
    if motion.get("fail_closed") is not True:
        errors.append("motion fail-closed gate")
    if tuple(motion.get("layers", [])) != STATIC_LAYERS:
        errors.append("motion layers")
    if tuple(motion.get("motions", [])) != tuple(MOTION_NAMES):
        errors.append("motion names")
    if tuple(motion.get("facings", [])) != tuple(FACING_NAMES):
        errors.append("motion facings")
    if status != "ready":
        if motion.get("available") is not False or motion.get("neural_output") is not False:
            errors.append("motion staged availability")
        if motion.get("identities", []) or int(motion.get("clip_count", -1)) != 0 or int(motion.get("frame_count", -1)) != 0:
            errors.append("motion staged payload")
        expected = dict(motion.get("expected", {}))
        expected_representatives = [
            {"family": family, "sample_id": sample_id, "static_cell": cell}
            for family, sample_id, cell in EXPECTED_NEURAL_MOTION_REPRESENTATIVES
        ]
        if (
            expected.get("bank_format") != NEURAL_MOTION_BANK_FORMAT
            or expected.get("identity_format") != NEURAL_MOTION_IDENTITY_FORMAT
            or expected.get("replay_format") != NEURAL_MOTION_REPLAY_FORMAT
            or int(expected.get("identity_count", -1)) != NEURAL_MOTION_IDENTITY_COUNT
            or int(expected.get("clip_count", -1)) != NEURAL_MOTION_CLIP_COUNT
            or int(expected.get("frame_count", -1)) != NEURAL_MOTION_FRAME_COUNT
            or int(expected.get("source_sample_count", -1)) != STATIC_SAMPLE_COUNT
            or int(expected.get("bindable_count", -1)) != 70
            or int(expected.get("rejected_count", -1)) != 10
            or expected.get("representatives") != expected_representatives
        ):
            errors.append("motion staged expected contract")
        if not motion.get("reasons") or int(index.get("asset_count", -1)) != 27:
            errors.append("motion staged evidence/inventory")
        return
    if motion.get("available") is not True or motion.get("neural_output") is not True:
        errors.append("motion ready authority")
    if int(index.get("asset_count", -1)) != 69:
        errors.append("motion ready inventory count")
    if motion.get("representative_policy") != (
        "first-bank-ordered-full-matrix-valid-identity-per-family-v1"
    ):
        errors.append("motion representative policy")
    if dict(motion.get("gates", {})) != {name: True for name in EXPECTED_BANK_GATES}:
        errors.append("motion bank gates")
    try:
        _validate_neural_motion_census(
            dict(motion.get("source_census", {})),
            list(dict(index.get("static", {})).get("identities", [])),
        )
    except (KeyError, TypeError, ValueError, SourceContractError):
        errors.append("motion source census")
    identities = list(motion.get("identities", []))
    if len(identities) != NEURAL_MOTION_IDENTITY_COUNT:
        errors.append("motion identity count")
    if int(motion.get("clip_count", -1)) != NEURAL_MOTION_CLIP_COUNT:
        errors.append("motion clip count")
    if int(motion.get("frame_count", -1)) != NEURAL_MOTION_FRAME_COUNT:
        errors.append("motion frame count")
    static_by_id = {
        str(identity.get("sample_id", "")): identity
        for identity in dict(index.get("static", {})).get("identities", [])
    }
    for (expected_family, expected_sample_id, expected_cell), identity in zip(
        EXPECTED_NEURAL_MOTION_REPRESENTATIVES,
        identities,
    ):
        if (
            identity.get("family") != expected_family
            or identity.get("sample_id") != expected_sample_id
            or identity.get("representative_static_sample_id") != expected_sample_id
            or int(identity.get("representative_static_cell", -1)) != expected_cell
        ):
            errors.append(f"motion family {expected_family}")
        static_identity = static_by_id.get(expected_sample_id)
        if (
            static_identity is None
            or int(static_identity.get("cell", -1)) != expected_cell
            or static_identity.get("family") != expected_family
        ):
            errors.append(f"motion static representative {expected_family}")
        if identity.get("source_identity_manifest_semantics") != SOURCE_IDENTITY_MANIFEST_SEMANTICS:
            errors.append(f"motion source manifest semantics {expected_family}")
        if not isinstance(identity.get("source_identity_manifest_audit_copy"), dict):
            errors.append(f"motion source manifest audit copy {expected_family}")
        if len(identity.get("clips", [])) != 104:
            errors.append(f"motion clips {expected_family}")
        expected_clip_sequence = [
            (motion_name, facing)
            for motion_name in MOTION_NAMES
            for facing in FACING_NAMES
        ]
        observed_clip_sequence: list[tuple[str, str]] = []
        next_cell = 0
        for clip in identity.get("clips", []):
            key = (str(clip.get("motion", "")), str(clip.get("facing", "")))
            observed_clip_sequence.append(key)
            frame_count = int(clip.get("frame_count", -1))
            if (
                int(clip.get("start_cell", -1)) != next_cell
                or frame_count != DEFAULT_FRAME_COUNTS.get(key[0], -1)
                or clip.get("loop") is not (key[0] in LOOPING_MOTIONS)
            ):
                errors.append(f"motion clip layout {expected_family}/{key}")
            next_cell += max(frame_count, 0)
        if observed_clip_sequence != expected_clip_sequence or next_cell != 944:
            errors.append(f"motion clip matrix {expected_family}")
        layout = dict(identity.get("layout", {}))
        if (
            int(layout.get("columns", -1)) != NEURAL_MOTION_COLUMNS
            or int(layout.get("rows", -1)) != NEURAL_MOTION_ROWS
            or int(layout.get("cell_size", -1)) != STATIC_CELL_SIZE
            or int(layout.get("frame_count", -1)) != 944
        ):
            errors.append(f"motion layout {expected_family}")
        for layer in STATIC_LAYERS:
            record = dict(identity.get("layers", {}).get(layer, {}))
            path = root / PurePosixPath(str(record.get("path", "")))
            if not path.is_file():
                errors.append(f"motion atlas {expected_family}/{layer}")
                continue
            try:
                with Image.open(path) as image:
                    if image.mode != "RGBA" or image.size != NEURAL_MOTION_ATLAS_SIZE:
                        errors.append(f"motion atlas dimensions {expected_family}/{layer}")
                    image.verify()
            except OSError:
                errors.append(f"motion atlas decode {expected_family}/{layer}")


def validate_synced_assets(index_path: Path) -> list[str]:
    index_path = Path(index_path)
    errors: list[str] = []
    if not index_path.is_file():
        return [f"missing index {index_path}"]
    try:
        index = _load_json(index_path)
    except SourceContractError as error:
        return [str(error)]
    root = index_path.parent
    if index.get("format") != INDEX_FORMAT or index.get("schema_version") != "1.0.0":
        errors.append("index format/version")
    if index.get("status") != "ready":
        errors.append("index status")
    if index.get("engine") != "Godot 4.3":
        errors.append("engine")
    if index.get("pixel_filter") != "nearest" or index.get("native_scale_options") != [1, 4]:
        errors.append("pixel display contract")
    if index.get("python_runtime_required") is not False:
        errors.append("python runtime contract")
    if tuple(index.get("runtime_asset_extensions", [])) != RUNTIME_EXTENSIONS:
        errors.append("runtime extensions")
    if index.get("errors", []) != []:
        errors.append("index errors")
    disk = dict(index.get("disk_budget", {}))
    if (
        disk.get("guard_passed") is not True
        or int(disk.get("minimum_free_bytes", -1)) != MIN_FREE_BYTES
        or int(disk.get("planned_bytes", -1)) != PLANNED_BYTES
    ):
        errors.append("disk guard")
    generator = dict(index.get("generator", {}))
    if generator.get("source_sha256") != _sha256_file(Path(__file__)):
        errors.append("generator source hash")
    _validate_inventory(index, root, errors)
    _validate_preservation(index, errors)
    _validate_static_runtime(index, root, errors)
    _validate_map_runtime(index, root, errors)
    _validate_motion_runtime(index, root, errors)
    try:
        expected_bundle = _bundle_id(
            dict(index["static"]), dict(index["maps"]), dict(index["motion"])
        )
        if index.get("bundle_id") != expected_bundle:
            errors.append("bundle id")
    except (KeyError, TypeError):
        errors.append("bundle id inputs")
    return errors


def asset_inventory(index_path: Path) -> list[dict[str, Any]]:
    index = _load_json(Path(index_path))
    return [dict(record) for record in index.get("inventory", [])]


def _runtime_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix in RUNTIME_EXTENSIONS
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def repeat_check(
    *,
    static_source: Path = DEFAULT_STATIC_SOURCE,
    map_index: Path = DEFAULT_MAP_INDEX,
    topology_source: Path = DEFAULT_TOPOLOGY_SOURCE,
    neural_motion_source: Path = DEFAULT_NEURAL_MOTION_SOURCE,
    require_neural_motion_ready: bool = False,
) -> dict[str, Any]:
    work_root = PROJECT_ROOT / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="neural-workshop-replay-", dir=work_root) as temporary:
        temporary_root = Path(temporary)
        first = sync_neural_workshop_assets(
            destination=temporary_root / "a",
            static_source=static_source,
            map_index=map_index,
            topology_source=topology_source,
            neural_motion_source=neural_motion_source,
            require_neural_motion_ready=require_neural_motion_ready,
        )
        second = sync_neural_workshop_assets(
            destination=temporary_root / "b",
            static_source=static_source,
            map_index=map_index,
            topology_source=topology_source,
            neural_motion_source=neural_motion_source,
            require_neural_motion_ready=require_neural_motion_ready,
        )
        first_digest = _runtime_tree_digest(first.destination)
        second_digest = _runtime_tree_digest(second.destination)
        return {
            "passed": first_digest == second_digest,
            "first_tree_sha256": first_digest,
            "second_tree_sha256": second_digest,
            "asset_count": first.index["asset_count"],
            "bundle_id": first.index["bundle_id"],
        }


def _parse_args(arguments: Iterable[str] | None = None) -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--static-source", type=Path, default=DEFAULT_STATIC_SOURCE)
    parser.add_argument("--map-index", type=Path, default=DEFAULT_MAP_INDEX)
    parser.add_argument("--topology-source", type=Path, default=DEFAULT_TOPOLOGY_SOURCE)
    parser.add_argument("--neural-motion-source", type=Path, default=DEFAULT_NEURAL_MOTION_SOURCE)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--repeat-check", action="store_true")
    parser.add_argument(
        "--require-neural-motion-ready",
        action="store_true",
        help="fail closed unless the exact 5-identity/520-clip neural-motion bank is ready",
    )
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    args = _parse_args(arguments)
    result = sync_neural_workshop_assets(
        destination=args.destination,
        static_source=args.static_source,
        map_index=args.map_index,
        topology_source=args.topology_source,
        neural_motion_source=args.neural_motion_source,
        require_neural_motion_ready=args.require_neural_motion_ready,
    )
    errors = validate_synced_assets(result.index_path)
    replay = (
        repeat_check(
            static_source=args.static_source,
            map_index=args.map_index,
            topology_source=args.topology_source,
            neural_motion_source=args.neural_motion_source,
            require_neural_motion_ready=args.require_neural_motion_ready,
        )
        if args.repeat_check
        else {"passed": None}
    )
    usage = shutil.disk_usage(result.destination)
    report = {
        "format": "nullvector-neural-workshop-sync-report-v1",
        "status": "passed" if not errors and replay.get("passed") is not False else "failed",
        "passed": not errors and replay.get("passed") is not False,
        "destination": str(result.destination.resolve()),
        "index": str(result.index_path.resolve()),
        "index_sha256": _sha256_file(result.index_path),
        "bundle_id": result.index["bundle_id"],
        "asset_count": result.index["asset_count"],
        "motion_status": result.index["motion"]["status"],
        "validation_errors": errors,
        "repeat_check": replay,
        "disk": {
            "free_bytes": usage.free,
            "minimum_free_bytes": MIN_FREE_BYTES,
            "floor_preserved": usage.free >= MIN_FREE_BYTES,
        },
        "preservation": result.index["preservation"],
    }
    if args.report is not None:
        _write_json(args.report, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
