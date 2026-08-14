from __future__ import annotations

import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Final
import uuid

import numpy as np
from PIL import Image
import torch

from ..config import PROJECT_ROOT
from ..map_art.provenance import source_hash as map_art_source_sha256
from ..map_decorator.catalog import build_legal_class_masks, catalog_for, validate_decoration_fields
from ..map_decorator.features import encode_features
from ..map_decorator.hashing import named_arrays_sha256, json_sha256
from ..map_decorator_ml.contract import HEAD_NAMES, ModelConfig, global_condition_vector
from ..map_decorator_ml.legality import TorchLegalMasks, apply_legal_mask, legal_masks_to_torch
from ..map_decorator_production.contract import FEATURE_SEED_SALT
from ..map_decorator_production.teacher import semantic_teacher_targets
from ..map_decorator_production_v4.contract import ProposalLocatorConfig
from ..map_decorator_production_v4.model import ProposalConditionedDecoratorV4
from ..map_decorator_production_v4.proposal import build_proposal_fields
from ..map_decorator_production_v4_calibration.runner import validate_supervised
from ..map_decorator_production_v4_selection.audit import selection_source_sha256, validate_audit
from ..map_decorator_production_v4_selection.contract import ProtectedSelectionConfig
from ..map_decorator_production_v4_selection.decoder import select_protected_proposal_argmax
from ..map_decorator_production_v4_training.checkpoint import inspect_checkpoint, tensor_state_sha256
from ..maps.io import array_digest, file_sha256, load_map_pack
from ..maps.model import THEMES, MapData, Terrain
from ..maps.validate import validate_pack
from ..safety import require_disk_floor
from .contract import (
    ANIMATED_LAYER,
    ATLAS_COLUMNS,
    BANK_FORMAT,
    CELL_SIZE,
    HAZARD_FRAMES,
    NEURAL_DECORATED_MAP_CONTRACT_SHA256,
    RUNTIME_FORMAT,
    STATIC_LAYERS,
)
from .renderer import SelectedMapLayers, composite_frame, render_selected_map


REPORT_NAME: Final[str] = "bank_report.json"
INDEX_NAME: Final[str] = "runtime_index.json"
ATLAS_NAME: Final[str] = "neural_map_atlas.png"
FIELDS_NAME: Final[str] = "selection_fields.npz"


def compiler_source_manifest(root: Path = PROJECT_ROOT) -> dict[str, object]:
    package = Path(root) / "forge/neural_decorated_maps"
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in package.glob("*.py") if item.is_file())
    }
    return {
        "selection_source_sha256": selection_source_sha256(root),
        "map_art_source_sha256": map_art_source_sha256(),
        "compiler_files": files,
    }


def compiler_source_sha256(root: Path = PROJECT_ROOT) -> str:
    return json_sha256(compiler_source_manifest(root))


def _mix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (value ^ (value >> 31)) & ((1 << 64) - 1)


def feature_seed(map_seed: int) -> int:
    return _mix64(int(map_seed) ^ FEATURE_SEED_SALT)


def select_source_maps(root: Path) -> dict[str, Path]:
    root = Path(root).resolve()
    selected: dict[str, Path] = {}
    for theme in THEMES:
        candidates = sorted(path.parent for path in root.glob(f"{theme}-*/manifest.json"))
        valid = [path for path in candidates if validate_pack(path)["passed"] and load_map_pack(path).theme == theme]
        if len(valid) != 1:
            raise ValueError(f"Expected exactly one valid topology-v2 source map for {theme}, observed {len(valid)}.")
        selected[theme] = valid[0]
    return selected


def _predict_once(
    model: ProposalConditionedDecoratorV4,
    data: MapData,
    config: ProtectedSelectionConfig,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    encoded = encode_features(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
        public_seed=feature_seed(data.seed),
    )
    base_masks = build_legal_class_masks(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
    )
    legal_arrays = {name: np.ascontiguousarray(getattr(base_masks, name), dtype=bool) for name in HEAD_NAMES}
    proposals = build_proposal_fields(
        map_seed=data.seed,
        theme=data.theme,
        shape=data.shape,
        legal_masks=legal_arrays,
        hard_empty=np.ascontiguousarray(base_masks.hard_empty, dtype=bool),
    )
    features = torch.from_numpy(encoded.tensor.copy())[None].to(device)
    labels = {name: torch.zeros((1, *data.shape), dtype=torch.long, device=device) for name in HEAD_NAMES}
    masked = {name: torch.ones((1, *data.shape), dtype=torch.bool, device=device) for name in HEAD_NAMES}
    theme = torch.tensor([THEMES.index(data.theme)], dtype=torch.long, device=device)
    conditions = torch.from_numpy(global_condition_vector(encoded).copy())[None].to(device)
    proposal_tensors = {
        "decal": torch.from_numpy(proposals.decal.copy())[None].to(device),
        "prop": torch.from_numpy(proposals.prop.copy())[None].to(device),
    }
    base_legal = legal_masks_to_torch(base_masks, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(features, labels, masked, theme, conditions, torch.ones((1,), device=device), proposal_tensors)
        selected, diagnostics = select_protected_proposal_argmax(output, base_legal, config=config)
    neural_fields = {
        name: np.ascontiguousarray(selected[name][0].cpu().numpy(), dtype=np.uint8)
        for name in ("variant", "decal", "prop")
    }
    neural_conditional = build_legal_class_masks(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
        selected_variant=neural_fields["variant"],
        selected_decal=neural_fields["decal"],
        selected_prop=neural_fields["prop"],
    )
    conditional_torch = legal_masks_to_torch(neural_conditional, device=device)
    with torch.inference_mode():
        emission = torch.argmax(apply_legal_mask(output.emission, conditional_torch.emission, "emission"), dim=1)
    neural_fields["emission"] = np.ascontiguousarray(emission[0].cpu().numpy(), dtype=np.uint8)
    teacher, _, _ = semantic_teacher_targets(data)
    fields = {
        "variant": np.ascontiguousarray(teacher["variant"], dtype=np.uint8),
        "decal": neural_fields["decal"],
        "prop": neural_fields["prop"],
    }
    conditional = build_legal_class_masks(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
        selected_variant=fields["variant"],
        selected_decal=fields["decal"],
        selected_prop=fields["prop"],
    )
    semantic_emission = np.zeros(data.shape, dtype=np.uint8)
    capable = conditional.emission[1]
    semantic_emission[capable] = 1
    semantic_emission[(data.terrain == int(Terrain.CRYSTAL)) & capable] = 2
    catalog = catalog_for(data.theme)
    for field, entries in ((fields["decal"], catalog.decal_classes), (fields["prop"], catalog.prop_classes)):
        for entry in entries:
            if entry.emission_capable:
                semantic_emission[(field == entry.class_id) & capable] = 3 if entry.color_role == "secondary" else 2
    fields["emission"] = np.ascontiguousarray(semantic_emission, dtype=np.uint8)
    validation = validate_decoration_fields(
        data,
        protected_backbone=data.protected_backbone,
        required_clearance=data.required_clearance,
        decoration_forbidden=data.decoration_forbidden,
        **fields,
    )
    if not validation["passed"]:
        raise RuntimeError(f"Neural decorated map prediction is illegal: {validation}")
    agreement = {
        name: float((fields[name] == teacher[name]).mean()) for name in HEAD_NAMES
    }
    return fields, {
        "feature_seed": feature_seed(data.seed),
        "feature_tensor_sha256": encoded.tensor_sha256,
        "proposal_fields_sha256": proposals.fields_sha256,
        "neural_raw_fields_sha256": named_arrays_sha256(neural_fields),
        "selection_fields_sha256": named_arrays_sha256(fields),
        "field_authority": {
            "variant": "deterministic_semantic_teacher",
            "decal": "accepted_neural_protected_selector",
            "prop": "accepted_neural_protected_selector",
            "emission": "conditional_semantic_projection",
        },
        "unsupported_neural_heads_cross_runtime_boundary": False,
        "validation": validation,
        "agreement_with_procedural_teacher": agreement,
        "protected_diagnostics": diagnostics,
    }


def _debug_layers(data: MapData, fields: dict[str, np.ndarray], layers: SelectedMapLayers) -> dict[str, list[np.ndarray]]:
    object_rgb = np.full((*data.shape, 3), (3, 8, 15), dtype=np.uint8)
    object_rgb[fields["decal"] == 1] = (38, 231, 255)
    object_rgb[fields["decal"] == 2] = (244, 55, 222)
    object_rgb[fields["prop"] == 1] = (255, 175, 42)
    object_rgb[fields["prop"] == 2] = (92, 255, 117)
    variant_palette = np.asarray([(5, 10, 20), (32, 88, 145), (49, 200, 214), (82, 255, 153), (216, 248, 87), (255, 179, 61), (255, 72, 170), (173, 78, 255)], dtype=np.uint8)
    emission_palette = np.asarray([(3, 8, 15), (23, 89, 120), (38, 220, 255), (247, 246, 255)], dtype=np.uint8)
    topology = np.full((*data.shape, 3), (3, 8, 15), dtype=np.uint8)
    topology[data.decoration_forbidden != 0] = (72, 28, 110)
    topology[data.required_clearance != 0] = (38, 220, 255)
    topology[data.protected_backbone != 0] = (255, 55, 180)
    topology[data.hazard != 0] = (255, 166, 42)

    def up(value: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(value).resize((CELL_SIZE, CELL_SIZE), Image.Resampling.NEAREST))

    return {
        "composite": [composite_frame(layers, 0)],
        "base_color": [layers.base_color],
        "emissive": [layers.emissive],
        "objects": [up(object_rgb)],
        "variant": [up(variant_palette[fields["variant"]])],
        "emission_level": [up(emission_palette[fields["emission"]])],
        "topology": [up(topology)],
        "hazard": [composite_frame(layers, frame) for frame in range(HAZARD_FRAMES)],
    }


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO(); image.save(buffer, format="PNG", optimize=False, compress_level=9); return buffer.getvalue()


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = BytesIO(); np.savez_compressed(buffer, **arrays); return buffer.getvalue()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True); raise


def _configure_cuda(seed: int) -> torch.device:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required before neural map compilation.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Neural map compilation requires CUDA BF16.")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    return torch.device("cuda", 0)


def build_bank(
    selection_audit: Path,
    calibration_root: Path,
    corpus_root: Path,
    index_root: Path,
    map_root: Path,
    output: Path,
    *,
    visually_inspected: bool = False,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Neural decorated map bank is immutable.")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=512 * 1024 * 1024)
    audit = validate_audit(selection_audit, calibration_root=calibration_root, corpus_root=corpus_root, index_root=index_root)
    if audit["status"] != "quality_passed" or not all(audit["gates"].values()):
        raise ValueError("Neural map compilation requires an accepted protected-selection audit.")
    calibration = validate_supervised(calibration_root, corpus_root=corpus_root, index_root=index_root)["calibration"]
    checkpoint_path = Path(calibration_root) / "calibration" / calibration["checkpoint"]["path"]
    payload = inspect_checkpoint(checkpoint_path)
    device = _configure_cuda(payload["training_config"]["seed"])
    model = ProposalConditionedDecoratorV4(ModelConfig(**payload["core_config"]), ProposalLocatorConfig(**payload["locator_config"])).to(device)
    model.load_state_dict(payload["ema_state"]["shadow"], strict=True); model.eval()
    if tensor_state_sha256(model.state_dict()) != audit["ema_tensor_sha256"]:
        raise ValueError("Neural map compiler EMA tensors differ from the accepted selector audit.")
    config = ProtectedSelectionConfig(
        decal_classes=tuple(audit["config"]["decal_classes"]),
        prop_classes=tuple(audit["config"]["prop_classes"]),
    )
    sources = select_source_maps(map_root)
    frames: list[np.ndarray] = []
    entries: list[dict[str, object]] = []
    field_arrays: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for theme in THEMES:
        pack = sources[theme]
        data = load_map_pack(pack)
        first_fields, first_record = _predict_once(model, data, config, device)
        replay_fields, replay_record = _predict_once(model, data, config, device)
        if first_record != replay_record or any(not np.array_equal(first_fields[name], replay_fields[name]) for name in HEAD_NAMES):
            raise RuntimeError(f"Neural map inference replay drifted for {theme}.")
        first_layers = render_selected_map(data, first_fields)
        replay_layers = render_selected_map(data, replay_fields)
        rendered = _debug_layers(data, first_fields, first_layers)
        replay_rendered = _debug_layers(data, replay_fields, replay_layers)
        if any(not np.array_equal(a, b) for name in rendered for a, b in zip(rendered[name], replay_rendered[name], strict=True)):
            raise RuntimeError(f"Neural map rendering replay drifted for {theme}.")
        layer_records = []
        for name in (*STATIC_LAYERS, ANIMATED_LAYER):
            start_cell = len(frames)
            frames.extend(rendered[name])
            layer_records.append({"name": name, "start_cell": start_cell, "frame_count": len(rendered[name]), "fps": 8.0 if name == ANIMATED_LAYER else 0.0})
        for name in HEAD_NAMES:
            field_arrays[f"{theme}__{name}"] = first_fields[name]
        source_manifest = pack / "manifest.json"
        entries.append({
            "theme": theme,
            "map_id": data.map_id,
            "seed": int(data.seed),
            "source_manifest_sha256": file_sha256(source_manifest),
            "source_semantic_sha256": array_digest(data.arrays()),
            "topology_masks_sha256": array_digest({name: data.arrays()[name] for name in ("protected_backbone", "required_clearance", "decoration_forbidden")}),
            "selection": first_record,
            "instance_count": len(first_layers.instances),
            "layers": layer_records,
        })
    rows = math.ceil(len(frames) / ATLAS_COLUMNS)
    atlas = Image.new("RGB", (ATLAS_COLUMNS * CELL_SIZE, rows * CELL_SIZE), (2, 5, 10))
    for index, frame in enumerate(frames):
        atlas.paste(Image.fromarray(frame), ((index % ATLAS_COLUMNS) * CELL_SIZE, (index // ATLAS_COLUMNS) * CELL_SIZE))
    atlas_payload = _png_bytes(atlas)
    fields_payload = _npz_bytes(field_arrays)
    runtime_index: dict[str, object] = {
        "format": RUNTIME_FORMAT,
        "contract_sha256": NEURAL_DECORATED_MAP_CONTRACT_SHA256,
        "themes": list(THEMES),
        "layers": [*STATIC_LAYERS, ANIMATED_LAYER],
        "atlas": ATLAS_NAME,
        "atlas_sha256": hashlib.sha256(atlas_payload).hexdigest(),
        "atlas_size": list(atlas.size),
        "columns": ATLAS_COLUMNS,
        "rows": rows,
        "cell_size": CELL_SIZE,
        "maps": entries,
        "python_runtime_required": False,
        "pixel_filter": "nearest",
    }
    runtime_payload = (json.dumps(runtime_index, indent=2, sort_keys=True) + "\n").encode()
    report: dict[str, object] = {
        "format": BANK_FORMAT,
        "status": "passed",
        "contract_sha256": NEURAL_DECORATED_MAP_CONTRACT_SHA256,
        "compiler_source_sha256": compiler_source_sha256(),
        "selection_audit_sha256": file_sha256(Path(selection_audit) / "selection_audit.json"),
        "calibration_checkpoint_sha256": calibration["checkpoint"]["sha256"],
        "ema_tensor_sha256": calibration["ema_tensor_sha256"],
        "map_source_root": Path(map_root).resolve().name,
        "counts": {"themes": 6, "layers": len(STATIC_LAYERS) + 1, "atlas_frames": len(frames), "hazard_frames_per_theme": HAZARD_FRAMES},
        "maps": entries,
        "artifacts": {
            "runtime_index": {"path": INDEX_NAME, "bytes": len(runtime_payload), "sha256": hashlib.sha256(runtime_payload).hexdigest()},
            "atlas": {"path": ATLAS_NAME, "bytes": len(atlas_payload), "sha256": hashlib.sha256(atlas_payload).hexdigest()},
            "fields": {"path": FIELDS_NAME, "bytes": len(fields_payload), "sha256": hashlib.sha256(fields_payload).hexdigest(), "semantic_sha256": named_arrays_sha256(field_arrays)},
        },
        "runtime": {"device": str(device), "precision": "bf16", "elapsed_seconds": time.perf_counter() - started, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))},
        "visual": {"visually_inspected": bool(visually_inspected)},
        "gates": {
            "accepted_selection_bound": True,
            "six_themes_exact": len(entries) == len(THEMES),
            "inference_replay_exact": True,
            "render_replay_exact": True,
            "all_fields_legal": all(entry["selection"]["validation"]["passed"] for entry in entries),  # type: ignore[index]
            "topology_arrays_immutable": True,
            "checkpoint_not_shipped": True,
            "runtime_png_json_only": True,
        },
    }
    report["semantic_sha256"] = json_sha256({key: value for key, value in report.items() if key != "runtime"})
    report["report_sha256"] = json_sha256(report)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"; staging.mkdir(parents=True, exist_ok=False)
    _atomic_bytes(staging / ATLAS_NAME, atlas_payload); _atomic_bytes(staging / FIELDS_NAME, fields_payload); _atomic_bytes(staging / INDEX_NAME, runtime_payload)
    _atomic_bytes(staging / REPORT_NAME, (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    os.replace(staging, output)
    return validate_bank(output, selection_audit=selection_audit, calibration_root=calibration_root, corpus_root=corpus_root, index_root=index_root, map_root=map_root)


def validate_bank(
    output: Path,
    *,
    selection_audit: Path,
    calibration_root: Path,
    corpus_root: Path,
    index_root: Path,
    map_root: Path,
) -> dict[str, Any]:
    output = Path(output).resolve(); report = json.loads((output / REPORT_NAME).read_text(encoding="utf-8"))
    stored = report.pop("report_sha256", None)
    if stored != json_sha256(report): raise ValueError("Neural decorated map report self-hash failed.")
    report["report_sha256"] = stored
    if report.get("format") != BANK_FORMAT or report.get("contract_sha256") != NEURAL_DECORATED_MAP_CONTRACT_SHA256: raise ValueError("Neural decorated map format/contract failed.")
    if report.get("compiler_source_sha256") != compiler_source_sha256(): raise ValueError("Neural decorated map compiler source drifted.")
    audit = validate_audit(selection_audit, calibration_root=calibration_root, corpus_root=corpus_root, index_root=index_root)
    if report.get("selection_audit_sha256") != file_sha256(Path(selection_audit) / "selection_audit.json") or audit["status"] != "quality_passed": raise ValueError("Neural decorated map selection authority failed.")
    sources = select_source_maps(map_root)
    if tuple(entry["theme"] for entry in report["maps"]) != THEMES: raise ValueError("Neural decorated map theme order failed.")
    for entry in report["maps"]:
        data = load_map_pack(sources[entry["theme"]])
        if entry["source_semantic_sha256"] != array_digest(data.arrays()) or entry["map_id"] != data.map_id: raise ValueError("Neural decorated map source identity failed.")
    for artifact in report["artifacts"].values():
        path = output / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or file_sha256(path) != artifact["sha256"]: raise ValueError("Neural decorated map artifact identity failed.")
    with np.load(output / FIELDS_NAME, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    if named_arrays_sha256(arrays) != report["artifacts"]["fields"]["semantic_sha256"]: raise ValueError("Neural decorated map field archive semantics failed.")
    runtime = json.loads((output / INDEX_NAME).read_text(encoding="utf-8"))
    if runtime.get("format") != RUNTIME_FORMAT or runtime.get("atlas_sha256") != report["artifacts"]["atlas"]["sha256"]: raise ValueError("Neural decorated map runtime index failed.")
    if not all(report.get("gates", {}).values()) or report.get("status") != "passed": raise ValueError("Neural decorated map hard gate failed.")
    semantic = json_sha256({key: value for key, value in report.items() if key not in {"runtime", "semantic_sha256", "report_sha256"}})
    if semantic != report.get("semantic_sha256"): raise ValueError("Neural decorated map semantic hash failed.")
    return report
