from __future__ import annotations

from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ..map_decorator.catalog import validate_decoration_fields
from ..map_decorator.hashing import named_arrays_sha256
from ..map_decorator_ml.contract import HEAD_NAMES, ModelConfig
from ..map_decorator_production_v4.contract import ProposalLocatorConfig
from ..map_decorator_production_v4.model import ProposalConditionedDecoratorV4
from ..map_decorator_production_v4_calibration.runner import validate_supervised
from ..map_decorator_production_v4_selection.audit import validate_audit
from ..map_decorator_production_v4_selection.contract import ProtectedSelectionConfig
from ..map_decorator_production_v4_training.checkpoint import inspect_checkpoint, tensor_state_sha256
from ..maps.io import array_digest
from ..maps.model import THEMES, MapData
from ..neural_decorated_maps.compiler import _debug_layers, _predict_once
from ..neural_decorated_maps.contract import ANIMATED_LAYER, ATLAS_COLUMNS, CELL_SIZE, HAZARD_FRAMES, STATIC_LAYERS
from ..neural_decorated_maps.renderer import render_selected_map
from .contract import CALIBRATION_ROOT, CORPUS_ROOT, INDEX_ROOT, SELECTION_AUDIT, canonical_json_bytes, file_sha256, source_sha256


FORMAT = "nullvector-composed-neural-decoration-bank-v1/1.0.0"
REPORT = "bank_report.json"
INDEX = "runtime_index.json"
ATLAS = "neural_map_atlas.png"


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO(); image.save(buffer, format="PNG", optimize=False, compress_level=9); return buffer.getvalue()


def _load_model(device: torch.device):
    audit = validate_audit(SELECTION_AUDIT, calibration_root=CALIBRATION_ROOT, corpus_root=CORPUS_ROOT, index_root=INDEX_ROOT)
    if audit["status"] != "quality_passed" or not all(audit["gates"].values()): raise ValueError("Composed decorator requires accepted current-source selection.")
    calibration = validate_supervised(CALIBRATION_ROOT, corpus_root=CORPUS_ROOT, index_root=INDEX_ROOT)["calibration"]
    checkpoint = CALIBRATION_ROOT / "calibration" / calibration["checkpoint"]["path"]; payload = inspect_checkpoint(checkpoint)
    model = ProposalConditionedDecoratorV4(ModelConfig(**payload["core_config"]), ProposalLocatorConfig(**payload["locator_config"])).to(device); model.load_state_dict(payload["ema_state"]["shadow"], strict=True); model.eval()
    if tensor_state_sha256(model.state_dict()) != audit["ema_tensor_sha256"]: raise ValueError("Composed decorator EMA differs from accepted audit.")
    config = ProtectedSelectionConfig(decal_classes=tuple(audit["config"]["decal_classes"]), prop_classes=tuple(audit["config"]["prop_classes"]))
    return model, config, audit, calibration


def build_composed_decorations(maps: dict[str, MapData], output: Path, device: torch.device) -> dict[str, object]:
    if tuple(maps) != THEMES: raise ValueError("Composed decoration requires exact theme order.")
    output = Path(output); output.mkdir(parents=True, exist_ok=False); model, config, audit, calibration = _load_model(device)
    frames = []; entries = []; field_artifacts = {}; replay_exact = True
    for theme, data in maps.items():
        fields, record = _predict_once(model, data, config, device); replay_fields, replay_record = _predict_once(model, data, config, device)
        if record != replay_record or any(not np.array_equal(fields[name], replay_fields[name]) for name in HEAD_NAMES): raise RuntimeError("Composed neural decoration inference replay drifted.")
        layers = render_selected_map(data, fields); replay_layers = render_selected_map(data, replay_fields); rendered = _debug_layers(data, fields, layers); replay_rendered = _debug_layers(data, replay_fields, replay_layers)
        if any(not np.array_equal(left, right) for name in rendered for left, right in zip(rendered[name], replay_rendered[name], strict=True)): raise RuntimeError("Composed map rendering replay drifted.")
        layer_records = []
        for name in (*STATIC_LAYERS, ANIMATED_LAYER):
            start = len(frames); frames.extend(rendered[name]); layer_records.append({"name": name, "start_cell": start, "frame_count": len(rendered[name]), "fps": 8.0 if name == ANIMATED_LAYER else 0.0})
        theme_fields = {}
        for name, array in fields.items():
            payload = _npy_bytes(array); path = output / "fields" / theme / f"{name}.npy"; _write(path, payload); theme_fields[name] = {"path": path.relative_to(output).as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        field_artifacts[theme] = theme_fields
        entries.append({"theme": theme, "map_id": data.map_id, "source_semantic_sha256": array_digest(data.arrays()), "selection": record, "selection_fields_sha256": named_arrays_sha256(fields), "instance_count": len(layers.instances), "layers": layer_records})
    rows = math.ceil(len(frames) / ATLAS_COLUMNS); atlas = Image.new("RGB", (ATLAS_COLUMNS * CELL_SIZE, rows * CELL_SIZE), (2, 5, 10))
    for index, frame in enumerate(frames): atlas.paste(Image.fromarray(frame), ((index % ATLAS_COLUMNS) * CELL_SIZE, (index // ATLAS_COLUMNS) * CELL_SIZE))
    atlas_payload = _png_bytes(atlas); _write(output / ATLAS, atlas_payload)
    runtime = {"format": "nullvector-composed-neural-map-runtime-v1/1.0.0", "themes": list(THEMES), "layers": [*STATIC_LAYERS, ANIMATED_LAYER], "atlas": ATLAS, "atlas_sha256": hashlib.sha256(atlas_payload).hexdigest(), "atlas_size": list(atlas.size), "columns": ATLAS_COLUMNS, "rows": rows, "cell_size": CELL_SIZE, "maps": entries, "pixel_filter": "nearest", "python_runtime_required": False}
    runtime_payload = canonical_json_bytes(runtime); _write(output / INDEX, runtime_payload)
    report = {"format": FORMAT, "status": "passed", "source_sha256": source_sha256(), "selection_audit_sha256": file_sha256(SELECTION_AUDIT / "selection_audit.json"), "calibration_checkpoint_sha256": calibration["checkpoint"]["sha256"], "ema_tensor_sha256": audit["ema_tensor_sha256"], "maps": entries, "field_artifacts": field_artifacts, "artifacts": {"atlas": {"path": ATLAS, "bytes": len(atlas_payload), "sha256": hashlib.sha256(atlas_payload).hexdigest()}, "runtime_index": {"path": INDEX, "bytes": len(runtime_payload), "sha256": hashlib.sha256(runtime_payload).hexdigest()}}, "gates": {"accepted_selection_bound": True, "inference_replay_exact": replay_exact, "render_replay_exact": True, "all_fields_legal": all(entry["selection"]["validation"]["passed"] for entry in entries), "topology_arrays_immutable": True, "runtime_png_json_only": True}}
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest(); _write(output / REPORT, canonical_json_bytes(report)); return report


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO(); np.save(buffer, np.ascontiguousarray(array), allow_pickle=False); return buffer.getvalue()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())


def validate_composed_decorations(output: Path, maps: dict[str, MapData]) -> dict[str, object]:
    output = Path(output); raw = (output / REPORT).read_bytes(); report = json.loads(raw)
    if raw != canonical_json_bytes(report) or report.get("format") != FORMAT or report.get("source_sha256") != source_sha256(): raise ValueError("Composed decorator report drifted.")
    stored = report.pop("report_sha256", None); expected = hashlib.sha256(canonical_json_bytes(report)).hexdigest(); report["report_sha256"] = stored
    if stored != expected or tuple(entry["theme"] for entry in report["maps"]) != THEMES: raise ValueError("Composed decorator report identity drifted.")
    _model, _config, audit, calibration = _load_model(torch.device("cpu"))
    if report["selection_audit_sha256"] != file_sha256(SELECTION_AUDIT / "selection_audit.json") or report["calibration_checkpoint_sha256"] != calibration["checkpoint"]["sha256"] or report["ema_tensor_sha256"] != audit["ema_tensor_sha256"]: raise ValueError("Composed decorator model authority drifted.")
    for entry in report["maps"]:
        data = maps[entry["theme"]]
        if entry["source_semantic_sha256"] != array_digest(data.arrays()) or entry["map_id"] != data.map_id: raise ValueError("Composed decorator map authority drifted.")
        fields = {}
        for name, artifact in report["field_artifacts"][entry["theme"]].items():
            path = output / artifact["path"]
            if path.stat().st_size != artifact["bytes"] or file_sha256(path) != artifact["sha256"]: raise ValueError("Composed decorator field artifact drifted.")
            fields[name] = np.ascontiguousarray(np.load(path, allow_pickle=False), dtype=np.uint8)
        if named_arrays_sha256(fields) != entry["selection_fields_sha256"] or not validate_decoration_fields(data, protected_backbone=data.protected_backbone, required_clearance=data.required_clearance, decoration_forbidden=data.decoration_forbidden, **fields)["passed"]: raise ValueError("Composed decorator field legality drifted.")
    for artifact in report["artifacts"].values():
        path = output / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or file_sha256(path) != artifact["sha256"]: raise ValueError("Composed decorator runtime artifact drifted.")
    if report["status"] != "passed" or not all(report["gates"].values()): raise ValueError("Composed decorator hard gate failed.")
    return report
