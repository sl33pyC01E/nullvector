from __future__ import annotations

from collections import deque
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from jsonschema import Draft202012Validator

from ..config import PROJECT_ROOT
from ..map_topology_neural.codec import build_codec
from ..map_topology_neural.compiler import compile_topology, make_raw_topology
from ..map_topology_neural_prior_generation.contract import CODEC_CHECKPOINT_RELATIVE
from ..map_topology_neural_prior_generation.render import case_preview_png_bytes, contact_sheet_png_bytes
from ..map_topology_neural_prior_training.contract import FROZEN_LATENT_CORPUS_RELATIVE
from ..map_topology_neural_prior_training.dataset import PriorTrainingDataset
from ..map_topology_neural_prior_v2.model import build_prior_v2, sample_parallel_v2
from ..map_topology_neural_prior_v2_training.contract import PriorV2CalibrationConfig
from ..map_topology_neural_prior_v3.training import load_checkpoint as load_prior_checkpoint
from ..map_topology_neural_production.checkpoint import load_checkpoint as load_codec_checkpoint
from ..map_topology_neural_production.contract import TopologyCodecCalibrationConfig
from ..map_topology_neural_production.dataset import TopologyProductionDataset
from ..maps.io import array_digest
from ..maps.model import THEMES, WALKABLE_TERRAIN
from ..safety import require_disk_floor
from .contract import CALIBRATION_ROOT, CORPUS_ROOT, DEFAULT_OUTPUT, FORMAT, INDEX_ROOT, PRIOR_CHECKPOINT, SELECTION_AUDIT, canonical_json_bytes, file_sha256, source_sha256
from .decorator import REPORT as DECORATOR_REPORT, build_composed_decorations, validate_composed_decorations
from .map_pack import load_neural_map_pack, validate_neural_map_pack, write_neural_map_pack


MANIFEST = "synthesis_manifest.json"
RAW_ROOT = "raw_topology"
MAP_ROOT = "topology_maps"
DECORATED_ROOT = "decorated_bank"
CONTACT_SHEET = "topology_contact_sheet.png"
DECORATED_CONTACT_SHEET = "decorated_world_contact_sheet.png"
MAX_REPAIR_FRACTION = 0.15


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO(); np.save(buffer, np.ascontiguousarray(array), allow_pickle=False); return buffer.getvalue()


def _decorated_contact_sheet(bank: Path) -> bytes:
    runtime = json.loads((bank / "runtime_index.json").read_text(encoding="utf-8"))
    atlas = Image.open(bank / runtime["atlas"]).convert("RGB"); cell = int(runtime["cell_size"]); columns = int(runtime["columns"])
    entries = []
    for record in runtime["maps"]:
        layer = next(item for item in record["layers"] if item["name"] == "composite")
        index = int(layer["start_cell"]); x = index % columns * cell; y = index // columns * cell
        entries.append((str(record["theme"]), atlas.crop((x, y, x + cell, y + cell))))
    if tuple(name for name, _image in entries) != THEMES:
        raise RuntimeError("Decorated world contact sheet theme order drifted.")
    label = 26; gutter = 8; sheet = Image.new("RGB", (3 * cell + 4 * gutter, 2 * (cell + label) + 3 * gutter), (2, 5, 11)); draw = ImageDraw.Draw(sheet); font = ImageFont.load_default()
    for index, (name, image) in enumerate(entries):
        column, row = index % 3, index // 3; x = gutter + column * (cell + gutter); y = gutter + row * (cell + label + gutter)
        draw.text((x + 4, y + 6), f"{name.upper()} // NEURAL TOPOLOGY + OBJECTS", fill=(119, 239, 255), font=font); sheet.paste(image, (x, y + label))
    payload = BytesIO(); sheet.save(payload, format="PNG", optimize=False, compress_level=9); return payload.getvalue()


def _reachable(mask: np.ndarray, start: tuple[int, int], targets: tuple[tuple[int, int], ...]) -> bool:
    sx, sy = start
    if not (0 <= sx < mask.shape[1] and 0 <= sy < mask.shape[0] and mask[sy, sx]):
        return False
    seen = {(sx, sy)}; queue = deque(((sx, sy),))
    while queue:
        x, y = queue.popleft()
        for point in ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)):
            px, py = point
            if 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0] and mask[py, px] and point not in seen:
                seen.add(point); queue.append(point)
    return all(point in seen for point in targets)


def _radius_one(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1); result = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            result &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return result


def _decode(codec: torch.nn.Module, tokens: torch.Tensor, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = codec.quantizer.embeddings
    embedded = table.index_select(0, tokens.flatten()).view(1, *tokens.shape[-2:], table.shape[1]).permute(0, 3, 1, 2).contiguous()
    with torch.inference_mode():
        logits = codec.decode(embedded)
    return tuple(np.ascontiguousarray(logits[name].argmax(1)[0, :height, :width].cpu().numpy().astype(dtype)) for name, dtype in (("terrain", np.uint8), ("hazard", np.uint8), ("elevation", np.int8)))  # type: ignore[return-value]


def _configure_cuda(seed: int) -> torch.device:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required before neural world synthesis.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Neural world synthesis requires CUDA BF16.")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    return torch.device("cuda", 0)


def _source_cases(device: torch.device):
    payload = load_prior_checkpoint(PRIOR_CHECKPOINT); config = PriorV2CalibrationConfig.from_dict(payload["config"])
    prior = build_prior_v2(config.model_config()).to(device); prior.load_state_dict(payload["ema_state"], strict=True); prior.eval()
    codec_payload = load_codec_checkpoint(PROJECT_ROOT / CODEC_CHECKPOINT_RELATIVE)
    codec_training = TopologyCodecCalibrationConfig.from_dict(codec_payload["config"])
    codec = build_codec(codec_training.codec_config(), init_seed=codec_training.seed); codec.load_state_dict(codec_payload["ema_state"], strict=True); codec.eval()
    latent = PriorTrainingDataset(CORPUS_ROOT, PROJECT_ROOT / FROZEN_LATENT_CORPUS_RELATIVE)
    source_dataset = TopologyProductionDataset(CORPUS_ROOT)
    refs = source_dataset.evaluation_refs("test", 6)
    if tuple(sorted(ref.theme for ref in refs)) != tuple(sorted(THEMES)):
        raise RuntimeError("Topology synthesis did not select exactly one source condition per theme.")
    latent_by_id = {ref.full_map_identity_sha256: ref for ref in latent.refs_by_split["test"]}
    for source_ref in sorted(refs, key=lambda ref: THEMES.index(ref.theme)):
        source = source_dataset.corpus.read_sample(source_ref.shard_id, source_ref.sample_index, expected_split="test")
        batch = latent.collate((latent_by_id[source_ref.full_map_identity_sha256],))
        conditions = {name: batch[name].to(device) for name in ("valid_mask", "point_conditions", "global_conditions", "theme_index")}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sampled = sample_parallel_v2(prior, conditions, sampling_steps=config.sampling_steps)
        terrain, hazard, elevation = _decode(codec, sampled["tokens"].cpu(), source.config.height, source.config.width)
        raw = make_raw_topology(terrain, hazard, elevation, shape=(source.config.height, source.config.width))
        compiled = compile_topology(raw, seed=config.seed, theme=source.theme, config=source.config, start=source.start, exit=source.exit, objectives=source.objectives, spawns=source.spawns)
        yield source_ref, source, sampled, raw, compiled


def build_world_bank(output: Path = DEFAULT_OUTPUT, *, visually_inspected: bool = False) -> dict[str, object]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Neural world synthesis output is immutable.")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1024 ** 3)
    prior_payload = load_prior_checkpoint(PRIOR_CHECKPOINT); config = PriorV2CalibrationConfig.from_dict(prior_payload["config"])
    device = _configure_cuda(config.seed); torch.cuda.reset_peak_memory_stats(device)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"; staging.mkdir(parents=True)
    records: list[dict[str, object]] = []; preview_rows: list[tuple[str, bytes]] = []; compiled_maps = {}
    started = time.perf_counter()
    try:
        for source_ref, source, sampled, raw, compiled in _source_cases(device):
            pack = write_neural_map_pack(compiled.data, staging / MAP_ROOT); compiled_maps[source.theme] = compiled.data
            theme_raw = staging / RAW_ROOT / source.theme; theme_raw.mkdir(parents=True)
            raw_arrays = {"tokens": sampled["tokens"][0].cpu().numpy().astype(np.uint16), **raw.arrays()}
            raw_artifacts = {}
            for name, array in raw_arrays.items():
                payload = _npy_bytes(array); path = theme_raw / f"{name}.npy"; _atomic_bytes(path, payload)
                raw_artifacts[name] = {"path": path.relative_to(staging).as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            ledger_payload = canonical_json_bytes({"theme": source.theme, "ledger": compiled.ledger})
            ledger_path = theme_raw / "repair_ledger.json"; _atomic_bytes(ledger_path, ledger_payload)
            preview = case_preview_png_bytes(source, raw, compiled.data, scale=max(2, min(4, 256 // max(source_ref.height, source_ref.width))))
            preview_path = theme_raw / "topology.png"; _atomic_bytes(preview_path, preview); preview_rows.append((source.theme, preview))
            walkable = np.isin(raw.terrain, tuple(WALKABLE_TERRAIN)); targets = (source.exit, *source.objectives)
            records.append({
                "theme": source.theme, "condition_identity_sha256": source_ref.full_map_identity_sha256,
                "map_id": compiled.data.map_id, "map_manifest": (pack / "manifest.json").relative_to(staging).as_posix(),
                "map_manifest_sha256": file_sha256(pack / "manifest.json"), "raw_sha256": raw.raw_sha256,
                "raw_arrays_sha256": array_digest(raw.arrays()), "compiled_arrays_sha256": compiled.compiled_arrays_sha256,
                "repair_ledger_sha256": compiled.ledger_sha256, "repair_ledger_file_sha256": hashlib.sha256(ledger_payload).hexdigest(),
                "repair_fraction": float(compiled.report["costs"]["repair_fraction"]),
                "raw_required_reachable": _reachable(walkable, source.start, targets),
                "raw_radius_one_required_reachable": _reachable(_radius_one(walkable), source.start, targets),
                "unique_tokens": int(torch.unique(sampled["tokens"]).numel()), "raw_artifacts": raw_artifacts,
                "preview": {"path": preview_path.relative_to(staging).as_posix(), "bytes": len(preview), "sha256": hashlib.sha256(preview).hexdigest()},
            })
        contact = contact_sheet_png_bytes(preview_rows); _atomic_bytes(staging / CONTACT_SHEET, contact)
        decorated = build_composed_decorations(compiled_maps, staging / DECORATED_ROOT, device)
        decorated_contact = _decorated_contact_sheet(staging / DECORATED_ROOT); _atomic_bytes(staging / DECORATED_CONTACT_SHEET, decorated_contact)
        repair = [float(record["repair_fraction"]) for record in records]
        gates = {
            "six_themes_exact": tuple(record["theme"] for record in records) == THEMES,
            "all_compiled_packs_valid": all(validate_neural_map_pack(staging / MAP_ROOT / str(record["map_id"]))["passed"] for record in records),
            "repair_fraction_bounded": max(repair) <= MAX_REPAIR_FRACTION,
            "accepted_neural_decoration_bound": decorated["status"] == "passed" and all(decorated["gates"].values()),
            "raw_and_repair_authorities_preserved": True,
            "runtime_artifacts_png_json_only": decorated["gates"]["runtime_png_json_only"],
        }
        manifest: dict[str, object] = {
            "format": FORMAT, "status": "experimental_ready" if all(gates.values()) else "failed",
            "source_sha256": source_sha256(), "prior_checkpoint_sha256": file_sha256(PRIOR_CHECKPOINT),
            "prior_step": int(prior_payload.get("base_step", 0)) + int(prior_payload.get("semantic_step", 0)),
            "topology_cases": records, "decorator_report": f"{DECORATED_ROOT}/{DECORATOR_REPORT}",
            "decorator_report_sha256": file_sha256(staging / DECORATED_ROOT / DECORATOR_REPORT),
            "aggregate": {"raw_required_reachable_rate": sum(bool(record["raw_required_reachable"]) for record in records) / len(records), "raw_radius_one_required_reachable_rate": sum(bool(record["raw_radius_one_required_reachable"]) for record in records) / len(records), "mean_repair_fraction": sum(repair) / len(repair), "maximum_repair_fraction": max(repair)},
            "contact_sheets": {
                "topology": {"path": CONTACT_SHEET, "bytes": len(contact), "sha256": hashlib.sha256(contact).hexdigest(), "visually_inspected": bool(visually_inspected)},
                "decorated": {"path": DECORATED_CONTACT_SHEET, "bytes": len(decorated_contact), "sha256": hashlib.sha256(decorated_contact).hexdigest(), "visually_inspected": bool(visually_inspected)},
            },
            "runtime": {"elapsed_seconds": time.perf_counter() - started, "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "display_target_fps": 30, "embodied_motion_hz": 30, "causal_world_hz": 15, "world_synthesis_cadence": "region_entry_or_background_only"},
            "gates": gates,
        }
        manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(); _atomic_bytes(staging / MANIFEST, canonical_json_bytes(manifest)); os.replace(staging, output)
        return validate_world_bank(output)
    finally:
        if staging.exists():
            for root, dirs, files in os.walk(staging, topdown=False):
                for name in files: Path(root, name).unlink(missing_ok=True)
                for name in dirs: Path(root, name).rmdir()
            staging.rmdir()


def validate_world_bank(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output = Path(output).resolve(); raw = (output / MANIFEST).read_bytes(); manifest = json.loads(raw)
    schema = json.loads((PROJECT_ROOT / "shared/schema/neural_world_synthesis_v1.schema.json").read_text(encoding="utf-8")); errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda error: list(error.absolute_path))
    if errors:
        raise ValueError("Neural world synthesis schema failed: " + "; ".join(error.message for error in errors[:4]))
    if raw != canonical_json_bytes(manifest) or manifest.get("format") != FORMAT or manifest.get("source_sha256") != source_sha256():
        raise ValueError("Neural world synthesis manifest authority drifted.")
    stored = manifest.pop("manifest_sha256", None); expected = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(); manifest["manifest_sha256"] = stored
    if stored != expected or manifest.get("prior_checkpoint_sha256") != file_sha256(PRIOR_CHECKPOINT):
        raise ValueError("Neural world synthesis self/checkpoint hash drifted.")
    maps = {}
    for record in manifest["topology_cases"]:
        pack = output / MAP_ROOT / record["map_id"]
        if not validate_neural_map_pack(pack)["passed"] or file_sha256(pack / "manifest.json") != record["map_manifest_sha256"]:
            raise ValueError("Composed topology pack validation failed.")
        maps[record["theme"]] = load_neural_map_pack(pack)
        for artifact in record["raw_artifacts"].values():
            path = output / artifact["path"]
            if path.stat().st_size != artifact["bytes"] or file_sha256(path) != artifact["sha256"]:
                raise ValueError("Raw neural topology artifact drifted.")
    decorated = validate_composed_decorations(output / DECORATED_ROOT, maps)
    if file_sha256(output / DECORATED_ROOT / DECORATOR_REPORT) != manifest["decorator_report_sha256"] or decorated["status"] != "passed":
        raise ValueError("Composed neural decoration authority drifted.")
    for contact in manifest["contact_sheets"].values():
        contact_path = output / contact["path"]
        if contact_path.stat().st_size != contact["bytes"] or file_sha256(contact_path) != contact["sha256"]:
            raise ValueError("Neural world synthesis contact sheet drifted.")
    if manifest["status"] != "experimental_ready" or not all(manifest["gates"].values()):
        raise ValueError("Neural world synthesis hard gate failed.")
    return manifest
