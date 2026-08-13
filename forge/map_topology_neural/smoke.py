from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
import uuid

import jsonschema
import numpy as np
from PIL import __version__ as PIL_VERSION
import torch

from ..config import PROJECT_ROOT
from ..maps.model import THEMES, Hazard, Terrain
from ..safety import disk_status, require_disk_floor
from .artifacts import (
    COMPILED_MANIFEST,
    RAW_MANIFEST,
    load_compiled_artifact,
    load_raw_artifact,
    write_compiled_artifact,
    write_raw_artifact,
)
from .checkpoint import load_codec_checkpoint, save_codec_checkpoint
from .codec import CodecConfig, build_codec, collate_topology_tensors, train_cpu_smoke
from .compiler import THEME_HAZARDS, compile_topology, make_raw_topology
from .contract import CONTRACT_SHA256, encode_topology_tensor, inferred_conditions
from .corpus import (
    FROZEN_CORPUS_MANIFEST_FILE_SHA256,
    FROZEN_CORPUS_SHA256,
    TopologyCorpus,
)
from .hashing import file_sha256, json_sha256, named_arrays_sha256
from .provenance import compiler_source_sha256, source_manifest, source_sha256
from .render import contact_sheet_png_bytes


SMOKE_FORMAT = "nullvector-neural-map-topology-smoke-v1"
SMOKE_MANIFEST = "smoke_manifest.json"
SMOKE_REPLAY = "replay_report.json"
CONTACT_SHEET = "topology_repair_contact_sheet.png"
CODEC_METRICS = "codec/metrics.json"
CODEC_CHECKPOINT = "codec/checkpoint.pt"
MAX_SMOKE_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_CONTACT_SHEET_BYTES = 32 * 1024 * 1024
SMOKE_INIT_SEED = 0x5641544F504F
SMOKE_TRAIN_SEED = 0x435055534D4F4B45


def _schema() -> dict[str, object]:
    payload = json.loads(
        (PROJECT_ROOT / "shared" / "schema" / "map_topology_neural_smoke.schema.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(payload, dict):
        raise TypeError("Neural topology smoke schema root must be an object.")
    return payload


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))


def _corrupt_sample(sample: object) -> tuple[object, dict[str, int]]:
    source = getattr(sample, "raw")
    terrain = source.terrain.copy()
    hazard = source.hazard.copy()
    elevation = source.elevation.copy()
    height, width = terrain.shape
    center_x = width // 2
    thin_y = max(3, min(height - 4, height // 3))
    hazard_id = THEME_HAZARDS[getattr(sample, "theme")][0]
    terrain[0, center_x] = int(Terrain.FLOOR)
    terrain[-1, center_x] = int(Terrain.FLOOR)
    terrain[1:-1, center_x] = int(Terrain.WALL)
    hazard[1:-1, center_x] = np.uint8(hazard_id)
    elevation[1:-1, center_x] = np.int8(5)
    terrain[thin_y, 1:-1] = int(Terrain.FLOOR)
    terrain[thin_y - 1, 1:-1] = int(Terrain.WALL)
    terrain[thin_y + 1, 1:-1] = int(Terrain.WALL)
    hazard[thin_y, 2:-2:3] = np.uint8(hazard_id)
    exit_x, exit_y = getattr(sample, "exit")
    terrain[exit_y - 1 : exit_y + 2, exit_x - 1 : exit_x + 2] = int(Terrain.WALL)
    hazard[exit_y - 1 : exit_y + 2, exit_x - 1 : exit_x + 2] = np.uint8(hazard_id)
    elevation[exit_y - 1 : exit_y + 2, exit_x - 1 : exit_x + 2] = np.int8(5)
    raw = make_raw_topology(terrain, hazard, elevation, shape=terrain.shape)
    changed = {
        "terrain": int((source.terrain != terrain).sum()),
        "hazard": int((source.hazard != hazard).sum()),
        "elevation": int((source.elevation != elevation).sum()),
    }
    return raw, changed


def _decode_sha256(model: object, batch: dict[str, torch.Tensor]) -> str:
    getattr(model, "eval")()
    with torch.no_grad():
        output = model(batch, update_ema=False)
    arrays = {
        name: tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        for name, tensor in output["logits"].items()
    }
    arrays["indices"] = output["indices"].detach().cpu().numpy().astype(np.int64, copy=False)
    return named_arrays_sha256(arrays)


def build_smoke(
    output: Path,
    *,
    corpus_root: Path = PROJECT_ROOT / "outputs" / "map_decorator_corpus_v1",
    visually_inspected: bool = False,
) -> dict[str, object]:
    if not isinstance(visually_inspected, bool):
        raise TypeError("visually_inspected must be an explicit boolean attestation.")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Neural topology smoke output is immutable.")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=512 * 1024 * 1024)
    corpus = TopologyCorpus(corpus_root)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    cases: list[dict[str, object]] = []
    render_rows = []
    tensors = []
    try:
        for theme in THEMES:
            shard_id = corpus.find_shard(
                theme=theme, width=32, height=32, objective_count=1, kind="main"
            )
            sample = corpus.read_sample(shard_id, 0)
            corrupt, corruption_counts = _corrupt_sample(sample)
            case_root = staging / "cases" / theme
            raw_artifact = write_raw_artifact(
                case_root / "raw",
                raw=corrupt,
                seed=sample.seed,
                theme=sample.theme,
                config=sample.config,
                start=sample.start,
                exit=sample.exit,
                objectives=sample.objectives,
                spawns=sample.spawns,
                provenance={
                    "corpus_sha256": sample.corpus_sha256,
                    "corpus_manifest_file_sha256": sample.corpus_manifest_file_sha256,
                    "topology_sample_sha256": sample.topology_sample_sha256,
                    "full_map_identity_sha256": sample.full_map_identity_sha256,
                    "sample_identity_sha256": sample.sample_identity_sha256,
                    "shard_id": sample.shard_id,
                    "shard_artifact_sha256": sample.shard_artifact_sha256,
                    "sample_index": 0,
                },
                proposal_source="deterministic_adversarial_smoke_not_model_sample",
            )
            result = compile_topology(
                corrupt,
                seed=sample.seed,
                theme=sample.theme,
                config=sample.config,
                start=sample.start,
                exit=sample.exit,
                objectives=sample.objectives,
                spawns=sample.spawns,
            )
            compiled_artifact = write_compiled_artifact(
                case_root / "compiled", raw_artifact=raw_artifact, result=result
            )
            tensor = encode_topology_tensor(
                terrain=corrupt.terrain,
                hazard=corrupt.hazard,
                elevation=corrupt.elevation,
                theme=sample.theme,
                config=sample.config,
                start=sample.start,
                exit=sample.exit,
                objectives=sample.objectives,
                spawns=sample.spawns,
                conditions=inferred_conditions(corrupt.terrain, corrupt.hazard),
            )
            tensors.append(tensor)
            render_rows.append(
                (
                    theme,
                    (sample.raw.terrain, sample.raw.hazard, sample.raw.elevation),
                    corrupt,
                    compiled_artifact.result.data,
                    sample.start,
                    sample.exit,
                    sample.objectives,
                    sample.spawns,
                )
            )
            cases.append(
                {
                    "theme": theme,
                    "source": {
                        "shard_id": shard_id,
                        "sample_index": 0,
                        "map_id": sample.map_id,
                        "topology_sample_sha256": sample.topology_sample_sha256,
                        "source_raw_topology_sha256": sample.raw.raw_sha256,
                    },
                    "corruption_changed_cells": corruption_counts,
                    "raw": {
                        "path": f"cases/{theme}/raw/{RAW_MANIFEST}",
                        "manifest_sha256": raw_artifact.manifest_sha256,
                        "raw_identity_sha256": raw_artifact.manifest["raw_identity_sha256"],
                        "raw_topology_sha256": corrupt.raw_sha256,
                    },
                    "compiled": {
                        "path": f"cases/{theme}/compiled/{COMPILED_MANIFEST}",
                        "manifest_sha256": compiled_artifact.manifest_sha256,
                        "compiled_identity_sha256": compiled_artifact.manifest["compiled_identity_sha256"],
                        "compiled_arrays_sha256": result.compiled_arrays_sha256,
                        "ledger_sha256": result.ledger_sha256,
                        "ledger_entry_count": len(result.ledger),
                    },
                    "costs": result.report["costs"],
                    "tensor_sha256": tensor.tensor_sha256,
                }
            )
        contact = contact_sheet_png_bytes(render_rows, scale=4)
        if len(contact) > MAX_CONTACT_SHEET_BYTES:
            raise ValueError("Neural topology contact sheet exceeds its strict byte bound.")
        _atomic_bytes(staging / CONTACT_SHEET, contact)

        config = CodecConfig(
            width=8,
            latent_dim=8,
            codebook_size=16,
            field_embedding_dim=2,
            residual_depth=0,
            ema_decay=0.95,
        )
        model = build_codec(config, init_seed=SMOKE_INIT_SEED)
        batch = collate_topology_tensors(tensors)
        state = train_cpu_smoke(
            model,
            batch,
            steps=2,
            training_seed=SMOKE_TRAIN_SEED,
            learning_rate=1.0e-3,
        )
        metrics: dict[str, object] = {
            "format": "nullvector-neural-topology-codec-cpu-smoke-metrics-v1",
            "authority": state["authority"],
            "device": state["device"],
            "steps": state["steps"],
            "history": state["history"],
            "batch_size": len(tensors),
            "theme_order": list(THEMES),
            "config": config.to_dict(),
            "model_init_seed": SMOKE_INIT_SEED,
            "training_seed": SMOKE_TRAIN_SEED,
        }
        metrics_path = staging / CODEC_METRICS
        _atomic_json(metrics_path, metrics)
        checkpoint_path = staging / CODEC_CHECKPOINT
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar = save_codec_checkpoint(
            checkpoint_path,
            model=model,
            model_init_seed=SMOKE_INIT_SEED,
            step=2,
            optimizer_state=state["optimizer_state"],  # type: ignore[arg-type]
            ema_state=state["ema_state"],  # type: ignore[arg-type]
            training_generator_state=state["training_generator_state"],  # type: ignore[arg-type]
            torch_cpu_rng_state=state["torch_cpu_rng_state"],  # type: ignore[arg-type]
            corpus_sha256=FROZEN_CORPUS_SHA256,
            metrics=metrics,
        )
        loaded_model, _, _ = load_codec_checkpoint(
            checkpoint_path, expected_corpus_sha256=FROZEN_CORPUS_SHA256
        )
        decode_hash = _decode_sha256(loaded_model, batch)
        manifest_core: dict[str, object] = {
            "format": SMOKE_FORMAT,
            "authority": "foundation_only_representation_codec_and_deterministic_compiler",
            "source_sha256": source_sha256(),
            "source_manifest": source_manifest(),
            "corpus_sha256": FROZEN_CORPUS_SHA256,
            "corpus_manifest_file_sha256": FROZEN_CORPUS_MANIFEST_FILE_SHA256,
            "tensor_contract_sha256": CONTRACT_SHA256,
            "compiler_source_sha256": compiler_source_sha256(),
            "theme_order": list(THEMES),
            "counts": {
                "themes": len(cases),
                "raw_artifacts": len(cases),
                "compiled_artifacts": len(cases),
                "codec_steps": 2,
            },
            "limits": {
                "disk_floor_gib": 100.0,
                "raw_array_bytes": 2 * 1024 * 1024,
                "compiled_array_bytes": 16 * 1024 * 1024,
                "checkpoint_bytes": 64 * 1024 * 1024,
                "process_policy": "single bounded CPU smoke; production work must be process-isolated",
            },
            "disk": disk_status(staging, floor_gb=100.0).to_dict(),
            "runtime": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "numpy": np.__version__,
                "pillow": PIL_VERSION,
                "torch": torch.__version__,
                "device": "cpu",
                "platform": sys.platform,
            },
            "cases": cases,
            "contact_sheet": {
                "file": CONTACT_SHEET,
                "bytes": len(contact),
                "sha256": file_sha256(staging / CONTACT_SHEET),
                "columns": ["source", "corrupt", "compiled", "edit_overlay"],
                "rows": list(THEMES),
                "visually_inspected": visually_inspected,
            },
            "codec": {
                "authority": "representation_only_not_generative",
                "config": config.to_dict(),
                "model_init_seed": SMOKE_INIT_SEED,
                "training_seed": SMOKE_TRAIN_SEED,
                "checkpoint": CODEC_CHECKPOINT,
                "checkpoint_sha256": sidecar["sha256"],
                "checkpoint_sidecar": CODEC_CHECKPOINT + ".json",
                "checkpoint_sidecar_sha256": file_sha256(checkpoint_path.with_suffix(".pt.json")),
                "ema_state_sha256": sidecar["ema_state_sha256"],
                "metrics": CODEC_METRICS,
                "metrics_sha256": file_sha256(metrics_path),
                "decode_sha256": decode_hash,
            },
            "gates": {
                "all_six_themes": True,
                "raw_artifacts_proposal_only": True,
                "compiled_authoritative_validation": True,
                "exact_compiler_replay": True,
                "contact_sheet_exact_replay": True,
                "checkpoint_bounded_safe_load": True,
                "cpu_only": True,
                "cuda_device_or_api_used": False,
                "production_training_started": False,
                "godot_integrated": False,
            },
        }
        manifest = {**manifest_core, "smoke_identity_sha256": json_sha256(manifest_core)}
        jsonschema.Draft202012Validator(_schema()).validate(manifest)
        _atomic_json(staging / SMOKE_MANIFEST, manifest)
        os.replace(staging, output)
    except BaseException:
        # Preserve the unique staging directory as crash evidence on this unstable host.
        raise
    return assert_exact_smoke_replay(output, write_report=True)


def assert_exact_smoke_replay(output: Path, *, write_report: bool = False) -> dict[str, object]:
    output = Path(output).resolve()
    manifest_path = output / SMOKE_MANIFEST
    if not manifest_path.is_file() or not 0 < manifest_path.stat().st_size <= MAX_SMOKE_MANIFEST_BYTES:
        raise ValueError("Neural topology smoke manifest is missing or exceeds its strict byte bound.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Neural topology smoke manifest root must be an object.")
    jsonschema.Draft202012Validator(_schema()).validate(manifest)
    core = {key: value for key, value in manifest.items() if key != "smoke_identity_sha256"}
    if json_sha256(core) != manifest.get("smoke_identity_sha256"):
        raise ValueError("Neural topology smoke identity payload drifted.")
    if (
        manifest.get("source_sha256") != source_sha256()
        or manifest.get("source_manifest") != source_manifest()
        or manifest.get("compiler_source_sha256") != compiler_source_sha256()
        or manifest.get("tensor_contract_sha256") != CONTRACT_SHA256
        or manifest.get("corpus_sha256") != FROZEN_CORPUS_SHA256
        or manifest.get("corpus_manifest_file_sha256") != FROZEN_CORPUS_MANIFEST_FILE_SHA256
    ):
        raise ValueError("Neural topology smoke source/corpus/contract provenance drifted.")
    corpus = TopologyCorpus(PROJECT_ROOT / "outputs" / "map_decorator_corpus_v1")
    rows = []
    tensors = []
    compared_arrays = 0
    compared_ledger_entries = 0
    for case in manifest["cases"]:
        theme = case["theme"]
        source = case["source"]
        sample = corpus.read_sample(source["shard_id"], source["sample_index"])
        if (
            sample.theme != theme
            or sample.topology_sample_sha256 != source["topology_sample_sha256"]
            or sample.raw.raw_sha256 != source["source_raw_topology_sha256"]
        ):
            raise ValueError("Smoke source sample drifted from frozen corpus identity.")
        raw_path = output / "cases" / theme / "raw"
        compiled_path = output / "cases" / theme / "compiled"
        raw_artifact = load_raw_artifact(raw_path)
        if raw_artifact.manifest_sha256 != case["raw"]["manifest_sha256"]:
            raise ValueError("Smoke raw manifest hash drifted.")
        compiled = load_compiled_artifact(
            compiled_path, raw_artifact=raw_artifact, exact_replay=True
        )
        if (
            compiled.manifest_sha256 != case["compiled"]["manifest_sha256"]
            or compiled.result.compiled_arrays_sha256 != case["compiled"]["compiled_arrays_sha256"]
            or compiled.result.ledger_sha256 != case["compiled"]["ledger_sha256"]
            or len(compiled.result.ledger) != case["compiled"]["ledger_entry_count"]
        ):
            raise ValueError("Smoke compiled artifact identity drifted.")
        compared_arrays += len(compiled.result.data.arrays()) + 3
        compared_ledger_entries += len(compiled.result.ledger)
        tensor = encode_topology_tensor(
            terrain=raw_artifact.raw.terrain,
            hazard=raw_artifact.raw.hazard,
            elevation=raw_artifact.raw.elevation,
            theme=raw_artifact.theme,
            config=raw_artifact.config,
            start=raw_artifact.start,
            exit=raw_artifact.exit,
            objectives=raw_artifact.objectives,
            spawns=raw_artifact.spawns,
            conditions=inferred_conditions(raw_artifact.raw.terrain, raw_artifact.raw.hazard),
        )
        if tensor.tensor_sha256 != case["tensor_sha256"]:
            raise ValueError("Smoke topology tensor replay drifted.")
        tensors.append(tensor)
        rows.append(
            (
                theme,
                (sample.raw.terrain, sample.raw.hazard, sample.raw.elevation),
                raw_artifact.raw,
                compiled.result.data,
                sample.start,
                sample.exit,
                sample.objectives,
                sample.spawns,
            )
        )
    expected_contact = contact_sheet_png_bytes(rows, scale=4)
    actual_contact = (output / CONTACT_SHEET).read_bytes()
    if (
        expected_contact != actual_contact
        or file_sha256(output / CONTACT_SHEET) != manifest["contact_sheet"]["sha256"]
        or len(actual_contact) != manifest["contact_sheet"]["bytes"]
    ):
        raise ValueError("Neural topology contact sheet failed exact byte replay.")
    codec = manifest["codec"]
    checkpoint_path = output / codec["checkpoint"]
    if (
        file_sha256(checkpoint_path) != codec["checkpoint_sha256"]
        or file_sha256(output / codec["checkpoint_sidecar"]) != codec["checkpoint_sidecar_sha256"]
        or file_sha256(output / codec["metrics"]) != codec["metrics_sha256"]
    ):
        raise ValueError("Neural topology codec artifact hash drifted.")
    model, payload, sidecar = load_codec_checkpoint(
        checkpoint_path, expected_corpus_sha256=FROZEN_CORPUS_SHA256
    )
    if sidecar["ema_state_sha256"] != codec["ema_state_sha256"]:
        raise ValueError("Neural topology codec EMA identity drifted.")
    batch = collate_topology_tensors(tensors)
    if _decode_sha256(model, batch) != codec["decode_sha256"]:
        raise ValueError("Neural topology codec exact decode replay drifted.")
    report: dict[str, object] = {
        "format": "nullvector-neural-map-topology-smoke-replay-v1",
        "passed": True,
        "smoke_identity_sha256": manifest["smoke_identity_sha256"],
        "manifest_sha256": file_sha256(manifest_path),
        "source_sha256": source_sha256(),
        "corpus_sha256": FROZEN_CORPUS_SHA256,
        "corpus_manifest_file_sha256": FROZEN_CORPUS_MANIFEST_FILE_SHA256,
        "theme_count": len(rows),
        "artifact_array_count_compared": compared_arrays,
        "ledger_entry_count_compared": compared_ledger_entries,
        "contact_sheet_exact_bytes": True,
        "checkpoint_bounded_safe_load": True,
        "checkpoint_step": payload["step"],
        "codec_decode_exact": True,
        "cpu_only": True,
    }
    if write_report:
        report_path = output / SMOKE_REPLAY
        if report_path.exists():
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            if existing != report:
                raise FileExistsError("Existing neural topology replay report disagrees with replay.")
        else:
            _atomic_json(report_path, report)
    return report
