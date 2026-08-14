from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Final

import numpy as np
from PIL import Image, ImageDraw
import torch

from ..config import PROJECT_ROOT
from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.checkpoint import file_sha256
from ..map_decorator_ml.contract import HEAD_NAMES, ModelConfig
from ..map_decorator_ml.legality import TorchLegalMasks
from ..map_decorator_ml.metrics import decoration_metrics
from ..map_decorator_production_v4.contract import ProposalLocatorConfig
from ..map_decorator_production_v4.decoding import select_proposal_conditioned_argmax
from ..map_decorator_production_v4.model import ProposalConditionedDecoratorV4
from ..map_decorator_production_v4.proposal import ProposalAuthority
from ..map_decorator_production_v4_calibration.evaluation import compare_to_baseline
from ..map_decorator_production_v4_calibration.runner import (
    REPORT_NAME as CALIBRATION_REPORT_NAME,
    calibration_source_sha256,
    validate_supervised,
)
from ..map_decorator_production_v4_training.checkpoint import inspect_checkpoint, tensor_state_sha256
from ..map_decorator_production_v4_training.dataset import ProposalTeacherSample, collate_proposal_samples
from ..maps.model import THEMES
from ..safety import require_disk_floor
from .contract import AUDIT_FORMAT, ProtectedSelectionConfig, V4_SELECTION_CONTRACT_SHA256
from .decoder import select_protected_proposal_argmax


REPORT_NAME: Final[str] = "selection_audit.json"
CONTACT_NAME: Final[str] = "target_trained_protected_contact_sheet.png"
MAX_REPORT_BYTES: Final[int] = 32 * 1024 * 1024


def selection_source_manifest(root: Path = PROJECT_ROOT) -> dict[str, object]:
    package = Path(root) / "forge/map_decorator_production_v4_selection"
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in package.glob("*.py") if item.is_file())
    }
    return {"calibration_source_sha256": calibration_source_sha256(root), "selection_files": files}


def selection_source_sha256(root: Path = PROJECT_ROOT) -> str:
    return json_sha256(selection_source_manifest(root))


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=100.0, planned_bytes=len(encoded) + 1024 * 1024)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _render(decal: np.ndarray, prop: np.ndarray, hard_empty: np.ndarray) -> Image.Image:
    height, width = decal.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:] = (3, 8, 15)
    rgb[hard_empty] = (18, 10, 32)
    for class_id, color in ((1, (38, 231, 255)), (2, (244, 55, 222))):
        rgb[decal == class_id] = color
    for class_id, color in ((1, (255, 175, 42)), (2, (92, 255, 117))):
        rgb[prop == class_id] = color
    return Image.fromarray(rgb)


def _fit(image: Image.Image, size: int) -> Image.Image:
    scale = min(size / image.width, size / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size, size), (2, 5, 10))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def _contact(rows: dict[str, tuple[object, dict[str, np.ndarray], dict[str, np.ndarray]]]) -> bytes:
    label, tile, gap, header = 92, 144, 8, 24
    canvas = Image.new("RGB", (label + 3 * tile + 4 * gap, header + 6 * (tile + gap)), (2, 5, 10))
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(("TARGET", "TRAINED EMA", "PROTECTED")):
        draw.text((label + gap + index * (tile + gap) + 42, 6), name, fill=(140, 225, 235))
    for row_index, theme in enumerate(THEMES):
        sample, trained, protected = rows[theme]
        y = header + row_index * (tile + gap)
        draw.text((5, y + tile // 2 - 5), theme.upper(), fill=(225, 91, 218))
        images = (
            _render(sample.targets["decal"], sample.targets["prop"], sample.hard_empty),  # type: ignore[attr-defined]
            _render(trained["decal"], trained["prop"], sample.hard_empty),  # type: ignore[attr-defined]
            _render(protected["decal"], protected["prop"], sample.hard_empty),  # type: ignore[attr-defined]
        )
        for column, image in enumerate(images):
            canvas.paste(_fit(image, tile), (label + gap + column * (tile + gap), y))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _configure(seed: int) -> torch.device:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required before protected audit CUDA startup.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Protected selection audit requires CUDA BF16.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.device("cuda", 0)


def _derive_config(calibration: dict[str, object]) -> ProtectedSelectionConfig:
    observed: set[int] | None = None
    for split in ("validation", "test"):
        rare = set(map(int, calibration["baseline_evaluation"][split]["heads"]["decal"]["rare_class_ids"]))  # type: ignore[index]
        observed = rare if observed is None else observed & rare
    if observed != {2}:
        raise ValueError("Protected decal class derivation differs from the frozen baseline registry.")
    return ProtectedSelectionConfig(decal_classes=tuple(sorted(observed)))


def _evaluate(
    model: ProposalConditionedDecoratorV4,
    authority: ProposalAuthority,
    split: str,
    config: ProtectedSelectionConfig,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, object], dict[str, tuple[object, dict[str, np.ndarray], dict[str, np.ndarray]]]]:
    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    targets: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    identities: list[str] = []
    restored: dict[str, dict[str, int]] = {"decal": {}, "prop": {}}
    rows: dict[str, tuple[object, dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    valid_count = 0
    model.eval()
    for ref in authority.authority.corpus.epoch_refs(split, 0, seed):
        sample, proposal = authority.sample_and_proposals(ref)
        batch = collate_proposal_samples([ProposalTeacherSample(sample, proposal)])
        features = batch["features"].to(device)
        truth = {name: batch["targets"][name].to(device) for name in HEAD_NAMES}
        valid = batch["valid_cells"].to(device)
        hard = batch["hard_empty"].to(device)
        legal_masks = {name: batch["legal_masks"][name].to(device) for name in HEAD_NAMES}
        proposals = {name: batch["proposals"][name].to(device) for name in ("decal", "prop")}
        masked = {name: valid.clone() for name in HEAD_NAMES}
        legal = TorchLegalMasks(hard_empty=hard, **legal_masks)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                features, truth, masked, batch["theme_index"].to(device),
                batch["global_conditions"].to(device), torch.ones((1,), device=device), proposals,
            )
            trained = select_proposal_conditioned_argmax(output, legal)
            protected, diagnostics = select_protected_proposal_argmax(output, legal, config=config)
        for head, values in diagnostics["restored"].items():
            for class_id, count in values.items():
                restored[head][class_id] = restored[head].get(class_id, 0) + int(count)
        for name in HEAD_NAMES:
            predictions[name].append(protected[name][valid].cpu())
            targets[name].append(truth[name][valid].cpu())
        valid_count += int(valid.sum())
        identities.append(ref.sample_identity_sha256)
        theme = THEMES[sample.theme_index]
        if split == "test" and theme not in rows:
            rows[theme] = (
                sample,
                {name: trained[name].squeeze(0).cpu().numpy().astype(np.uint8) for name in ("decal", "prop")},
                {name: protected[name].squeeze(0).cpu().numpy().astype(np.uint8) for name in ("decal", "prop")},
            )
    metrics = decoration_metrics(
        {name: torch.cat(values) for name, values in predictions.items()},
        {name: torch.cat(values) for name, values in targets.items()},
        torch.ones((valid_count,), dtype=torch.bool),
    )
    metrics.update({
        "split": split,
        "sample_count": len(identities),
        "sample_set_sha256": json_sha256(sorted(identities)),
        "full_split": len(identities) == len(authority.authority.corpus.refs_by_split[split]),
        "valid_cell_count": valid_count,
        "hard_legality": 1.0,
        "immutable_semantic_changes": 0,
        "source_provenance_failures": 0,
        "restored": restored,
    })
    return metrics, rows


def build_audit(
    calibration_root: Path,
    corpus_root: Path,
    index_root: Path,
    output: Path,
    *,
    visually_inspected: bool = False,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Protected selection audit output is immutable.")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=512 * 1024 * 1024)
    validated = validate_supervised(Path(calibration_root), corpus_root=corpus_root, index_root=index_root)
    calibration = validated["calibration"]
    config = _derive_config(calibration)
    device = _configure(int(calibration["config"]["training"]["seed"]))
    checkpoint_path = Path(calibration_root) / "calibration" / calibration["checkpoint"]["path"]
    payload = inspect_checkpoint(checkpoint_path)
    model = ProposalConditionedDecoratorV4(
        ModelConfig(**payload["core_config"]), ProposalLocatorConfig(**payload["locator_config"])
    ).to(device)
    model.load_state_dict(payload["ema_state"]["shadow"], strict=True)
    if tensor_state_sha256(model.state_dict()) != calibration["ema_tensor_sha256"]:
        raise ValueError("Protected selection did not load the exact calibrated EMA tensors.")
    authority = ProposalAuthority.load(Path(corpus_root), Path(index_root))
    started = time.perf_counter()
    evaluation: dict[str, object] = {}
    contact_rows = {}
    for split in ("validation", "test"):
        metrics, rows = _evaluate(model, authority, split, config, device, payload["training_config"]["seed"])
        evaluation[split] = metrics
        contact_rows.update(rows)
    if set(contact_rows) != set(THEMES):
        raise RuntimeError("Protected selection contact sheet did not cover all six themes.")
    vs_baseline = compare_to_baseline(calibration["baseline_evaluation"], evaluation)
    vs_trained = compare_to_baseline(calibration["ema_evaluation"], evaluation)
    contact = _contact(contact_rows)
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}"
    staging.mkdir(parents=True, exist_ok=False)
    (staging / CONTACT_NAME).write_bytes(contact)
    report: dict[str, object] = {
        "format": AUDIT_FORMAT,
        "status": "quality_passed" if vs_baseline["passed"] and vs_trained["passed"] else "quality_failed",
        "selection_contract_sha256": V4_SELECTION_CONTRACT_SHA256,
        "selection_source_sha256": selection_source_sha256(),
        "calibration_source_sha256": calibration_source_sha256(),
        "calibration_report_sha256": file_sha256(Path(calibration_root) / "calibration" / CALIBRATION_REPORT_NAME),
        "checkpoint_sha256": calibration["checkpoint"]["sha256"],
        "ema_tensor_sha256": calibration["ema_tensor_sha256"],
        "config": config.to_dict(),
        "evaluation": evaluation,
        "versus_procedural_baseline": vs_baseline,
        "versus_trained_ema": vs_trained,
        "runtime": {
            "device": str(device),
            "precision": "bf16",
            "elapsed_seconds": time.perf_counter() - started,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "contact_sheet": {
            "path": CONTACT_NAME,
            "bytes": len(contact),
            "sha256": hashlib.sha256(contact).hexdigest(),
            "visually_inspected": bool(visually_inspected),
        },
        "gates": {
            "rare_class_registry_derived_exactly": config.decal_classes == (2,),
            "full_validation_and_test": all(evaluation[split]["full_split"] for split in ("validation", "test")),  # type: ignore[index]
            "hard_legality_exact": all(evaluation[split]["hard_legality"] == 1.0 for split in ("validation", "test")),  # type: ignore[index]
            "nonregressing_and_improved_vs_baseline": bool(vs_baseline["passed"]),
            "nonregressing_and_improved_vs_trained_ema": bool(vs_trained["passed"]),
            "no_runtime_integration_performed": True,
        },
    }
    report["semantic_sha256"] = json_sha256({key: value for key, value in report.items() if key != "runtime"})
    report["report_sha256"] = json_sha256(report)
    _atomic_json(staging / REPORT_NAME, report)
    os.replace(staging, output)
    return validate_audit(output, calibration_root=calibration_root, corpus_root=corpus_root, index_root=index_root)


def validate_audit(
    output: Path,
    *,
    calibration_root: Path,
    corpus_root: Path,
    index_root: Path,
) -> dict[str, Any]:
    output = Path(output).resolve()
    report_path = output / REPORT_NAME
    if not report_path.is_file() or report_path.is_symlink() or not 0 < report_path.stat().st_size <= MAX_REPORT_BYTES:
        raise ValueError("Protected selection report is missing, unsafe, or oversized.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stored = report.pop("report_sha256", None)
    if stored != json_sha256(report):
        raise ValueError("Protected selection report self-hash failed.")
    report["report_sha256"] = stored
    if report.get("format") != AUDIT_FORMAT or report.get("selection_contract_sha256") != V4_SELECTION_CONTRACT_SHA256:
        raise ValueError("Protected selection format or contract drifted.")
    if report.get("selection_source_sha256") != selection_source_sha256():
        raise ValueError("Protected selection source drifted.")
    calibration = validate_supervised(Path(calibration_root), corpus_root=corpus_root, index_root=index_root)["calibration"]
    if report.get("calibration_report_sha256") != file_sha256(Path(calibration_root) / "calibration" / CALIBRATION_REPORT_NAME):
        raise ValueError("Protected selection calibration identity drifted.")
    config = ProtectedSelectionConfig(
        decal_classes=tuple(report["config"]["decal_classes"]),
        prop_classes=tuple(report["config"]["prop_classes"]),
        restore_only_into_empty=report["config"]["restore_only_into_empty"],
        preserve_cross_head_noncollision=report["config"]["preserve_cross_head_noncollision"],
    )
    if config != _derive_config(calibration) or report["config"] != config.to_dict():
        raise ValueError("Protected selection configuration drifted.")
    expected_baseline = compare_to_baseline(calibration["baseline_evaluation"], report["evaluation"])
    expected_trained = compare_to_baseline(calibration["ema_evaluation"], report["evaluation"])
    if report.get("versus_procedural_baseline") != expected_baseline or report.get("versus_trained_ema") != expected_trained:
        raise ValueError("Protected selection comparison replay failed.")
    contact = output / CONTACT_NAME
    if contact.stat().st_size != report["contact_sheet"]["bytes"] or file_sha256(contact) != report["contact_sheet"]["sha256"]:
        raise ValueError("Protected selection contact sheet identity failed.")
    semantic = json_sha256({key: value for key, value in report.items() if key not in {"runtime", "semantic_sha256", "report_sha256"}})
    if semantic != report.get("semantic_sha256"):
        raise ValueError("Protected selection semantic hash failed.")
    expected_gates = {
        "rare_class_registry_derived_exactly": config.decal_classes == (2,),
        "full_validation_and_test": all(report["evaluation"][split]["full_split"] for split in ("validation", "test")),
        "hard_legality_exact": all(report["evaluation"][split]["hard_legality"] == 1.0 for split in ("validation", "test")),
        "nonregressing_and_improved_vs_baseline": bool(expected_baseline["passed"]),
        "nonregressing_and_improved_vs_trained_ema": bool(expected_trained["passed"]),
        "no_runtime_integration_performed": True,
    }
    if report.get("gates") != expected_gates or not all(expected_gates.values()):
        raise ValueError("Protected selection acceptance gate failed.")
    if report.get("status") != "quality_passed":
        raise ValueError("Protected selection did not achieve its quality contract.")
    return report
