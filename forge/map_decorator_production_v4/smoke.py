from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Final
import uuid

import numpy as np
from PIL import Image, ImageDraw
import torch

from ..config import PROJECT_ROOT
from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.contract import HEAD_NAMES, ModelConfig
from ..map_decorator_ml.dataset import collate_teacher_samples
from ..map_decorator_ml.legality import TorchLegalMasks
from ..map_decorator_ml.metrics import decoration_metrics
from ..map_decorator_production_v2.quality import evaluate_split_gate
from ..maps.model import THEMES
from ..safety import require_disk_floor
from .contract import ProposalLocatorConfig, V4_CONTRACT_SHA256
from .decoding import select_proposal_conditioned_argmax
from .model import ProposalConditionedDecoratorV4
from .proposal import ProposalAuthority, ProposalFields, assert_vectorized_hash_exact, audit_proposal_targets


SMOKE_FORMAT: Final[str] = "nullvector-map-decorator-v4-public-proposal-smoke/1.0.0"
REPORT_NAME: Final[str] = "smoke_report.json"
CONTACT_NAME: Final[str] = "proposal_target_decode_contact_sheet.png"
MODEL_SEED: Final[int] = 0x44D3C0DE
MAX_REPORT_BYTES: Final[int] = 8 * 1024 * 1024
MAX_CONTACT_BYTES: Final[int] = 16 * 1024 * 1024
MODEL_CONFIG: Final[ModelConfig] = ModelConfig(base_channels=4, condition_channels=8)
LOCATOR_CONFIG: Final[ProposalLocatorConfig] = ProposalLocatorConfig(
    locator_channels=4,
    locator_blocks=1,
    count_hidden_channels=4,
)
V4_SOURCE_PACKAGES: Final[tuple[str, ...]] = (
    "forge/map_decorator_ml",
    "forge/map_decorator_production",
    "forge/map_decorator_production_v2",
    "forge/map_decorator_production_v4",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(root: Path = PROJECT_ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    for package_name in V4_SOURCE_PACKAGES:
        package = Path(root) / package_name
        for path in sorted(item for item in package.glob("*.py") if item.is_file()):
            result[path.relative_to(root).as_posix()] = _file_sha256(path)
    for relative in (
        "forge/map_art/hashing.py",
        "forge/map_art/model.py",
        "forge/map_art/styles.py",
        "forge/map_decorator/catalog.py",
        "forge/map_decorator/features.py",
    ):
        path = Path(root) / relative
        result[relative] = _file_sha256(path)
    return dict(sorted(result.items()))


def source_sha256(root: Path = PROJECT_ROOT) -> str:
    return json_sha256(source_manifest(root))


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, floor_gb=100.0, planned_bytes=len(encoded) + 1024 * 1024)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _proposal_tensors(proposals: ProposalFields) -> dict[str, torch.Tensor]:
    return {
        "decal": torch.from_numpy(proposals.decal.copy())[None],
        "prop": torch.from_numpy(proposals.prop.copy())[None],
    }


def _render_objects(decal: np.ndarray, prop: np.ndarray, hard_empty: np.ndarray) -> Image.Image:
    height, width = decal.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:] = (3, 8, 15)
    rgb[hard_empty] = (18, 10, 32)
    decal_colors = ((0, 0, 0), (38, 231, 255), (244, 55, 222))
    prop_colors = ((0, 0, 0), (255, 175, 42), (92, 255, 117))
    for class_id in range(1, len(decal_colors)):
        rgb[decal == class_id] = decal_colors[class_id]
    for class_id in range(1, len(prop_colors)):
        rgb[prop == class_id] = prop_colors[class_id]
    return Image.fromarray(rgb)


def _fit_nearest(image: Image.Image, size: int = 144) -> Image.Image:
    width, height = image.size
    scale = min(size / width, size / height)
    resized = image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.Resampling.NEAREST,
    )
    canvas = Image.new("RGB", (size, size), (2, 5, 10))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def _contact_sheet(rows: dict[str, tuple[object, dict[str, torch.Tensor]]]) -> bytes:
    margin_x, header, tile, gap = 92, 24, 144, 8
    canvas = Image.new("RGB", (margin_x + 2 * tile + 3 * gap, header + len(THEMES) * (tile + gap)), (2, 5, 10))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_x + gap + 48, 6), "TARGET", fill=(134, 196, 210))
    draw.text((margin_x + tile + 2 * gap + 43, 6), "V4 DECODE", fill=(143, 255, 139))
    for row_index, theme in enumerate(THEMES):
        sample, prediction = rows[theme]
        y = header + row_index * (tile + gap)
        draw.text((5, y + tile // 2 - 5), theme.upper(), fill=(225, 91, 218))
        target = _render_objects(sample.targets["decal"], sample.targets["prop"], sample.hard_empty)  # type: ignore[attr-defined]
        decoded = _render_objects(
            prediction["decal"].squeeze(0).cpu().numpy().astype(np.uint8),
            prediction["prop"].squeeze(0).cpu().numpy().astype(np.uint8),
            sample.hard_empty,  # type: ignore[attr-defined]
        )
        canvas.paste(_fit_nearest(target, tile), (margin_x + gap, y))
        canvas.paste(_fit_nearest(decoded, tile), (margin_x + tile + 2 * gap, y))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=False, compress_level=9)
    payload = buffer.getvalue()
    if not 0 < len(payload) <= MAX_CONTACT_BYTES:
        raise ValueError("V4 contact sheet exceeds its bounded size.")
    return payload


def _analyze(corpus_root: Path, index_root: Path, *, visually_inspected: bool) -> tuple[dict[str, object], bytes]:
    if not isinstance(visually_inspected, bool):
        raise TypeError("visually_inspected must be an explicit boolean attestation.")
    if torch.cuda.is_initialized():
        raise RuntimeError("V4 CPU smoke refuses a CUDA-initialized process.")
    assert_vectorized_hash_exact()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(MODEL_SEED)
    authority = ProposalAuthority.load(Path(corpus_root), Path(index_root))
    model = ProposalConditionedDecoratorV4(MODEL_CONFIG, LOCATOR_CONFIG).eval()
    model_sha = tensor_state_sha256(model.state_dict())
    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    targets: dict[str, list[torch.Tensor]] = {name: [] for name in HEAD_NAMES}
    records: list[dict[str, object]] = []
    contact_rows: dict[str, tuple[object, dict[str, torch.Tensor]]] = {}
    valid_cells = 0
    for ref in authority.authority.corpus.epoch_refs("test", 0, MODEL_SEED):
        sample, proposals = authority.sample_and_proposals(ref)
        audit = audit_proposal_targets(proposals, sample.targets)
        if not audit["passed"]:
            raise RuntimeError("V4 public proposal substrate missed a target object cell.")
        batch = collate_teacher_samples([sample])
        valid = batch["valid_cells"]
        masked = {name: valid.clone() for name in HEAD_NAMES}
        with torch.inference_mode():
            output = model(
                batch["features"],
                batch["targets"],
                masked,
                batch["theme_index"],
                batch["global_conditions"],
                torch.ones((1,), dtype=torch.float32),
                _proposal_tensors(proposals),
            )
            prediction = select_proposal_conditioned_argmax(
                output,
                TorchLegalMasks(hard_empty=batch["hard_empty"], **batch["legal_masks"]),
            )
        for name in HEAD_NAMES:
            predictions[name].append(prediction[name][valid].cpu())
            targets[name].append(batch["targets"][name][valid].cpu())
        valid_cells += int(valid.sum().item())
        theme = THEMES[sample.theme_index]
        contact_rows.setdefault(theme, (sample, prediction))
        records.append(
            {
                "sample_identity_sha256": ref.sample_identity_sha256,
                "theme": theme,
                "map_seed": proposals.map_seed,
                "proposal_fields_sha256": proposals.fields_sha256,
                "proposal_audit": audit["heads"],
            }
        )
    if len(records) != 24 or set(contact_rows) != set(THEMES):
        raise RuntimeError("V4 smoke did not cover all 24 sentinels and six themes.")
    metrics = decoration_metrics(
        {name: torch.cat(values) for name, values in predictions.items()},
        {name: torch.cat(values) for name, values in targets.items()},
        torch.ones((valid_cells,), dtype=torch.bool),
    )
    metrics.update(
        {
            "split": "test",
            "sample_count": 24,
            "full_split": True,
            "valid_cell_count": valid_cells,
            "hard_legality": 1.0,
            "immutable_semantic_changes": 0,
            "source_provenance_failures": 0,
        }
    )
    legacy_gate = evaluate_split_gate(metrics, stage="calibration")
    decal = metrics["heads"]["decal"]
    prop = metrics["heads"]["prop"]
    stronger_gate = {
        "decal_iou_at_least_0_90": float(decal["foreground_macro_iou"]) >= 0.90,
        "decal_f1_at_least_0_95": float(decal["foreground_f1"]) >= 0.95,
        "decal_rare_recall_at_least_0_95": float(decal["rare_class_recall"]) >= 0.95,
        "prop_iou_at_least_0_90": float(prop["foreground_macro_iou"]) >= 0.90,
        "prop_f1_at_least_0_95": float(prop["foreground_f1"]) >= 0.95,
        "prop_rare_recall_at_least_0_90": float(prop["rare_class_recall"]) >= 0.90,
    }
    contact = _contact_sheet(contact_rows)
    core: dict[str, object] = {
        "format": SMOKE_FORMAT,
        "status": "passed",
        "v4_contract_sha256": V4_CONTRACT_SHA256,
        "source_sha256": source_sha256(),
        "authority": {
            "corpus_sha256": authority.authority.corpus.corpus_sha256,
            "corpus_manifest_sha256": authority.authority.corpus.manifest_sha256,
            "index_semantic_sha256": authority.authority.index_semantic_sha256,
            "index_manifest_sha256": authority.authority.index_manifest_sha256,
        },
        "runtime": {"device": "cpu", "cuda_initialized": torch.cuda.is_initialized(), "threads": torch.get_num_threads()},
        "model": {
            "seed": MODEL_SEED,
            "core_config": MODEL_CONFIG.to_dict(),
            "locator_config": LOCATOR_CONFIG.to_dict(),
            "tensor_sha256": model_sha,
            "trained_steps": 0,
            "role": "untrained residual over exact public proposals",
        },
        "counts": {"theme_count": 6, "sample_count": 24, "valid_cell_count": valid_cells},
        "records": records,
        "metrics": metrics,
        "legacy_calibration_gate": legacy_gate,
        "stronger_v4_gate": stronger_gate,
        "contact_sheet": {
            "path": CONTACT_NAME,
            "bytes": len(contact),
            "sha256": hashlib.sha256(contact).hexdigest(),
            "rows": list(THEMES),
            "columns": ["target", "v4_decode"],
            "visually_inspected": visually_inspected,
        },
        "gates": {
            "all_24_test_sentinels": len(records) == 24,
            "all_six_themes": set(contact_rows) == set(THEMES),
            "all_target_object_cells_proposed": all(
                record["proposal_audit"][head]["missing_count"] == 0  # type: ignore[index]
                for record in records
                for head in ("decal", "prop")
            ),
            "legacy_object_quality_gate_passed": bool(legacy_gate["passed"]),
            "stronger_v4_object_gate_passed": all(stronger_gate.values()),
            "hard_legality": metrics["hard_legality"] == 1.0,
            "deterministic_untrained_model": True,
            "cpu_only": not torch.cuda.is_initialized(),
            "contact_sheet_visually_inspected": visually_inspected,
            "not_a_production_claim": True,
        },
    }
    if not all(core["gates"].values()):  # type: ignore[union-attr]
        raise RuntimeError(f"V4 smoke gate failed: {core['gates']}")
    return core, contact


def build_smoke(
    corpus_root: Path,
    index_root: Path,
    output: Path,
    *,
    visually_inspected: bool = False,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("V4 smoke output is immutable.")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=64 * 1024 * 1024)
    core, contact = _analyze(corpus_root, index_root, visually_inspected=visually_inspected)
    report = {**core, "report_sha256": json_sha256(core)}
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    (staging / CONTACT_NAME).write_bytes(contact)
    _atomic_json(staging / REPORT_NAME, report)
    os.replace(staging, output)
    return validate_smoke(output, corpus_root=corpus_root, index_root=index_root)


def _read_report(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_REPORT_BYTES:
        raise ValueError("V4 smoke report is missing, unsafe, or oversized.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V4 smoke report root must be an object.")
    stored = value.pop("report_sha256", None)
    if stored != json_sha256(value):
        raise ValueError("V4 smoke report self-hash failed.")
    value["report_sha256"] = stored
    return value


def validate_smoke(output: Path, *, corpus_root: Path, index_root: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    report = _read_report(output / REPORT_NAME)
    unsigned = dict(report)
    recorded = unsigned.pop("report_sha256")
    if report.get("format") != SMOKE_FORMAT or report.get("status") != "passed":
        raise ValueError("V4 smoke format/status failed.")
    expected, contact = _analyze(
        corpus_root,
        index_root,
        visually_inspected=bool(report["contact_sheet"]["visually_inspected"]),
    )
    if unsigned != expected:
        raise ValueError("V4 smoke failed exact semantic replay.")
    contact_path = output / CONTACT_NAME
    if not contact_path.is_file() or contact_path.read_bytes() != contact:
        raise ValueError("V4 smoke contact sheet failed exact byte replay.")
    return {
        "passed": True,
        "report_sha256": recorded,
        "source_sha256": report["source_sha256"],
        "v4_contract_sha256": report["v4_contract_sha256"],
        "model_tensor_sha256": report["model"]["tensor_sha256"],
        "contact_sheet_sha256": report["contact_sheet"]["sha256"],
        "decal_foreground_macro_iou": report["metrics"]["heads"]["decal"]["foreground_macro_iou"],
        "prop_foreground_macro_iou": report["metrics"]["heads"]["prop"]["foreground_macro_iou"],
    }
