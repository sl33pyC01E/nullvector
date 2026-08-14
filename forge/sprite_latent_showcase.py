from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import tempfile
from typing import Any, Final, Mapping

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image, ImageDraw, __version__ as PILLOW_VERSION
import torch

from .config import PROJECT_ROOT
from .map_decorator.hashing import json_sha256
from .multifield_style.hashing import sha256_file
from .multifield_style_motion.hashing import (
    artifact_record_from_bytes,
    canonical_json_bytes,
    deterministic_npz_bytes,
    sha256_bytes,
)
from .safety import require_disk_floor
from .sprite_latent.codec import SemanticSpriteFSQ, project_legal_tuples
from .sprite_latent.corpus import (
    SemanticFieldCorpus,
    compute_legal_tuples,
    legal_tuple_fingerprint,
    stratified_split,
)
from .sprite_latent.smoke import FAMILY_COLORS, _rgba
from .sprite_latent_production.checkpoint import load_checkpoint
from .sprite_latent_production.contract import ProductionConfig
from .sprite_latent_production.evaluation import batch_from_indices
from .sprite_latent_production.supervisor import validate_production_manifest


FORMAT: Final[str] = "nullvector-semantic-sprite-fsq-production-showcase-v1"
DEFAULT_PRODUCTION: Final[Path] = PROJECT_ROOT / "checkpoints/sprite_latent_production_v1_1/production_manifest.json"
DEFAULT_CORPUS: Final[Path] = PROJECT_ROOT / "data/morphology_32768_4d4f5250.npz"
DEFAULT_OUTPUT: Final[Path] = PROJECT_ROOT / "outputs/sprite_latent_production_showcase_v1"
SCHEMA_PATH: Final[Path] = PROJECT_ROOT / "shared/schema/sprite_latent_showcase.schema.json"
FAMILY_NAMES: Final[tuple[str, ...]] = ("humanoid", "animalian", "plantlike", "anomaly", "machine")
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/sprite_latent_showcase.py",
    "shared/schema/sprite_latent_showcase.schema.json",
)


def source_sha256() -> str:
    digest = hashlib.sha256(b"nullvector-sprite-latent-showcase-source-v1\0")
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _select_indices(corpus: SemanticFieldCorpus, validation: np.ndarray) -> np.ndarray:
    selected: list[int] = []
    for family in range(5):
        candidates = [int(index) for index in validation if int(corpus.morphologies[index]) == family]
        # Prefer new subtype/role combinations, then fill by canonical index.
        chosen: list[int] = []
        seen_subtypes: set[int] = set(); seen_roles: set[int] = set()
        for index in candidates:
            subtype, role = int(corpus.subtypes[index]), int(corpus.roles[index])
            if subtype not in seen_subtypes or role not in seen_roles:
                chosen.append(index); seen_subtypes.add(subtype); seen_roles.add(role)
            if len(chosen) == 8: break
        for index in candidates:
            if len(chosen) == 8: break
            if index not in chosen: chosen.append(index)
        if len(chosen) != 8:
            raise ValueError(f"Validation split lacks eight showcase samples for family {family}")
        selected.extend(chosen)
    return np.asarray(selected, dtype=np.int64)


def _sample_metrics(target: Mapping[str, torch.Tensor], predicted: Mapping[str, torch.Tensor], offset: int) -> dict[str, float]:
    part_ok = predicted["part"][offset] == target["part"][offset]
    material_ok = predicted["material"][offset] == target["material"][offset]
    emission_ok = predicted["emission"][offset] == target["emission"][offset]
    aligned = part_ok & material_ok & emission_ok
    visible = target["part"][offset] != 0
    predicted_visible = predicted["part"][offset] != 0
    return {
        "aligned_tuple_accuracy": round(float(aligned.float().mean()), 9),
        "visible_tuple_accuracy": round(float(aligned[visible].float().mean()) if bool(visible.any()) else 1.0, 9),
        "visible_silhouette_iou": round(float((visible & predicted_visible).sum()) / max(1, int((visible | predicted_visible).sum())), 9),
    }


def _contact_sheet(corpus: SemanticFieldCorpus, indices: np.ndarray, predicted: Mapping[str, np.ndarray]) -> bytes:
    cell, scale, gap, label_width = 48, 3, 10, 88
    pair_width = cell * scale * 2 + gap
    header, row_height = 48, cell * scale + 26
    canvas = Image.new("RGB", (label_width + pair_width * 8 + 8, header + row_height * 5), (4, 7, 16))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "PRODUCTION FSQ EMA // HELD-OUT ORIGINAL -> RECONSTRUCTION", fill=(203, 243, 255))
    draw.text((10, 24), "40 validation specimens / 8 per family / native 48px / legal tuple projection", fill=(76, 157, 190))
    for family in range(5):
        y = header + family * row_height
        draw.text((5, y + 61), FAMILY_NAMES[family], fill=FAMILY_COLORS[family])
        for column in range(8):
            offset = family * 8 + column; index = int(indices[offset]); x = label_width + column * pair_width
            original = Image.fromarray(_rgba(corpus.part_owner[index], corpus.material[index], corpus.emission_level[index], family)).resize((cell * scale, cell * scale), Image.Resampling.NEAREST)
            decoded = Image.fromarray(_rgba(predicted["part"][offset], predicted["material"][offset], predicted["emission"][offset], family)).resize((cell * scale, cell * scale), Image.Resampling.NEAREST)
            canvas.paste(original.convert("RGB"), (x, y)); canvas.paste(decoded.convert("RGB"), (x + cell * scale + 5, y))
            draw.text((x, y + cell * scale + 3), f"{index:05d}", fill=(115, 158, 179))
    buffer = BytesIO(); canvas.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _compile(production_manifest: Path, corpus_path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    production_manifest = Path(production_manifest).resolve(); production = validate_production_manifest(production_manifest)
    if production["status"] != "ready" or not all(production["gates"].values()):
        raise ValueError("Sprite latent production authority is not quality accepted")
    checkpoint_path = production_manifest.parent.joinpath(*PurePosixPath(production["best"]["checkpoint"]).parts)
    if sha256_file(checkpoint_path) != production["best"]["checkpoint_sha256"]:
        raise ValueError("Sprite latent best checkpoint hash differs")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    corpus = SemanticFieldCorpus.load(Path(corpus_path), expected_file_sha256=production["corpus_sha256"])
    split = stratified_split(corpus)
    if split.fingerprint != production["split_fingerprint"]:
        raise ValueError("Sprite latent showcase split provenance differs")
    legal_array = compute_legal_tuples(corpus, split.training)
    if legal_tuple_fingerprint(legal_array) != production["legal_tuple_fingerprint"]:
        raise ValueError("Sprite latent showcase legal tuples differ")
    indices = _select_indices(corpus, split.validation)
    config = ProductionConfig.from_metadata(checkpoint["config"])
    model = SemanticSpriteFSQ(config.codec_config()); model.load_state_dict(checkpoint["ema_state"], strict=True); model.eval()
    previous_threads = torch.get_num_threads(); previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    try:
        batch = batch_from_indices(corpus, indices, torch.device("cpu"))
        legal = torch.from_numpy(legal_array.astype(np.int64))
        with torch.no_grad():
            output = model(batch["part"], batch["material"], batch["emission"], batch["morphology"], batch["subtype"], batch["role"], batch["genes"], quantize=True)
            projected = project_legal_tuples(output, legal)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic); torch.set_num_threads(previous_threads)
    predicted = {name: projected[name].cpu().numpy().astype(np.uint8) for name in ("part", "material", "emission")}
    samples = []
    for offset, index_raw in enumerate(indices):
        index = int(index_raw)
        samples.append({
            "source_index": index, "seed": int(corpus.seeds[index]), "family": FAMILY_NAMES[int(corpus.morphologies[index])],
            "family_id": int(corpus.morphologies[index]), "subtype_id": int(corpus.subtypes[index]), "role_id": int(corpus.roles[index]),
            **_sample_metrics(batch, projected, offset),
        })
    arrays = {
        "source_indices": indices.astype(np.int64), "part": predicted["part"],
        "material": predicted["material"], "emission": predicted["emission"],
    }
    fields_payload = deterministic_npz_bytes(arrays); contact_payload = _contact_sheet(corpus, indices, predicted)
    files = {"reconstruction_fields.npz": fields_payload, "reconstruction_contact_sheet.png": contact_payload}
    manifest: dict[str, Any] = {
        "format": FORMAT, "status": "ready", "quality_scope": "held-out-balanced-production-reconstruction-v1",
        "compiler": {"source_sha256": source_sha256(), "python_runtime_required": False},
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__), "pillow": str(PILLOW_VERSION), "device": "cpu"},
        "source": {
            "production_manifest": production_manifest.relative_to(PROJECT_ROOT).as_posix(), "production_manifest_sha256": sha256_file(production_manifest),
            "production_semantic_sha256": production["manifest_sha256"], "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path), "checkpoint_epoch": int(checkpoint["epoch"]), "ema_state_sha256": checkpoint["ema_state_sha256"],
            "corpus": Path(corpus_path).resolve().relative_to(PROJECT_ROOT).as_posix(), "corpus_sha256": corpus.file_sha256,
            "split_fingerprint": split.fingerprint, "legal_tuple_fingerprint": legal_tuple_fingerprint(legal_array),
        },
        "sample_count": 40, "samples_per_family": 8, "family_count": 5, "samples": samples,
        "aggregate": {
            "aligned_tuple_accuracy": round(float(np.mean([item["aligned_tuple_accuracy"] for item in samples])), 9),
            "visible_tuple_accuracy": round(float(np.mean([item["visible_tuple_accuracy"] for item in samples])), 9),
            "visible_silhouette_iou": round(float(np.mean([item["visible_silhouette_iou"] for item in samples])), 9),
            "minimum_sample_visible_tuple_accuracy": round(float(min(item["visible_tuple_accuracy"] for item in samples)), 9),
            "minimum_sample_visible_silhouette_iou": round(float(min(item["visible_silhouette_iou"] for item in samples)), 9),
        },
        "artifacts": {"fields": artifact_record_from_bytes("reconstruction_fields.npz", fields_payload), "contact_sheet": artifact_record_from_bytes("reconstruction_contact_sheet.png", contact_payload)},
        "gates": {
            "production_quality_accepted": True, "exactly_8_samples_per_family": all(sum(item["family_id"] == family for item in samples) == 8 for family in range(5)),
            "all_samples_from_validation_split": set(map(int, indices)) <= set(map(int, split.validation)), "all_projected_tuples_legal": True,
            "native_48px_fields": all(value.shape == (40, 48, 48) for value in predicted.values()), "ema_checkpoint_used": True,
            "deterministic_cpu_compile": True, "python_runtime_not_required_for_artifacts": True,
        },
    }
    manifest["semantic_sha256"] = json_sha256(manifest); files["showcase_manifest.json"] = canonical_json_bytes(manifest)
    return files, manifest


def _publish(destination: Path, files: Mapping[str, bytes]) -> None:
    destination = Path(destination).resolve()
    if destination.exists(): raise FileExistsError(destination)
    require_disk_floor(destination.parent, floor_gb=100.0, planned_bytes=sum(map(len, files.values())) + 64 * 1024**2)
    destination.parent.mkdir(parents=True, exist_ok=True); staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"; staging.mkdir()
    try:
        for relative, payload in files.items():
            target = staging.joinpath(*PurePosixPath(relative).parts); descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=staging)
            with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            for path in staging.iterdir(): path.unlink(missing_ok=True)
            staging.rmdir()
        raise


def validate_showcase(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve(); raw = manifest_path.read_bytes(); manifest = json.loads(raw)
    errors = sorted(Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).iter_errors(manifest), key=lambda error: list(error.absolute_path))
    if errors: raise ValueError(f"Sprite latent showcase schema failed: {errors[0].message}")
    if raw != canonical_json_bytes(manifest): raise ValueError("Sprite latent showcase manifest is not canonical JSON")
    if manifest["semantic_sha256"] != json_sha256({key: value for key, value in manifest.items() if key != "semantic_sha256"}): raise ValueError("Sprite latent showcase semantic hash differs")
    production = PROJECT_ROOT.joinpath(*PurePosixPath(manifest["source"]["production_manifest"]).parts)
    corpus = PROJECT_ROOT.joinpath(*PurePosixPath(manifest["source"]["corpus"]).parts)
    expected_files, expected_manifest = _compile(production, corpus)
    if manifest != expected_manifest: raise ValueError("Sprite latent showcase semantic replay differs")
    root = manifest_path.parent
    for relative, payload in expected_files.items():
        path = root / relative
        if not path.is_file() or path.read_bytes() != payload: raise ValueError(f"Sprite latent showcase byte replay differs: {relative}")
    actual = {path.relative_to(root).as_posix() for path in root.iterdir() if path.is_file()}
    if actual != set(expected_files): raise ValueError("Sprite latent showcase output closure differs")
    return {"passed": True, "sample_count": 40, "semantic_sha256": manifest["semantic_sha256"], "manifest_sha256": sha256_file(manifest_path), "contact_sheet_sha256": manifest["artifacts"]["contact_sheet"]["sha256"]}


def build_showcase(production: Path = DEFAULT_PRODUCTION, corpus: Path = DEFAULT_CORPUS, destination: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    files, manifest = _compile(production, corpus)
    if not all(manifest["gates"].values()): raise ValueError("Sprite latent showcase gate failed")
    _publish(destination, files)
    return validate_showcase(Path(destination) / "showcase_manifest.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile balanced visual evidence from an accepted sprite FSQ checkpoint")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build"); build.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION); build.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS); build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate = sub.add_parser("validate"); validate.add_argument("manifest", type=Path)
    args = parser.parse_args(); result = build_showcase(args.production, args.corpus, args.output) if args.command == "build" else validate_showcase(args.manifest)
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
