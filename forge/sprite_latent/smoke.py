from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import hashlib
import json
import platform
from pathlib import Path
from pathlib import PurePosixPath
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, __version__ as PILLOW_VERSION
import torch

from ..morphology import FAMILIES, allowed_training_field_tuples, genome_from_seed, render_specimen
from ..morphology.constants import RENDERER_VERSION
from ..morphology.corpus import corpus_source_hash
from .artifacts import (
    artifact_record,
    canonical_json_bytes,
    prepare_destination,
    sha256_bytes,
    sha256_file,
    source_hash,
    write_bytes_new,
    write_json_new,
)
from .codec import SemanticSpriteFSQ, SpriteLatentConfig, project_legal_tuples
from .schema import validate_schema
from .training import canonical_state_hash, exact_reconstruction_metrics, training_step


SMOKE_FORMAT = "nullvector-semantic-sprite-fsq-smoke-v1"
SMOKE_SEED = 0x4653515350524954
FAMILY_COLORS = (
    (53, 207, 255),
    (255, 146, 68),
    (84, 242, 148),
    (229, 83, 255),
    (102, 238, 242),
)


def _fixture() -> tuple[list[Any], dict[str, torch.Tensor]]:
    fields: list[Any] = []
    for family in range(len(FAMILIES)):
        for variation in range(4):
            genome = genome_from_seed(0x53505200 + family * 101 + variation * 17, family)
            genome = replace(genome, role_id=(family + variation * 2) % 8)
            genome.validate()
            fields.append(render_specimen(genome).training_fields())
    batch = {
        "part": torch.from_numpy(np.stack([item.part_owner for item in fields])).long(),
        "material": torch.from_numpy(np.stack([item.material for item in fields])).long(),
        "emission": torch.from_numpy(np.stack([item.emission_level for item in fields])).long(),
        "morphology": torch.tensor([item.morphology_index for item in fields]),
        "subtype": torch.tensor([item.subtype_id for item in fields]),
        "role": torch.tensor([item.role_id for item in fields]),
        "genes": torch.from_numpy(np.stack([item.genes for item in fields])),
    }
    return fields, batch


def _class_weights(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    counts = {
        "part": (batch["part"], 17),
        "material": (batch["material"], 10),
        "emission": (batch["emission"], 4),
    }
    result: dict[str, torch.Tensor] = {}
    for name, (values, count) in counts.items():
        frequency = torch.bincount(values.reshape(-1), minlength=count).float()
        weights = frequency.sum().div(frequency.clamp_min(1.0)).sqrt()
        weights = weights / weights.mean()
        result[name] = weights.clamp(0.25, 5.0)
    return result


def _rgba(part: np.ndarray, material: np.ndarray, emission: np.ndarray, family: int) -> np.ndarray:
    image = np.zeros((48, 48, 4), dtype=np.uint8)
    visible = part != 0
    base = np.asarray(FAMILY_COLORS[family], dtype=np.int16)
    shade = ((material.astype(np.int16) * 17 + part.astype(np.int16) * 7) % 61) - 30
    rgb = np.clip(base[None, None, :] + shade[..., None], 0, 255).astype(np.uint8)
    image[visible, :3] = rgb[visible]
    image[visible, 3] = 255
    hot = visible & (emission >= 2)
    image[hot, :3] = np.asarray((242, 252, 255), dtype=np.uint8)
    return image


def _contact_sheet(
    batch: Mapping[str, torch.Tensor], projected: Mapping[str, torch.Tensor]
) -> bytes:
    cell, scale = 48, 3
    pair_width = cell * scale * 2 + 14
    left = 78
    width = left + pair_width * 4 + 6
    height = 48 + (cell * scale + 24) * 5
    canvas = Image.new("RGB", (width, height), (5, 8, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "SEMANTIC SPRITE FSQ / CPU FOUNDATION SMOKE", fill=(204, 241, 255))
    draw.text((12, 23), "original -> legal tuple projection / not production quality", fill=(83, 157, 189))
    part = batch["part"].cpu().numpy()
    material = batch["material"].cpu().numpy()
    emission = batch["emission"].cpu().numpy()
    pred_part = projected["part"].cpu().numpy()
    pred_material = projected["material"].cpu().numpy()
    pred_emission = projected["emission"].cpu().numpy()
    for family, family_name in enumerate(FAMILIES):
        y = 46 + family * (cell * scale + 24)
        draw.text((4, y + 61), family_name, fill=FAMILY_COLORS[family])
        for variation in range(4):
            index = family * 4 + variation
            x = left + variation * pair_width
            original = Image.fromarray(_rgba(part[index], material[index], emission[index], family)).resize(
                (cell * scale, cell * scale), resample=Image.Resampling.NEAREST
            )
            decoded = Image.fromarray(
                _rgba(pred_part[index], pred_material[index], pred_emission[index], family)
            ).resize((cell * scale, cell * scale), resample=Image.Resampling.NEAREST)
            canvas.paste(original.convert("RGB"), (x, y))
            canvas.paste(decoded.convert("RGB"), (x + cell * scale + 6, y))
    buffer = BytesIO()
    canvas.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(b"nullvector-semantic-sprite-smoke-fixture-v1\0")
    for name in ("part", "material", "emission", "morphology", "subtype", "role", "genes"):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _legal_tuple_sha256(values: torch.Tensor) -> str:
    array = values.detach().cpu().to(torch.int64).contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(b"nullvector-semantic-sprite-smoke-legal-tuples-v1\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _runtime_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "torch": str(torch.__version__),
        "torch_build_cuda": str(torch.version.cuda),
        "numpy": str(np.__version__),
        "pillow": str(PILLOW_VERSION),
        "execution_device": "cpu",
    }


def _semantic_block_sha256(name: str, payload: Mapping[str, Any]) -> str:
    return sha256_bytes(
        name.encode("ascii") + b"\0" + canonical_json_bytes(payload)
    )


def _execute_cpu_smoke(*, continuous_steps: int, quantized_steps: int) -> dict[str, Any]:
    if continuous_steps < 1 or quantized_steps < 1:
        raise ValueError("Both warm-up and quantized smoke phases need at least one step")
    previous_threads = torch.get_num_threads()
    previous_torch_rng = torch.get_rng_state()
    previous_numpy_rng = np.random.get_state()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(SMOKE_SEED & 0x7FFFFFFF)
        np.random.seed(SMOKE_SEED & 0xFFFFFFFF)
        _, batch = _fixture()
        if any(tensor.device.type != "cpu" for tensor in batch.values()):
            raise ValueError("Sprite latent smoke fixture left the CPU")
        legal = torch.tensor(sorted(allowed_training_field_tuples()), dtype=torch.long)
        config = SpriteLatentConfig(
            width=32,
            latent_levels=(8, 5, 5, 5),
            residual_depth=1,
            condition_dim=48,
        )
        model = SemanticSpriteFSQ(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
        weights = _class_weights(batch)
        history: list[dict[str, Any]] = []
        total_steps = continuous_steps + quantized_steps
        initial_state_sha256 = canonical_state_hash(model)
        for step in range(total_steps):
            quantize = step >= continuous_steps
            model.train()
            metrics = training_step(
                model,
                batch,
                legal,
                optimizer,
                quantize=quantize,
                class_weights=weights,
                gradient_clip=1.0,
            )
            history.append(
                {
                    "step": step + 1,
                    "phase": "fsq" if quantize else "continuous_autoencoder_warmup",
                    **{name: round(value, 9) for name, value in metrics.items()},
                }
            )
        model.eval()
        with torch.no_grad():
            codec_output = model(**batch, quantize=True)
            projected = project_legal_tuples(codec_output, legal)
        metrics = exact_reconstruction_metrics(model, batch, legal)
        final_state_sha256 = canonical_state_hash(model)
        source = {
            "source_sha256": source_hash(),
            "morphology_corpus_source_sha256": corpus_source_hash(),
            "morphology_renderer_version": RENDERER_VERSION,
            "fixture": "20 deterministic procedural semantic specimens / 4 per family",
            "fixture_sha256": _tensor_mapping_sha256(batch),
            "legal_tuple_sha256": _legal_tuple_sha256(legal),
            "legal_tuple_count": int(len(legal)),
            "runtime": _runtime_metadata(),
            "research_basis": [
                "https://arxiv.org/abs/1711.00937",
                "https://arxiv.org/abs/2309.15505",
                "https://arxiv.org/abs/2605.06870",
            ],
        }
        training = {
            "seed": SMOKE_SEED,
            "continuous_warmup_steps": continuous_steps,
            "quantized_steps": quantized_steps,
            "total_steps": total_steps,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": final_state_sha256,
            "history": history,
        }
        reconstruction = {
            **{name: round(value, 9) for name, value in metrics.items()},
            "minimum_code": int(codec_output.codes.min()),
            "maximum_code": int(codec_output.codes.max()),
            "unique_code_count": int(torch.unique(codec_output.codes).numel()),
        }
        gates = {
            "cpu_only": all(parameter.device.type == "cpu" for parameter in model.parameters())
            and all(tensor.device.type == "cpu" for tensor in batch.values()),
            "cuda_not_initialized": not torch.cuda.is_initialized(),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "finite_losses_and_gradients": all(
                np.isfinite(value)
                for row in history
                for name, value in row.items()
                if name not in {"step", "phase"}
            ),
            "continuous_warmup_exercised": any(
                row["phase"] == "continuous_autoencoder_warmup" for row in history
            ),
            "fsq_phase_exercised": any(row["phase"] == "fsq" for row in history),
            "state_changed": initial_state_sha256 != final_state_sha256,
            "all_projected_tuples_legal": metrics["legal_projection_fraction"] == 1.0,
            "all_five_families_present": sorted(set(map(int, batch["morphology"].tolist())))
            == list(range(5)),
            "native_48px_reconstruction": tuple(projected["part"].shape[-2:]) == (48, 48),
            "implicit_code_indices_bounded": int(codec_output.codes.min()) >= 0
            and int(codec_output.codes.max()) < config.implicit_code_count,
            "soft_usage_regularizer_has_gradient": bool(
                codec_output.soft_marginal_entropy.requires_grad is False
            ),
            "production_quality_not_claimed": True,
        }
        # The evaluation output is under no_grad, so prove the differentiable
        # proxy on a training-mode forward separately without stepping state.
        probe = model(**batch, quantize=True)
        probe_gradient = torch.autograd.grad(
            1.0 - probe.soft_marginal_entropy,
            model.to_latent.weight,
            retain_graph=False,
            allow_unused=False,
        )[0]
        gates["soft_usage_regularizer_has_gradient"] = bool(
            torch.isfinite(probe_gradient).all() and torch.count_nonzero(probe_gradient) > 0
        )
        if not all(gates.values()):
            raise ValueError("Sprite latent smoke gate failure")
        checkpoint_payload = {
            "format": "nullvector-semantic-sprite-fsq-checkpoint-v2",
            "config": config.metadata(),
            "model": model.state_dict(),
            "state_sha256": final_state_sha256,
            "seed": SMOKE_SEED,
            "source_sha256": source["source_sha256"],
            "morphology_corpus_source_sha256": source["morphology_corpus_source_sha256"],
            "fixture_sha256": source["fixture_sha256"],
            "legal_tuple_sha256": source["legal_tuple_sha256"],
            "training_sha256": _semantic_block_sha256("training", training),
            "reconstruction_sha256": _semantic_block_sha256("reconstruction", reconstruction),
            "gates_sha256": _semantic_block_sha256("gates", gates),
        }
        checkpoint_buffer = BytesIO()
        torch.save(checkpoint_payload, checkpoint_buffer)
        return {
            "source": source,
            "config": config.metadata(),
            "training": training,
            "reconstruction": reconstruction,
            "gates": gates,
            "checkpoint_payload": checkpoint_payload,
            "checkpoint_bytes": checkpoint_buffer.getvalue(),
            "contact_bytes": _contact_sheet(batch, projected),
        }
    finally:
        torch.set_rng_state(previous_torch_rng)
        np.random.set_state(previous_numpy_rng)
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)


def run_cpu_smoke(
    destination: Path,
    *,
    continuous_steps: int = 12,
    quantized_steps: int = 36,
) -> dict[str, Any]:
    execution = _execute_cpu_smoke(
        continuous_steps=continuous_steps,
        quantized_steps=quantized_steps,
    )
    output_root = prepare_destination(destination)
    checkpoint_path = output_root / "smoke_checkpoint.pt"
    contact_path = output_root / "reconstruction_contact_sheet.png"
    write_bytes_new(checkpoint_path, execution["checkpoint_bytes"])
    write_bytes_new(contact_path, execution["contact_bytes"])
    manifest = {
        "format": SMOKE_FORMAT,
        "status": "passed",
        "scope": "cpu-foundation-smoke-not-production",
        "neural_reconstruction_output": True,
        "production_quality_claimed": False,
        "generative_prior_present": False,
        "source": execution["source"],
        "config": execution["config"],
        "training": execution["training"],
        "reconstruction": execution["reconstruction"],
        "artifacts": {
            "checkpoint": artifact_record(checkpoint_path, output_root),
            "contact_sheet": artifact_record(contact_path, output_root),
        },
        "gates": execution["gates"],
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    validate_schema(manifest)
    write_json_new(output_root / "smoke_manifest.json", manifest)
    return manifest


def _resolve_artifact(root: Path, record: Mapping[str, Any]) -> Path:
    raw_path = str(record["path"])
    raw_parts = raw_path.split("/")
    relative = PurePosixPath(raw_path)
    if (
        "\\" in raw_path
        or relative.is_absolute()
        or not raw_parts
        or any(part in {"", ".", ".."} or ":" in part for part in raw_parts)
    ):
        raise ValueError("Sprite latent smoke artifact path is unsafe")
    root = root.resolve()
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("Sprite latent smoke artifact path crosses a symlink")
    target = cursor.resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("Sprite latent smoke artifact escapes its output root") from error
    return target


def validate_smoke_output(manifest_path: Path) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("Sprite latent smoke manifest must be a regular non-symlink file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if canonical_json_bytes(payload) != path.read_bytes():
        raise ValueError("Sprite latent smoke manifest is not canonical JSON")
    validate_schema(payload)
    base = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload["manifest_sha256"] != sha256_bytes(canonical_json_bytes(base)):
        raise ValueError("Sprite latent smoke semantic manifest hash mismatch")
    expected_source = {
        "source_sha256": source_hash(),
        "morphology_corpus_source_sha256": corpus_source_hash(),
        "morphology_renderer_version": RENDERER_VERSION,
        "runtime": _runtime_metadata(),
    }
    for name, expected in expected_source.items():
        if payload["source"][name] != expected:
            raise ValueError(f"Sprite latent smoke source provenance drifted: {name}")
    artifacts: dict[str, Path] = {}
    for name, record in payload["artifacts"].items():
        target = _resolve_artifact(path.parent, record)
        if (
            not target.is_file()
            or target.stat().st_size != record["bytes"]
            or sha256_file(target) != record["sha256"]
        ):
            raise ValueError(f"Sprite latent smoke artifact mismatch: {name}")
        artifacts[name] = target

    execution = _execute_cpu_smoke(
        continuous_steps=payload["training"]["continuous_warmup_steps"],
        quantized_steps=payload["training"]["quantized_steps"],
    )
    for name in ("source", "config", "training", "reconstruction", "gates"):
        if payload[name] != execution[name]:
            raise ValueError(f"Sprite latent smoke exact semantic replay mismatch: {name}")
    checkpoint_bytes = artifacts["checkpoint"].read_bytes()
    contact_bytes = artifacts["contact_sheet"].read_bytes()
    if checkpoint_bytes != execution["checkpoint_bytes"]:
        raise ValueError("Sprite latent smoke checkpoint does not replay byte-exactly")
    if contact_bytes != execution["contact_bytes"]:
        raise ValueError("Sprite latent smoke contact sheet does not replay byte-exactly")

    checkpoint = torch.load(
        artifacts["checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    expected_checkpoint = execution["checkpoint_payload"]
    expected_keys = set(expected_checkpoint)
    if not isinstance(checkpoint, dict) or set(checkpoint) != expected_keys:
        raise ValueError("Sprite latent smoke checkpoint key contract drifted")
    for name in expected_keys - {"model"}:
        if checkpoint[name] != expected_checkpoint[name]:
            raise ValueError(f"Sprite latent smoke checkpoint metadata mismatch: {name}")
    config_values = dict(payload["config"])
    for derived in (
        "latent_dim",
        "implicit_code_count",
        "latent_grid_size",
        "quantizer",
        "usage_regularizer",
    ):
        config_values.pop(derived)
    config_values["latent_levels"] = tuple(config_values["latent_levels"])
    model = SemanticSpriteFSQ(SpriteLatentConfig(**config_values))
    model.load_state_dict(checkpoint["model"], strict=True)
    if canonical_state_hash(model) != payload["training"]["final_state_sha256"]:
        raise ValueError("Sprite latent checkpoint state hash mismatch")
    return payload
