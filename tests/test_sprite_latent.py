from __future__ import annotations

import json
import numpy as np
import pytest
import shutil
import torch
from pathlib import Path
import zipfile

from forge.morphology import allowed_training_field_tuples, genome_from_seed, render_specimen
from forge.sprite_latent import (
    FSQQuantizer,
    SemanticSpriteFSQ,
    SpriteLatentConfig,
    project_legal_tuples,
    sprite_codec_loss,
)
from forge.sprite_latent.training import canonical_state_hash, exact_reconstruction_metrics, training_step
from forge.sprite_latent.artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from forge.sprite_latent.smoke import _resolve_artifact, run_cpu_smoke, validate_smoke_output
from forge.sprite_latent.corpus import (
    SemanticFieldCorpus,
    compute_legal_tuples,
    legal_tuple_fingerprint,
    stratified_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CORPUS = PROJECT_ROOT / "data" / "morphology_32768_4d4f5250.npz"


def _reseal_manifest(path: Path, payload: dict[str, object]) -> None:
    base = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    payload["manifest_sha256"] = sha256_bytes(canonical_json_bytes(base))
    path.write_bytes(canonical_json_bytes(payload))


@pytest.fixture(scope="module")
def adversarial_smoke(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("sprite-latent-adversarial") / "smoke"
    run_cpu_smoke(destination, continuous_steps=1, quantized_steps=1)
    return destination


def _batch(count: int = 5) -> dict[str, torch.Tensor]:
    fields = [render_specimen(genome_from_seed(1000 + index, index % 5)).training_fields() for index in range(count)]
    return {
        "part": torch.from_numpy(np.stack([item.part_owner for item in fields])).long(),
        "material": torch.from_numpy(np.stack([item.material for item in fields])).long(),
        "emission": torch.from_numpy(np.stack([item.emission_level for item in fields])).long(),
        "morphology": torch.tensor([item.morphology_index for item in fields]),
        "subtype": torch.tensor([item.subtype_id for item in fields]),
        "role": torch.tensor([item.role_id for item in fields]),
        "genes": torch.from_numpy(np.stack([item.genes for item in fields])),
    }


def _legal() -> torch.Tensor:
    return torch.tensor(sorted(allowed_training_field_tuples()), dtype=torch.long)


def test_fsq_mixed_radix_is_exact_and_roundtrips() -> None:
    quantizer = FSQQuantizer((8, 5, 5, 5))
    codes = torch.arange(quantizer.code_count)
    digits = quantizer.codes_to_digits(codes)
    assert torch.equal(quantizer.digits_to_codes(digits), codes)
    assert digits.shape == (1000, 4)
    assert torch.all(quantizer.digits_to_normalized(digits) >= -1.0)
    assert torch.all(quantizer.digits_to_normalized(digits) <= 1.0)


def test_fsq_continuous_warmup_and_quantized_shapes() -> None:
    quantizer = FSQQuantizer((4, 4, 4))
    values = torch.randn(2, 3, 12, 12)
    warmup = quantizer(values, quantize=False)
    discrete = quantizer(values, quantize=True)
    assert warmup["quantized"].shape == values.shape
    assert discrete["codes"].shape == (2, 12, 12)
    assert int(discrete["codes"].min()) >= 0
    assert int(discrete["codes"].max()) < 64
    assert not torch.equal(warmup["quantized"], discrete["quantized"])


def test_codec_reconstructs_native_aligned_fields_and_legal_projection() -> None:
    config = SpriteLatentConfig(width=16, latent_levels=(4, 4, 4), residual_depth=1, condition_dim=32)
    model = SemanticSpriteFSQ(config)
    batch = _batch()
    output = model(**batch, quantize=True)
    assert output.part_logits.shape == (5, 17, 48, 48)
    assert output.material_logits.shape == (5, 10, 48, 48)
    assert output.emission_logits.shape == (5, 4, 48, 48)
    assert output.codes.shape == (5, 12, 12)
    projected = project_legal_tuples(output, _legal())
    triples = torch.stack((projected["part"], projected["material"], projected["emission"]), dim=-1)
    observed = {tuple(map(int, row)) for row in triples.reshape(-1, 3).tolist()}
    legal = {tuple(map(int, row)) for row in _legal().tolist()}
    assert observed <= legal


def test_codec_loss_and_training_step_are_finite() -> None:
    torch.manual_seed(123)
    config = SpriteLatentConfig(width=16, latent_levels=(4, 4, 4), residual_depth=1, condition_dim=32)
    model = SemanticSpriteFSQ(config)
    batch = _batch()
    output = model(**batch, quantize=False)
    loss, pieces = sprite_codec_loss(
        output,
        batch["part"],
        batch["material"],
        batch["emission"],
        _legal(),
        config=config,
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in pieces.values())
    before = canonical_state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = training_step(model, batch, _legal(), optimizer, quantize=False)
    assert all(np.isfinite(value) for value in metrics.values())
    assert canonical_state_hash(model) != before
    reconstruction = exact_reconstruction_metrics(model, batch, _legal())
    assert reconstruction["legal_projection_fraction"] == 1.0


def test_soft_fsq_usage_loss_has_nonzero_encoder_gradient() -> None:
    torch.manual_seed(0x465351)
    config = SpriteLatentConfig(
        width=16,
        latent_levels=(4, 4, 4),
        residual_depth=1,
        condition_dim=32,
        latent_usage_weight=0.07,
    )
    model = SemanticSpriteFSQ(config)
    output = model(**_batch(5), quantize=True)
    assert output.soft_marginal_entropy.requires_grad
    assert not output.marginal_entropy.requires_grad
    usage_loss = config.latent_usage_weight * (1.0 - output.soft_marginal_entropy)
    gradient = torch.autograd.grad(
        usage_loss,
        model.to_latent.weight,
        retain_graph=False,
        allow_unused=False,
    )[0]
    assert bool(torch.isfinite(gradient).all())
    assert int(torch.count_nonzero(gradient)) > 0
    assert float(gradient.abs().sum()) > 0.0


def test_legal_tuple_contract_rejects_duplicates_and_unsorted_rows() -> None:
    config = SpriteLatentConfig(width=16, latent_levels=(4, 4, 4), residual_depth=1, condition_dim=32)
    model = SemanticSpriteFSQ(config)
    output = model(**_batch(2), quantize=True)
    legal = _legal()
    with pytest.raises(ValueError, match="duplicates"):
        project_legal_tuples(output, torch.cat((legal, legal[-1:])))
    reversed_table = legal.flip(0)
    with pytest.raises(ValueError, match="sorted"):
        project_legal_tuples(output, reversed_table)


def test_configuration_metadata_declares_representation_only_latent_geometry() -> None:
    config = SpriteLatentConfig()
    metadata = config.metadata()
    assert metadata["latent_grid_size"] == 12
    assert metadata["implicit_code_count"] == 1000
    assert metadata["quantizer"] == "finite-scalar-quantization-sigmoid-ste-v1"


def test_cpu_smoke_is_immutable_hash_bound_and_safe(tmp_path) -> None:
    destination = tmp_path / "smoke"
    manifest = run_cpu_smoke(destination, continuous_steps=1, quantized_steps=1)
    assert manifest["status"] == "passed"
    assert manifest["production_quality_claimed"] is False
    assert manifest["generative_prior_present"] is False
    assert all(manifest["gates"].values())
    assert validate_smoke_output(destination / "smoke_manifest.json") == manifest
    repeat = tmp_path / "repeat"
    repeated = run_cpu_smoke(repeat, continuous_steps=1, quantized_steps=1)
    assert repeated == manifest
    for name in ("smoke_checkpoint.pt", "reconstruction_contact_sheet.png", "smoke_manifest.json"):
        assert (destination / name).read_bytes() == (repeat / name).read_bytes()
    with pytest.raises(FileExistsError):
        run_cpu_smoke(destination, continuous_steps=1, quantized_steps=1)


def test_smoke_manifest_and_checkpoint_reject_resealed_tampering(
    adversarial_smoke: Path,
    tmp_path: Path,
) -> None:
    manifest_case = tmp_path / "manifest-case"
    shutil.copytree(adversarial_smoke, manifest_case)
    manifest_path = manifest_case / "smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training"]["history"][0]["loss"] += 0.125
    _reseal_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="exact semantic replay mismatch: training"):
        validate_smoke_output(manifest_path)

    checkpoint_case = tmp_path / "checkpoint-case"
    shutil.copytree(adversarial_smoke, checkpoint_case)
    checkpoint_path = checkpoint_case / "smoke_checkpoint.pt"
    checkpoint_bytes = bytearray(checkpoint_path.read_bytes())
    checkpoint_bytes[-1] ^= 0x01
    checkpoint_path.write_bytes(checkpoint_bytes)
    checkpoint_manifest_path = checkpoint_case / "smoke_manifest.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    checkpoint_manifest["artifacts"]["checkpoint"]["sha256"] = sha256_file(checkpoint_path)
    _reseal_manifest(checkpoint_manifest_path, checkpoint_manifest)
    with pytest.raises(ValueError, match="checkpoint does not replay byte-exactly"):
        validate_smoke_output(checkpoint_manifest_path)


@pytest.mark.parametrize(
    "candidate",
    (
        "../outside.pt",
        "nested/../../outside.pt",
        "/absolute.pt",
        "C:/drive-qualified.pt",
        "nested\\..\\outside.pt",
        "./checkpoint.pt",
    ),
)
def test_artifact_resolution_rejects_unsafe_paths(tmp_path: Path, candidate: str) -> None:
    with pytest.raises(ValueError, match="unsafe|escapes"):
        _resolve_artifact(tmp_path, {"path": candidate})


def test_corpus_loader_strictly_rejects_adversarial_inputs(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate-member.npz"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("format.npy", b"first")
            archive.writestr("format.npy", b"second")
    with pytest.raises(ValueError, match="duplicate ZIP members"):
        SemanticFieldCorpus.load(duplicate)

    with pytest.raises(ValueError, match="file SHA-256 does not match"):
        SemanticFieldCorpus.load(PRODUCTION_CORPUS, expected_file_sha256="0" * 64)


def test_field_only_production_corpus_preserves_frozen_split_and_tuple_contract() -> None:
    corpus = SemanticFieldCorpus.load(PRODUCTION_CORPUS)
    assert corpus.file_sha256 == "77dc7313ca6411295bad883f483a6edf4be75016ebfd7c107d0f286d2cb1cd7b"
    assert corpus.count == 32768
    assert corpus.image_size == 48
    assert corpus.loaded_array_bytes < 300 * 1024**2
    assert corpus.metadata()["guide_loaded"] is False
    split = stratified_split(corpus)
    assert split.fingerprint == "5e400872460dc527c01a2a301f006e761abd1621773c5f67b45568d68886007b"
    legal = compute_legal_tuples(corpus, split.training)
    assert legal.shape == (69, 3)
    assert legal_tuple_fingerprint(legal) == "0b15074b76ca69ea9a93e0b73db7e5df0b242dc0ecc46c5e842342fb0378948d"
