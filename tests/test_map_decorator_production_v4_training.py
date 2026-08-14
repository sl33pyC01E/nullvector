from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest
import torch

from forge.map_decorator.hashing import json_sha256
from forge.map_decorator_ml.contract import HEAD_NAMES, ModelConfig
from forge.map_decorator_ml.legality import TorchLegalMasks
from forge.map_decorator_production_v2.training import WarmStartEMA
from forge.map_decorator_production_v4.contract import ProposalLocatorConfig
from forge.map_decorator_production_v4.decoding import select_proposal_conditioned_argmax
from forge.map_decorator_production_v4.model import ProposalConditionedDecoratorV4
from forge.map_decorator_production_v4.proposal import ProposalAuthority
from forge.map_decorator_production_v4_training.contract import (
    ResidualLossConfig,
    ResidualTrainingConfig,
    V4_TRAINING_CONTRACT_SHA256,
    v4_training_contract_manifest,
)
from forge.map_decorator_production_v4_training.dataset import (
    ProposalTeacherSample,
    collate_proposal_samples,
)
from forge.map_decorator_production_v4_training.smoke import REPORT_NAME, run_smoke, validate_smoke
from forge.map_decorator_production_v4_training.training import make_optimizer, train_batch


CORPUS = Path("outputs/map_decorator_corpus_v1")
INDEX = Path("outputs/map_decorator_production_v2/foreground_index_v2")
CORE = ModelConfig(base_channels=4, condition_channels=8)
LOCATOR = ProposalLocatorConfig(locator_channels=4, locator_blocks=1, count_hidden_channels=4)
TRAINING = ResidualTrainingConfig(ema_decay=0.9, full_mask_stride=1)


def _sample_batch():
    authority = ProposalAuthority.load(CORPUS, INDEX)
    teacher, proposals = authority.sample_and_proposals(authority.authority.corpus.refs_by_split["test"][0])
    return teacher, proposals, collate_proposal_samples([ProposalTeacherSample(teacher, proposals)])


def test_v4_training_contract_preserves_public_proposal_authority() -> None:
    manifest = v4_training_contract_manifest()
    assert V4_TRAINING_CONTRACT_SHA256 == json_sha256(manifest)
    assert manifest["authority"] == {
        "proposal_fields_are_immutable_inputs": True,
        "target_fields_never_generate_proposals": True,
        "off_proposal_object_decode_impossible": True,
        "procedural_baseline_reported_separately": True,
    }
    with pytest.raises(ValueError, match="cannot be disabled"):
        ResidualLossConfig(proposal_presence_weight=0)


def test_v4_train_step_is_finite_and_decode_cannot_escape_proposals() -> None:
    teacher, proposals, batch = _sample_batch()
    torch.manual_seed(0x44D3C011)
    model = ProposalConditionedDecoratorV4(CORE, LOCATOR)
    optimizer = make_optimizer(model, TRAINING)
    ema = WarmStartEMA(model, TRAINING.ema_decay)
    generator = torch.Generator().manual_seed(TRAINING.seed)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    metrics = train_batch(
        model,
        optimizer,
        ema,
        batch,
        generator=generator,
        training_config=TRAINING,
    )
    assert metrics["loss"]["total"] > 0
    assert ema.updates == 1
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
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
            batch["proposals"],
        )
        prediction = select_proposal_conditioned_argmax(
            output,
            TorchLegalMasks(hard_empty=batch["hard_empty"], **batch["legal_masks"]),
        )
    for head in ("decal", "prop"):
        selected = prediction[head]
        selected_proposal = batch["proposals"][head].gather(
            1, (selected - 1).clamp(min=0).unsqueeze(1)
        ).squeeze(1)
        assert not bool(((selected != 0) & ~selected_proposal).any())
        assert not bool((selected[batch["hard_empty"]] != 0).any())
    assert not bool(((prediction["decal"] != 0) & (prediction["prop"] != 0)).any())
    assert teacher.sample_identity_sha256
    assert proposals.fields_sha256


@pytest.fixture(scope="module")
def smoke_bank(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("v4-residual-smoke") / "bank"
    report = run_smoke(CORPUS, INDEX, output)
    assert report["gates"]["interrupted_resume_exact"] is True
    return output


def test_v4_residual_smoke_exact_replay(smoke_bank: Path) -> None:
    report = validate_smoke(smoke_bank, corpus_root=CORPUS, index_root=INDEX)
    assert report["runtime"] == {"device": "cpu", "cuda_initialized": False, "threads": 1}
    assert report["gates"]["not_a_quality_claim"] is True
    assert len(report["history"]) == 2


def test_v4_residual_smoke_rejects_fully_rehashed_metric_tamper(smoke_bank: Path) -> None:
    report_path = smoke_bank / REPORT_NAME
    original = report_path.read_bytes()
    report = json.loads(original)
    try:
        report["history"][1]["loss"]["total"] += 0.01
        unsigned = dict(report)
        unsigned.pop("report_sha256")
        report["report_sha256"] = json_sha256(unsigned)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="semantic replay"):
            validate_smoke(smoke_bank, corpus_root=CORPUS, index_root=INDEX)
    finally:
        report_path.write_bytes(original)


def test_v4_residual_smoke_rejects_checkpoint_sidecar_tamper(smoke_bank: Path, tmp_path: Path) -> None:
    copied = tmp_path / "copied"
    shutil.copytree(smoke_bank, copied)
    sidecar_path = copied / "checkpoint_step_0001.pt.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["global_step"] = 2
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar self-hash"):
        validate_smoke(copied, corpus_root=CORPUS, index_root=INDEX)


def test_v4_residual_smoke_fails_closed_if_cuda_was_initialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="refuses CUDA initialization"):
        run_smoke(CORPUS, INDEX, tmp_path / "forbidden")
