from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from forge.map_decorator.hashing import json_sha256
from forge.map_decorator_ml.contract import HEAD_NAMES, ModelConfig
from forge.map_decorator_ml.dataset import collate_teacher_samples
from forge.map_decorator_ml.legality import TorchLegalMasks
from forge.map_decorator_production_v4.audit import _read_hashed_json, audit_chunk
from forge.map_decorator_production_v4.contract import (
    ProposalLocatorConfig,
    V4_CONTRACT_SHA256,
    v4_contract_manifest,
)
from forge.map_decorator_production_v4.decoding import select_proposal_conditioned_argmax
from forge.map_decorator_production_v4.model import ProposalConditionedDecoratorV4
from forge.map_decorator_production_v4.proposal import (
    ProposalAuthority,
    assert_vectorized_hash_exact,
    audit_proposal_targets,
)
from forge.map_decorator_production_v4.smoke import REPORT_NAME, build_smoke, validate_smoke
from forge.maps.model import THEMES


CORPUS = Path("outputs/map_decorator_corpus_v1")
INDEX = Path("outputs/map_decorator_production_v2/foreground_index_v2")


def test_v4_contract_and_vectorized_public_hash_are_exact() -> None:
    assert V4_CONTRACT_SHA256 == json_sha256(v4_contract_manifest())
    assert_vectorized_hash_exact()
    manifest = v4_contract_manifest()
    assert manifest["proposal_channels"]["semantics"]["target_fields_read"] is False
    assert manifest["proposal_channels"]["semantics"]["inference_available"] is True
    assert manifest["locator"]["maximum_objects_per_head"] == 4096
    with pytest.raises(ValueError, match="quota"):
        ProposalLocatorConfig(maximum_objects_per_head=4097)


def test_v4_public_proposals_cover_every_theme_without_target_leakage() -> None:
    authority = ProposalAuthority.load(CORPUS, INDEX)
    observed: set[str] = set()
    for ref in authority.authority.corpus.refs_by_split["test"]:
        sample, proposals = authority.sample_and_proposals(ref)
        theme = THEMES[sample.theme_index]
        if theme in observed:
            continue
        audit = audit_proposal_targets(proposals, sample.targets)
        assert audit["passed"]
        assert all(audit["heads"][head]["recall"] == 1.0 for head in ("decal", "prop"))
        observed.add(theme)
    assert observed == set(THEMES)


def test_untrained_v4_residual_decodes_only_legal_proposed_cells() -> None:
    authority = ProposalAuthority.load(CORPUS, INDEX)
    sample, proposals = authority.sample_and_proposals(authority.authority.corpus.refs_by_split["test"][0])
    batch = collate_teacher_samples([sample])
    torch.manual_seed(41)
    model = ProposalConditionedDecoratorV4(
        ModelConfig(base_channels=4, condition_channels=8),
        ProposalLocatorConfig(locator_channels=4, locator_blocks=1, count_hidden_channels=4),
    ).eval()
    valid = batch["valid_cells"]
    masked = {name: valid.clone() for name in HEAD_NAMES}
    proposal_tensors = {
        "decal": torch.from_numpy(proposals.decal.copy())[None],
        "prop": torch.from_numpy(proposals.prop.copy())[None],
    }
    with torch.inference_mode():
        output = model(
            batch["features"],
            batch["targets"],
            masked,
            batch["theme_index"],
            batch["global_conditions"],
            torch.ones((1,), dtype=torch.float32),
            proposal_tensors,
        )
        prediction = select_proposal_conditioned_argmax(
            output,
            TorchLegalMasks(hard_empty=batch["hard_empty"], **batch["legal_masks"]),
        )
    for head in ("decal", "prop"):
        selected = prediction[head]
        proposed = proposal_tensors[head].gather(
            1, (selected - 1).clamp(min=0).unsqueeze(1)
        ).squeeze(1)
        assert not bool(((selected != 0) & ~proposed).any())
        assert not bool((selected[batch["hard_empty"]] != 0).any())
    assert not bool(((prediction["decal"] != 0) & (prediction["prop"] != 0)).any())


@pytest.fixture(scope="module")
def smoke_bank(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("map-decorator-v4-smoke") / "bank"
    result = build_smoke(CORPUS, INDEX, output, visually_inspected=True)
    assert result["passed"]
    return output


def test_v4_smoke_is_exact_and_exceeds_sparse_object_gates(smoke_bank: Path) -> None:
    result = validate_smoke(smoke_bank, corpus_root=CORPUS, index_root=INDEX)
    assert result["decal_foreground_macro_iou"] >= 0.90
    assert result["prop_foreground_macro_iou"] >= 0.90
    report = json.loads((smoke_bank / REPORT_NAME).read_text(encoding="utf-8"))
    assert report["counts"]["sample_count"] == 24
    assert report["model"]["trained_steps"] == 0
    assert all(report["gates"].values())


def test_v4_smoke_rejects_fully_rehashed_metric_tamper(smoke_bank: Path) -> None:
    report_path = smoke_bank / REPORT_NAME
    value = json.loads(report_path.read_text(encoding="utf-8"))
    original = report_path.read_bytes()
    try:
        value["metrics"]["heads"]["decal"]["foreground_macro_iou"] -= 0.01
        unsigned = dict(value)
        unsigned.pop("report_sha256")
        value["report_sha256"] = json_sha256(unsigned)
        report_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="semantic replay"):
            validate_smoke(smoke_bank, corpus_root=CORPUS, index_root=INDEX)
    finally:
        report_path.write_bytes(original)


def test_v4_audit_chunk_is_source_bound_and_has_zero_missing_targets(tmp_path: Path) -> None:
    path = tmp_path / "chunk.json"
    report = audit_chunk(CORPUS, INDEX, path, chunk_index=0)
    assert report == _read_hashed_json(path)
    assert report["sample_count"] > 0
    assert report["gates"]["zero_missing_target_cells"] is True
    assert all(
        report["counts"][split][f"{head}_missing"] == 0
        for split in ("train", "validation", "test")
        for head in ("decal", "prop")
    )
