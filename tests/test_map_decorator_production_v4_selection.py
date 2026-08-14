from __future__ import annotations

from pathlib import Path

import pytest
import torch

from forge.map_decorator.hashing import json_sha256
from forge.map_decorator_ml.contract import HEAD_NAMES, ModelConfig
from forge.map_decorator_ml.legality import TorchLegalMasks
from forge.map_decorator_production_v4.contract import ProposalLocatorConfig
from forge.map_decorator_production_v4.decoding import select_proposal_conditioned_argmax
from forge.map_decorator_production_v4.model import ProposalConditionedDecoratorV4
from forge.map_decorator_production_v4.proposal import ProposalAuthority
from forge.map_decorator_production_v4_calibration.runner import calibration_source_sha256, validate_supervised
from forge.map_decorator_production_v4_selection.audit import (
    _derive_config,
    selection_source_manifest,
    selection_source_sha256,
)
from forge.map_decorator_production_v4_selection.contract import (
    ProtectedSelectionConfig,
    V4_SELECTION_CONTRACT_SHA256,
    selection_contract_manifest,
)
from forge.map_decorator_production_v4_selection.decoder import apply_protected_proposals
from forge.map_decorator_production_v4_training.dataset import ProposalTeacherSample, collate_proposal_samples


CORPUS = Path("outputs/map_decorator_corpus_v1")
INDEX = Path("outputs/map_decorator_production_v2/foreground_index_v2")
CALIBRATION = Path("outputs/map_decorator_production_v4_calibration/calibration_100step_v1")


def test_protected_selection_contract_is_conservative() -> None:
    manifest = selection_contract_manifest()
    assert V4_SELECTION_CONTRACT_SHA256 == json_sha256(manifest)
    assert manifest["rule"]["destination_must_be_empty"] is True
    assert manifest["rule"]["other_object_head_must_be_empty"] is True
    assert manifest["rule"]["no_new_off_proposal_cells"] is True
    with pytest.raises(ValueError, match="At least one"):
        ProtectedSelectionConfig(decal_classes=(), prop_classes=())


def test_protected_selector_restores_only_empty_legal_noncolliding_proposals() -> None:
    authority = ProposalAuthority.load(CORPUS, INDEX)
    sample, proposals = authority.sample_and_proposals(authority.authority.corpus.refs_by_split["test"][0])
    batch = collate_proposal_samples([ProposalTeacherSample(sample, proposals)])
    torch.manual_seed(9)
    model = ProposalConditionedDecoratorV4(
        ModelConfig(base_channels=4, condition_channels=8),
        ProposalLocatorConfig(locator_channels=4, locator_blocks=1, count_hidden_channels=4),
    ).eval()
    valid = batch["valid_cells"]
    masked = {name: valid.clone() for name in HEAD_NAMES}
    legal = TorchLegalMasks(hard_empty=batch["hard_empty"], **batch["legal_masks"])
    with torch.inference_mode():
        output = model(
            batch["features"], batch["targets"], masked, batch["theme_index"],
            batch["global_conditions"], torch.ones((1,)), batch["proposals"],
        )
        selected = select_proposal_conditioned_argmax(output, legal)
    selected = {name: value.clone() for name, value in selected.items()}
    candidate = batch["proposals"]["decal"][:, 1] & legal.decal[:, 2] & ~legal.hard_empty
    available = torch.nonzero(candidate & (selected["prop"] == 0), as_tuple=False)
    assert available.numel() > 0
    batch_id, y, x = map(int, available[0])
    selected["decal"][batch_id, y, x] = 0
    protected, diagnostics = apply_protected_proposals(selected, output, legal)
    assert protected["decal"][batch_id, y, x].item() == 2
    assert diagnostics["total_restored"] >= 1
    assert not bool(((protected["decal"] != 0) & (protected["prop"] != 0)).any())
    for head in ("decal", "prop"):
        proposed = output.proposals[head].gather(1, (protected[head] - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        assert not bool(((protected[head] != 0) & ~proposed).any())


def test_protected_selection_source_and_rare_class_derivation_are_frozen() -> None:
    manifest = selection_source_manifest()
    assert manifest["calibration_source_sha256"] == calibration_source_sha256()
    assert selection_source_sha256() == json_sha256(manifest)
    calibration = validate_supervised(CALIBRATION, corpus_root=CORPUS, index_root=INDEX)["calibration"]
    assert _derive_config(calibration) == ProtectedSelectionConfig(decal_classes=(2,))
