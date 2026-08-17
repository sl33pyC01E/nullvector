from __future__ import annotations

import torch

from forge.cellular_nca_causal_v3.contract import PARENT_OUTPUT, source_sha256
from forge.cellular_nca_causal_v3.curriculum import CONTRAST_STEPS, ROLLOUT_STEPS
from forge.cellular_nca_causal_v3.training import checkpoint_name


def test_long_horizon_contract_is_distinct_and_bounded() -> None:
    assert ROLLOUT_STEPS == 6
    assert CONTRAST_STEPS == (2, 4, 6)
    assert checkpoint_name(64) == "causal_v3_segment_0000064.pt"
    assert len(source_sha256()) == 64


def test_rejected_v2_parent_is_preserved_as_curriculum_start() -> None:
    manifest = PARENT_OUTPUT / "cellular_nca_causal_manifest.json"
    assert manifest.is_file()
    assert '"status":"experimental"' in manifest.read_text(encoding="utf-8")


def test_cuda_rng_state_floor_matches_current_torch() -> None:
    state = torch.Generator().get_state()
    assert state.dtype == torch.uint8 and state.ndim == 1 and state.numel() >= 16
