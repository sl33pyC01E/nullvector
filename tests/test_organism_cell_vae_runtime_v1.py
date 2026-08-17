from __future__ import annotations

import torch

from forge.organism_cell_vae_runtime_v1.runtime import ContinuousCellVAERuntime


def test_runtime_rejects_malformed_feature_geometry():
    runtime = object.__new__(ContinuousCellVAERuntime)
    runtime.device = torch.device("cpu")
    try:
        runtime.render_features(torch.zeros(4, 52), torch.ones(4, dtype=torch.bool))
    except ValueError as error:
        assert "geometry" in str(error)
    else:
        raise AssertionError("malformed cellular field was accepted")
