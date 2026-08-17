from __future__ import annotations

import numpy as np
import torch

from forge.creature_stage_developmental import develop
from forge.nature_sim_v2 import founder_genomes
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


def test_live_organism_coordinates_match_centered_training_raster() -> None:
    organism=develop(founder_genomes(variants_per_family=1)[0].developmental)
    features,mask=ContinuousCellVAERuntime.organism_features(organism,organism.cell_xy)
    raster=(features[mask,:2].numpy()+1)*47*.5
    assert np.allclose((raster.min(0)+raster.max(0))*.5,(23.5,23.5),atol=1e-5)
