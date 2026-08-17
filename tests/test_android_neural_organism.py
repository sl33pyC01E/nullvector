from pathlib import Path

import numpy as np
import pytest

from forge.android_port_v1.organism import (
    DEFAULT_OUTPUT,
    cellular_static_to_vae_features,
    validate_neural_organism,
)


def _static() -> np.ndarray:
    value = np.zeros((85, 48, 48), dtype=np.float32)
    value[0, 18:30, 19:29] = 1
    value[14, 18:30, 19:29] = 1
    value[27, 18:30, 19:29] = 1
    value[52:60, 18:30, 19:29] = .5
    value[66:73, 18:30, 19:29] = .6
    return value


def test_cellular_bridge_preserves_geometry_family_and_bounds() -> None:
    features, mask = cellular_static_to_vae_features(_static())
    assert features.shape == (1, 576, 52)
    assert mask.shape == (1, 576)
    assert mask.dtype == features.dtype == np.float32
    assert int(mask.sum()) == 120
    assert np.all(features[0, :120, 21] == 1)  # machine family slot
    assert np.all(features[0, :120, 51] == 1)
    assert np.all((features[0, :120, :2] >= -1) & (features[0, :120, :2] <= 1))


def test_cellular_bridge_rejects_oversize_and_bad_dtype() -> None:
    oversized = np.zeros((85, 48, 48), dtype=np.float32); oversized[0] = 1
    with pytest.raises(ValueError, match="cell count"):
        cellular_static_to_vae_features(oversized)
    with pytest.raises(ValueError, match="static tensor"):
        cellular_static_to_vae_features(_static().astype(np.float64))


@pytest.mark.skipif(not (DEFAULT_OUTPUT / "manifest.json").is_file(), reason="local exported Android organism bank absent")
def test_local_android_organism_export_is_closed_and_ready() -> None:
    report = validate_neural_organism(Path(DEFAULT_OUTPUT))
    assert report["passed"] is True
    assert report["family"] == "machine"
    assert report["cell_count"] == 458
