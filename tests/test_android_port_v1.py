from forge.android_port_v1.contract import TARGET
from forge.android_port_v1.export import export_mobile_bundle


def test_android_target_contract() -> None:
    assert TARGET["device"] == "Samsung Galaxy S25 Ultra"
    assert TARGET["abi"] == "arm64-v8a"
    assert TARGET["target_display_fps"] == 30
    assert TARGET["organism_motion_hz"] >= 12
    assert TARGET["execution_provider_order"][:2] == ["QNN-HTP", "NNAPI"]


def test_android_export_owns_latent_normalization_boundary() -> None:
    names = export_mobile_bundle.__code__.co_names
    assert "load_sequences" in names
    assert "tofile" in names
