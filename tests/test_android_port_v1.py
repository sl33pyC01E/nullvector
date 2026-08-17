from forge.android_port_v1.contract import TARGET


def test_android_target_contract() -> None:
    assert TARGET["device"] == "Samsung Galaxy S25 Ultra"
    assert TARGET["abi"] == "arm64-v8a"
    assert TARGET["target_display_fps"] == 30
    assert TARGET["organism_motion_hz"] >= 12
    assert TARGET["execution_provider_order"][:2] == ["QNN-HTP", "NNAPI"]
