from pathlib import Path

from forge.neural_world_synthesis_v1.contract import DEFAULT_OUTPUT, FORMAT, PRIOR_CHECKPOINT, SOURCE_FILES, source_manifest, source_sha256


def test_neural_world_synthesis_contract_is_bound_and_additive() -> None:
    assert FORMAT == "nullvector-neural-world-synthesis-v1/1.0.0"
    assert PRIOR_CHECKPOINT.is_file()
    assert len(source_manifest()) == len(SOURCE_FILES) == 7
    assert len(source_sha256()) == 64
    assert "neural_world_synthesis_v1" in DEFAULT_OUTPUT.as_posix()


def test_neural_world_synthesis_source_preserves_runtime_cadence() -> None:
    source = (Path(__file__).parents[1] / "forge/neural_world_synthesis_v1/build.py").read_text("utf-8")
    assert '"display_target_fps": 30' in source
    assert '"embodied_motion_hz": 30' in source
    assert '"causal_world_hz": 15' in source
    assert '"region_entry_or_background_only"' in source
    assert "MAX_REPAIR_FRACTION = 0.15" in source
    assert "decorated_world_contact_sheet.png" in source
