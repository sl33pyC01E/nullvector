from __future__ import annotations

import json
from pathlib import Path
import shutil
import zipfile

import jsonschema
import numpy as np
import pytest

from forge.config import PROJECT_ROOT
from forge.map_topology_neural.artifacts import (
    COMPILED_MANIFEST,
    LEDGER_FILE,
    load_compiled_artifact,
    write_compiled_artifact,
    write_raw_artifact,
)
from forge.map_topology_neural.compiler import (
    THEME_HAZARDS,
    assert_exact_compiler_replay,
    compile_topology,
    make_raw_topology,
)
from forge.map_topology_neural.contract import (
    CONTRACT_SHA256,
    FIELD_ORDER,
    POINT_CHANNELS,
    TopologyConditions,
    contract_manifest,
    crop_categorical,
    encode_topology_tensor,
)
from forge.map_topology_neural.corpus import FROZEN_CORPUS_SHA256, TopologyCorpus
from forge.map_topology_neural.hashing import file_sha256, json_sha256
from forge.maps.model import THEMES, Hazard, MapConfig, Terrain


CORPUS_ROOT = PROJECT_ROOT / "outputs" / "map_decorator_corpus_v1"


def _case(
    *,
    theme: str = "arena",
    width: int = 32,
    height: int = 32,
    thin: bool = False,
    hazards: bool = False,
) -> tuple[object, MapConfig, tuple[int, int], tuple[int, int], tuple[tuple[int, int], ...]]:
    terrain = np.full((height, width), int(Terrain.WALL), dtype=np.uint8)
    hazard = np.zeros((height, width), dtype=np.uint8)
    elevation = np.zeros((height, width), dtype=np.int8)
    start = (2, height // 2)
    exit_point = (width - 3, height // 2)
    objectives = ((width // 2, 2),)
    if thin:
        terrain[start[1], 1 : width - 1] = int(Terrain.FLOOR)
        terrain[1 : height - 1, objectives[0][0]] = int(Terrain.FLOOR)
    if hazards:
        hazard[start[1], 2 : width - 2] = THEME_HAZARDS[theme][0]
    # Deliberately illegal tuples and open boundaries; IDs remain legal.
    terrain[0, width // 2] = int(Terrain.FLOOR)
    hazard[0, width // 2] = THEME_HAZARDS[theme][0]
    elevation[0, width // 2] = 5
    config = MapConfig(width=width, height=height, objective_count=1, spawn_count=0)
    return make_raw_topology(terrain, hazard, elevation), config, start, exit_point, objectives


def _compile_case(**kwargs):
    raw, config, start, exit_point, objectives = _case(**kwargs)
    result = compile_topology(
        raw,
        seed=0x1234,
        theme=kwargs.get("theme", "arena"),
        config=config,
        start=start,
        exit=exit_point,
        objectives=objectives,
        spawns=(),
    )
    return raw, result


def test_contract_exact_right_bottom_padding_and_crop_rectangular() -> None:
    contract_schema = json.loads(
        (PROJECT_ROOT / "shared" / "schema" / "map_topology_neural_contract.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(contract_schema).validate(contract_manifest())
    raw, config, start, exit_point, objectives = _case(width=47, height=33)
    tensor = encode_topology_tensor(
        terrain=raw.terrain,
        hazard=raw.hazard,
        elevation=raw.elevation,
        theme="arena",
        config=config,
        start=start,
        exit=exit_point,
        objectives=objectives,
        spawns=(),
        conditions=TopologyConditions(0.1, 0.4, 0.7),
    )
    assert tensor.contract_sha256 == CONTRACT_SHA256
    assert tensor.categorical.shape == (3, 36, 48)
    assert tensor.pad_bottom == 3 and tensor.pad_right == 1
    assert tensor.point_heatmaps.shape == (len(POINT_CHANNELS), 36, 48)
    assert int(tensor.valid_mask.sum()) == 33 * 47
    assert tensor.point_heatmaps[0, start[1], start[0]] == 1
    assert tensor.point_heatmaps[1, exit_point[1], exit_point[0]] == 1
    assert tensor.point_heatmaps[2].sum() == 1
    decoded = crop_categorical(tensor, tensor.categorical)
    assert tuple(decoded) == FIELD_ORDER
    assert np.array_equal(decoded["terrain"], raw.terrain)
    assert np.array_equal(decoded["hazard"], raw.hazard)
    assert np.array_equal(decoded["elevation"], raw.elevation)
    assert not tensor.categorical.flags.writeable
    assert not tensor.point_heatmaps.flags.writeable


def test_contract_rejects_independent_or_illegal_fields() -> None:
    terrain = np.zeros((32, 32), dtype=np.uint8)
    hazard = np.zeros_like(terrain)
    elevation = np.zeros((32, 32), dtype=np.int8)
    terrain[5, 5] = 255
    with pytest.raises(ValueError, match="illegal categorical ID"):
        make_raw_topology(terrain, hazard, elevation)
    terrain[5, 5] = 0
    with pytest.raises(TypeError, match="dtype"):
        make_raw_topology(terrain.astype(np.int16), hazard, elevation)


@pytest.mark.parametrize(("width", "height"), [(32, 32), (256, 255)])
def test_contract_supports_exact_dimension_extrema(width: int, height: int) -> None:
    terrain = np.full((height, width), int(Terrain.WALL), dtype=np.uint8)
    hazard = np.zeros((height, width), dtype=np.uint8)
    elevation = np.zeros((height, width), dtype=np.int8)
    config = MapConfig(width=width, height=height, objective_count=1, spawn_count=0)
    tensor = encode_topology_tensor(
        terrain=terrain,
        hazard=hazard,
        elevation=elevation,
        theme="rooms",
        config=config,
        start=(2, height // 2),
        exit=(width - 3, height // 2),
        objectives=((width // 2, 2),),
        spawns=(),
        conditions=TopologyConditions(0.0, 0.0, 0.0),
    )
    assert tensor.original_width == width and tensor.original_height == height
    assert tensor.padded_width % 4 == 0 and tensor.padded_height % 4 == 0
    assert tensor.padded_width - width == tensor.pad_right
    assert tensor.padded_height - height == tensor.pad_bottom


def test_compiler_repairs_disconnected_points_thin_corridor_hazards_and_boundary() -> None:
    raw, result = _compile_case(thin=True, hazards=True)
    assert result.report["validation"]["passed"]
    assert not result.data.walkability[0].any()
    assert not result.data.walkability[-1].any()
    assert not result.data.walkability[:, 0].any()
    assert not result.data.walkability[:, -1].any()
    assert (result.data.hazard[result.data.protected_backbone != 0] == 0).all()
    padded = np.pad(result.data.walkability.astype(bool), 1, constant_values=False)
    eroded = np.logical_and.reduce(
        [
            padded[dy : dy + result.data.shape[0], dx : dx + result.data.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ]
    )
    assert eroded[result.data.start[1], result.data.start[0]]
    assert eroded[result.data.exit[1], result.data.exit[0]]
    assert result.report["costs"]["radius_one_widening_cells"] > 0
    assert result.report["costs"]["hazard_cells_cleared"] > 0
    assert not result.report["quality"]["pre"]["agent_scale_mission_connected"]
    assert result.report["quality"]["post"]["agent_scale_mission_connected"]
    assert "agent_scale_mission_articulation_count" in result.report["quality"]["post"]
    assert_exact_compiler_replay(result, raw)


def test_compiler_ledger_has_no_silent_mutations_and_is_deterministic() -> None:
    raw, result = _compile_case(thin=True, hazards=True)
    state = {
        "terrain": raw.terrain.copy(),
        "hazard": raw.hazard.copy(),
        "elevation": raw.elevation.copy(),
        "protected_backbone": np.zeros(raw.terrain.shape, dtype=np.uint8),
        "required_clearance": np.zeros(raw.terrain.shape, dtype=np.uint8),
    }
    for sequence, entry in enumerate(result.ledger):
        assert entry["sequence"] == sequence
        field = entry["field"]
        x, y = entry["x"], entry["y"]
        assert int(state[field][y, x]) == entry["before"]
        assert entry["before"] != entry["after"]
        state[field][y, x] = entry["after"]
    for name in ("terrain", "hazard", "elevation", "protected_backbone", "required_clearance"):
        assert np.array_equal(state[name], getattr(result.data, name))
    replay = compile_topology(
        raw,
        seed=result.data.seed,
        theme=result.data.theme,
        config=result.data.config,
        start=result.data.start,
        exit=result.data.exit,
        objectives=result.data.objectives,
        spawns=result.data.spawns,
    )
    assert replay.ledger == result.ledger
    assert replay.report == result.report


def test_compiler_rejects_unsafe_immutable_points() -> None:
    raw, config, _, exit_point, objectives = _case()
    with pytest.raises(ValueError, match="boundary margin"):
        compile_topology(
            raw,
            seed=1,
            theme="arena",
            config=config,
            start=(1, 16),
            exit=exit_point,
            objectives=objectives,
            spawns=(),
        )


@pytest.mark.parametrize(
    ("theme", "width", "height"),
    [
        ("arena", 32, 32),
        ("rooms", 33, 47),
        ("caves", 48, 32),
        ("archipelago", 32, 48),
        ("garden", 39, 35),
        ("anomaly", 48, 48),
    ],
)
def test_bounded_six_theme_rectangular_compiler_fuzz(theme: str, width: int, height: int) -> None:
    rng = np.random.Generator(np.random.PCG64(0xF022 + THEMES.index(theme)))
    terrain = rng.integers(0, 9, size=(height, width), dtype=np.uint8)
    hazard = rng.integers(0, 5, size=(height, width), dtype=np.uint8)
    elevation = rng.integers(0, 6, size=(height, width), dtype=np.int8)
    raw = make_raw_topology(terrain, hazard, elevation)
    config = MapConfig(width=width, height=height, objective_count=1, spawn_count=0)
    start = (2, height // 2)
    exit_point = (width - 3, height // 2)
    objective = ((width // 2, 2),)
    result = compile_topology(
        raw,
        seed=0xA11CE + THEMES.index(theme),
        theme=theme,
        config=config,
        start=start,
        exit=exit_point,
        objectives=objective,
        spawns=(),
    )
    assert result.report["passed"]
    assert result.report["validation"]["passed"]
    assert result.data.shape == (height, width)
    assert set(np.unique(result.data.hazard)) <= {0, *THEME_HAZARDS[theme]}


@pytest.fixture(scope="module")
def frozen_corpus() -> TopologyCorpus:
    return TopologyCorpus(CORPUS_ROOT)


def test_corpus_reader_loads_only_bounded_topology_and_point_members(frozen_corpus: TopologyCorpus) -> None:
    shard_id = frozen_corpus.find_shard(theme="arena", width=32, height=32, objective_count=1)
    sample = frozen_corpus.read_sample(shard_id, 0, expected_split="train")
    assert sample.corpus_sha256 == FROZEN_CORPUS_SHA256
    assert set(sample.member_array_sha256) == {
        "semantic_terrain", "semantic_hazard", "semantic_elevation", "seeds",
        "theme_index", "start", "exit", "objectives", "spawns",
    }
    assert "features" not in sample.member_array_sha256
    assert sample.raw.terrain.shape == (32, 32)
    assert sample.raw.terrain.nbytes + sample.raw.hazard.nbytes + sample.raw.elevation.nbytes == 3072


def _copy_corpus_subset(tmp_path: Path) -> tuple[Path, str, dict[str, object]]:
    destination = tmp_path / "corpus"
    manifest = json.loads((CORPUS_ROOT / "corpus_manifest.json").read_text(encoding="utf-8"))
    shard_id = "main-t00-p00-o00"
    entry = next(item for item in manifest["shards"] if item["shard_id"] == shard_id)
    for relative in ("corpus_validation.json", entry["sidecar"], entry["artifact"]):
        source = CORPUS_ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(CORPUS_ROOT / "corpus_manifest.json", destination / "corpus_manifest.json")
    return destination, shard_id, entry


def _rehash_subset(root: Path, shard_id: str, *, artifact_changed: bool) -> None:
    manifest_path = root / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["shards"] if item["shard_id"] == shard_id)
    sidecar_path = root / entry["sidecar"]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if artifact_changed:
        artifact_path = root / entry["artifact"]
        digest = file_sha256(artifact_path)
        sidecar["artifact"]["sha256"] = digest
        sidecar["artifact"]["compressed_bytes"] = artifact_path.stat().st_size
        entry["artifact_sha256"] = digest
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    entry["sidecar_sha256"] = file_sha256(sidecar_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_corpus_rejects_member_traversal_even_after_local_rehash(tmp_path: Path) -> None:
    root, shard_id, entry = _copy_corpus_subset(tmp_path)
    artifact = root / entry["artifact"]
    with zipfile.ZipFile(artifact, mode="a", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("../escape.npy", b"not an array")
    _rehash_subset(root, shard_id, artifact_changed=True)
    with pytest.raises(ValueError, match="member census|unsafe member path"):
        TopologyCorpus(
            root,
            expected_manifest_file_sha256=file_sha256(root / "corpus_manifest.json"),
        ).read_sample(shard_id, 0)


def test_corpus_default_reader_pins_complete_root_manifest_file(tmp_path: Path) -> None:
    root, _, _ = _copy_corpus_subset(tmp_path)
    assert TopologyCorpus(root).corpus_sha256 == FROZEN_CORPUS_SHA256
    manifest_path = root / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["telemetry"]["build_attempt_count"] += 1
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned frozen artifact"):
        TopologyCorpus(root)


def test_corpus_rejects_duplicate_zip_member_even_after_local_rehash(tmp_path: Path) -> None:
    root, shard_id, entry = _copy_corpus_subset(tmp_path)
    artifact = root / entry["artifact"]
    with zipfile.ZipFile(artifact, mode="r") as archive:
        payload = archive.read("semantic_terrain.npy")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(artifact, mode="a", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("semantic_terrain.npy", payload)
    _rehash_subset(root, shard_id, artifact_changed=True)
    with pytest.raises(ValueError, match="duplicates"):
        TopologyCorpus(
            root,
            expected_manifest_file_sha256=file_sha256(root / "corpus_manifest.json"),
        ).read_sample(shard_id, 0)


@pytest.mark.parametrize(
    ("field", "value"),
    [("dtype", "|i1"), ("shape", [16, 32, 31])],
)
def test_corpus_rejects_topology_member_dtype_or_shape_sidecar_tamper(
    tmp_path: Path, field: str, value: object
) -> None:
    root, shard_id, entry = _copy_corpus_subset(tmp_path)
    sidecar_path = root / entry["sidecar"]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["artifact"]["members"]["semantic_terrain"][field] = value
    if field == "shape":
        sidecar["artifact"]["members"]["semantic_terrain"]["nbytes"] = 16 * 32 * 31
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    _rehash_subset(root, shard_id, artifact_changed=False)
    with pytest.raises(ValueError, match="descriptor drifted"):
        TopologyCorpus(
            root,
            expected_manifest_file_sha256=file_sha256(root / "corpus_manifest.json"),
        ).read_sample(shard_id, 0)


def test_corpus_rejects_member_header_oversize_bound(monkeypatch: pytest.MonkeyPatch, frozen_corpus: TopologyCorpus) -> None:
    import forge.map_topology_neural.corpus as corpus_module

    monkeypatch.setattr(corpus_module, "MAX_MEMBER_HEADER_BYTES", 1)
    shard_id = frozen_corpus.find_shard(theme="arena", width=32, height=32, objective_count=1)
    with pytest.raises(ValueError, match="bounded NPY header|array bytes"):
        frozen_corpus.read_sample(shard_id, 0)


def test_raw_compiled_artifacts_reject_forged_rehashed_ledger(tmp_path: Path) -> None:
    raw, result = _compile_case(thin=True, hazards=True)
    raw_artifact = write_raw_artifact(
        tmp_path / "case" / "raw",
        raw=raw,
        seed=result.data.seed,
        theme=result.data.theme,
        config=result.data.config,
        start=result.data.start,
        exit=result.data.exit,
        objectives=result.data.objectives,
        spawns=result.data.spawns,
        provenance={"test": "adversarial-ledger"},
        proposal_source="unit_test",
    )
    compiled_path = tmp_path / "case" / "compiled"
    write_compiled_artifact(compiled_path, raw_artifact=raw_artifact, result=result)
    ledger_path = compiled_path / LEDGER_FILE
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["entries"][0]["reason"] = "forged_but_locally_rehashed"
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    manifest_path = compiled_path / COMPILED_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger_semantic_hash = json_sha256(ledger)
    manifest["ledger_sha256"] = ledger_semantic_hash
    manifest["ledger"]["ledger_sha256"] = ledger_semantic_hash
    manifest["ledger"]["bytes"] = ledger_path.stat().st_size
    manifest["ledger"]["sha256"] = file_sha256(ledger_path)
    identity_keys = (
        "schema_version", "artifact_type", "authority", "source_sha256",
        "tensor_contract_sha256", "compiler", "compiler_source_sha256", "seed",
        "theme", "config", "points", "raw_manifest_sha256", "raw_identity_sha256",
        "raw_topology_sha256", "compiled_arrays_sha256", "ledger_sha256", "report_sha256",
    )
    manifest["compiled_identity_sha256"] = json_sha256({key: manifest[key] for key in identity_keys})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="exact compiler replay"):
        load_compiled_artifact(compiled_path, raw_artifact=raw_artifact, exact_replay=True)


def test_codec_deterministic_cpu_init_training_checkpoint_and_provenance(tmp_path: Path) -> None:
    import torch

    from forge.map_topology_neural.checkpoint import load_codec_checkpoint, save_codec_checkpoint
    from forge.map_topology_neural.codec import (
        CodecConfig,
        build_codec,
        collate_topology_tensors,
        train_cpu_smoke,
    )

    raw, config, start, exit_point, objectives = _case()
    tensor = encode_topology_tensor(
        terrain=raw.terrain,
        hazard=raw.hazard,
        elevation=raw.elevation,
        theme="arena",
        config=config,
        start=start,
        exit=exit_point,
        objectives=objectives,
        spawns=(),
        conditions=TopologyConditions(0.0, 0.1, 0.5),
    )
    config_model = CodecConfig(width=8, latent_dim=8, codebook_size=16, field_embedding_dim=2, residual_depth=0)
    global_before = torch.get_rng_state().clone()
    first = build_codec(config_model, init_seed=77)
    second = build_codec(config_model, init_seed=77)
    assert torch.equal(global_before, torch.get_rng_state())
    assert all(torch.equal(first.state_dict()[key], second.state_dict()[key]) for key in first.state_dict())
    batch = collate_topology_tensors([tensor])
    state = train_cpu_smoke(first, batch, steps=1, training_seed=88)
    checkpoint = tmp_path / "checkpoint.pt"
    sidecar = save_codec_checkpoint(
        checkpoint,
        model=first,
        model_init_seed=77,
        step=1,
        optimizer_state=state["optimizer_state"],
        ema_state=state["ema_state"],
        training_generator_state=state["training_generator_state"],
        torch_cpu_rng_state=state["torch_cpu_rng_state"],
        corpus_sha256=FROZEN_CORPUS_SHA256,
        metrics={"passed": True, "device": "cpu"},
    )
    loaded, payload, loaded_sidecar = load_codec_checkpoint(
        checkpoint, expected_corpus_sha256=FROZEN_CORPUS_SHA256
    )
    assert payload["authority"] == "representation_only_not_generative"
    assert all(value.device.type == "cpu" for value in loaded.state_dict().values())
    assert loaded_sidecar == sidecar
    schema = json.loads(
        (PROJECT_ROOT / "shared" / "schema" / "map_topology_neural_checkpoint.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(sidecar)
    with pytest.raises(ValueError, match="provenance"):
        load_codec_checkpoint(checkpoint, expected_corpus_sha256="0" * 64)


def test_six_theme_smoke_bank_builds_and_replays_exactly(tmp_path: Path) -> None:
    import torch

    from forge.map_topology_neural.smoke import assert_exact_smoke_replay, build_smoke

    output = tmp_path / "smoke"
    report = build_smoke(output, corpus_root=CORPUS_ROOT)
    assert report["passed"]
    assert report["theme_count"] == 6
    assert report["artifact_array_count_compared"] == 72
    assert report["checkpoint_step"] == 2
    assert report["codec_decode_logits_exact"]
    assert report["codec_decode_decisions_exact"]
    assert (output / "topology_repair_contact_sheet.png").is_file()
    assert assert_exact_smoke_replay(output) == report
    manifest = json.loads((output / "smoke_manifest.json").read_text())
    assert manifest["format"] == "nullvector-neural-map-topology-smoke-v2"
    assert manifest["codec"]["decode_execution"] == {
        "device": "cpu",
        "dtype": "float32",
        "torch_num_threads": 1,
        "mkldnn_enabled": False,
        "deterministic_algorithms": True,
        "update_ema": False,
    }

    # Ambient worker/thread choices cannot change the canonical replay.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2 if previous_threads != 2 else 3)
        assert assert_exact_smoke_replay(output) == report
    finally:
        torch.set_num_threads(previous_threads)
