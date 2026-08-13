from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.map_decorator_ml.checkpoint import file_sha256
from forge.map_decorator_production_v2 import recovery


def _manifest() -> dict[str, object]:
    return {
        "corpus_sha256": "c" * 64,
        "shards": [
            {"shard_id": "shard-a"},
            {"shard_id": "shard-b"},
        ],
    }


def _fake_validator(
    path: Path,
    *,
    corpus_sha256: str,
    entry: dict[str, object],
    shard_index: int,
) -> dict[str, object]:
    assert corpus_sha256 == "c" * 64
    assert path.read_text(encoding="utf-8") == f"payload-{shard_index}"
    return {
        "samples": [{"sample_identity_sha256": f"{shard_index + 1:064x}"}],
        "samples_sha256": f"{shard_index + 10:064x}",
    }


def test_scan_accepts_atomic_shards_and_reports_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "stage"
    path = staging / "shards" / "shard-a" / "counts.json"
    path.parent.mkdir(parents=True)
    path.write_text("payload-0", encoding="utf-8")
    monkeypatch.setattr(recovery, "_validate_shard_index", _fake_validator)
    scan = recovery._scan_valid_shards(staging, _manifest())
    assert [entry["shard_id"] for entry in scan["valid"]] == ["shard-a"]
    assert scan["missing"] == [1]
    assert scan["valid_sample_count"] == 1


def test_scan_fails_closed_on_unknown_published_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "stage"
    path = staging / "shards" / "unknown" / "counts.json"
    path.parent.mkdir(parents=True)
    path.write_text("payload", encoding="utf-8")
    monkeypatch.setattr(recovery, "_validate_shard_index", _fake_validator)
    with pytest.raises(ValueError, match="unknown shard"):
        recovery._scan_valid_shards(staging, _manifest())


def test_atomic_copy_is_byte_exact_and_never_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    target = tmp_path / "target" / "counts.json"
    source.write_bytes(b"abc\x00def\n")
    digest = file_sha256(source)
    recovery._atomic_copy(source, target, expected_sha256=digest)
    assert target.read_bytes() == source.read_bytes()
    assert file_sha256(target) == digest
    with pytest.raises(FileExistsError):
        recovery._atomic_copy(source, target, expected_sha256=digest)


def test_all_imported_recovery_preserves_evidence_and_atomically_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corpus_manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    parent = tmp_path / "output-parent"
    source = parent / ".foreground.tmp-outage"
    for index, name in enumerate(("shard-a", "shard-b")):
        path = source / "shards" / name / "counts.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"payload-{index}", encoding="utf-8")
    evidence_before = recovery._tree_inventory(source)
    output = parent / "foreground"

    monkeypatch.setattr(
        recovery,
        "validate_corpus",
        lambda *args, **kwargs: {"passed": True},
    )
    monkeypatch.setattr(recovery, "_validate_shard_index", _fake_validator)

    def fake_aggregate(
        corpus_path: Path,
        staging: Path,
        telemetry: list[dict[str, object]],
    ) -> dict[str, object]:
        assert corpus_path == corpus.resolve()
        assert telemetry == []
        (staging / recovery.INDEX_VALIDATION_FILE).write_text(
            json.dumps({"passed": True, "sample_count": 2}), encoding="utf-8"
        )
        manifest = {"foreground_index_sha256": "f" * 64}
        (staging / recovery.INDEX_MANIFEST_FILE).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest

    monkeypatch.setattr(recovery, "_aggregate", fake_aggregate)
    monkeypatch.setattr(
        recovery,
        "validate_foreground_index",
        lambda *args, **kwargs: {"passed": True, "sample_count": 2},
    )
    result = recovery.recover_foreground_index(
        corpus,
        source,
        output,
        python=Path(__import__("sys").executable),
        max_workers=1,
    )
    assert result["passed"]
    assert result["imported_shard_count"] == 2
    assert result["built_shard_count"] == 0
    assert output.is_dir()
    assert source.is_dir()
    assert recovery._tree_inventory(source) == evidence_before
    report = json.loads((output / recovery.RECOVERY_REPORT_FILE).read_text(encoding="utf-8"))
    assert report["source_staging_preserved"] is True
    assert report["atomic_publication"] is True
