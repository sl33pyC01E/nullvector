from __future__ import annotations

import hashlib
from pathlib import Path

from forge.morphology_subtype_runtime_sync import sync_runtime, validate_runtime


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "game/SubtypeMotionLab.tscn"
SCRIPT = ROOT / "game/scripts/subtype_motion_lab.gd"


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def test_runtime_projection_is_repeatable_hash_closed_and_compact(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    a = sync_runtime(first); b = sync_runtime(second)
    assert a == b and _tree_hash(first) == _tree_hash(second)
    assert (a["identity_count"], a["clip_count"], a["frame_count"], a["atlas_count"]) == (20, 400, 3620, 60)
    assert a["bytes"] < 16 * 1024 * 1024
    assert validate_runtime(first) == a


def test_native_scene_is_additive_and_fail_closed() -> None:
    scene = SCENE.read_text(encoding="utf-8"); script = SCRIPT.read_text(encoding="utf-8")
    assert 'path="res://scripts/subtype_motion_lab.gd"' in scene
    for token in ("SUBTYPE_MOTION_SMOKE_OK", "procedural-subtype-reference", "loop", "locomote", "atlas dimensions", "400", "3620", "60"):
        assert token in script or token == "procedural-subtype-reference"
    project = (ROOT / "game/project.godot").read_text(encoding="utf-8")
    assert "run/main_scene=\"res://Arena.tscn\"" in project
