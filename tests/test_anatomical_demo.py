from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "game/generated/anatomical_demo/v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_anatomical_demo_bundle_is_closed_and_feature_complete() -> None:
    manifest = json.loads((BUNDLE / "manifest.json").read_text("utf-8"))
    anatomy = json.loads((BUNDLE / "anatomy.json").read_text("utf-8"))
    assert manifest["status"] == "ready"
    assert manifest["layout"] == {
        "cell_size": 96, "families": 5, "phases": 16,
        "width": 480, "height": 1536,
    }
    for descriptor in manifest["artifacts"].values():
        path = BUNDLE / descriptor["path"]
        assert path.stat().st_size == descriptor["bytes"]
        assert _sha(path) == descriptor["sha256"]
    with Image.open(BUNDLE / "neural_motion_atlas.png") as image:
        assert image.mode == "RGBA"
        assert image.size == (480, 1536)
        assert image.getbbox() is not None
    assert len(anatomy["specimens"]) == 5
    assert [row["family_id"] for row in anatomy["specimens"]] == list(range(5))
    for specimen in anatomy["specimens"]:
        assert len(specimen["cells"]) >= 100
        assert any(component["organ"] != "none" for component in specimen["components"])
        assert specimen["skeleton"]["nodes"]
        assert specimen["skeleton"]["edges"]
        assert specimen["skeleton"]["muscles"]


def test_native_demo_and_launcher_are_present() -> None:
    assert (ROOT / "game/AnatomicalDemo.tscn").is_file()
    script = (ROOT / "game/scripts/anatomical_demo.gd").read_text("utf-8")
    for capability in (
        "_apply_cut", "_apply_radial", "_draw_organs", "_draw_skeleton",
        "_draw_contacts", "_systems", "_locomotion_integrity",
    ):
        assert capability in script
    assert (ROOT / "Launch_Anatomical_Demo.bat").is_file()
