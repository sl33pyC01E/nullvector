from __future__ import annotations

import json

from PIL import Image

from forge.grammar import genome_from_seed, render_layers
from forge.rig import ANIMATIONS, bake_animation_atlas


def test_animation_baker_emits_atlas_and_manifest(tmp_path) -> None:
    layers = render_layers(genome_from_seed(456, "dart"))
    atlas_path = tmp_path / "dart.png"
    manifest = bake_animation_atlas(layers, 456, 0, atlas_path)
    assert atlas_path.exists()
    assert atlas_path.with_name("dart_emission.png").exists()
    with Image.open(atlas_path) as atlas:
        assert atlas.size == (max(map(len, ANIMATIONS.values())) * 32, len(ANIMATIONS) * 32)
        assert atlas.mode == "RGBA"
    assert set(manifest["animations"]) == set(ANIMATIONS)
    assert json.loads(json.dumps(manifest))["seed"] == 456
