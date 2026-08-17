from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..config import PROJECT_ROOT
from ..organism_cell_vae_v1.cache import load as load_cache
from ..organism_cell_vae_v1.contract import canonical, sha256_file
from .runtime import ContinuousCellVAERuntime, DEFAULT_RELEASE


FAMILIES = ("HUMANOID", "ANIMALIAN", "PLANTLIKE", "ANOMALY", "MACHINE")
IDENTITIES = (5, 11, 17, 23, 29)


def build(output: Path = PROJECT_ROOT / "examples/showcase/neural_cell_vae_five_family.gif", *, device: str = "cuda") -> dict:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = ContinuousCellVAERuntime.from_release(device=device)
    cache = load_cache(DEFAULT_RELEASE)
    rows = []
    for identity in IDENTITIES:
        indices = [index for index, value in enumerate(cache["identity"].tolist()) if int(value) == identity]
        indices.sort(key=lambda index: int(cache["phase_index"][index]))
        if len(indices) != 16:
            raise ValueError("continuous cell VAE showcase phase census drifted")
        rows.append(indices)
    frames = []
    for phase in range(16):
        chosen = [row[phase] for row in rows]
        rgba = runtime.render_features(cache["features"][chosen], cache["mask"][chosen]).permute(0, 2, 3, 1).numpy()
        canvas = Image.new("RGBA", (5 * 144, 188), (3, 8, 14, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), "CONTINUOUS CELL VAE // LIVE POSED CELLS", fill=(74, 235, 255, 255))
        for column, family in enumerate(FAMILIES):
            value = (np.clip(rgba[column], 0, 1) * 255 + .5).astype(np.uint8)
            sprite = Image.fromarray(value, "RGBA").resize((144, 144), Image.Resampling.NEAREST)
            canvas.alpha_composite(sprite, (column * 144, 26))
            draw.text((column * 144 + 8, 171), family, fill=(190, 255, 80, 255))
        frames.append(canvas.convert("P", palette=Image.Palette.ADAPTIVE, colors=255))
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=False, disposal=2)
    report = {
        "format": "nullvector-continuous-cell-vae-runtime-showcase/1.0.0",
        "status": "ready",
        "release_manifest_sha256": json.loads((DEFAULT_RELEASE / "evaluation_manifest.json").read_text("utf-8"))["manifest_sha256"],
        "families": list(FAMILIES),
        "identities": list(IDENTITIES),
        "frames": len(frames),
        "neural_raster": True,
        "cell_positions_supplied_by_motion_scaffold": True,
        "artifact": {"path": output.relative_to(PROJECT_ROOT).as_posix(), "bytes": output.stat().st_size, "sha256": sha256_file(output)},
    }
    report["report_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    report_path = DEFAULT_RELEASE / "runtime_showcase_report.json"
    report_path.write_bytes(canonical(report))
    return report
