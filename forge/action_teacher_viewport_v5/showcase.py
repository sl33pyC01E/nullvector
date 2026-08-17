from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .recorder import validate_trajectory


def contact_sheet(session: Path, output: Path, *, columns=4, rows=3) -> Path:
    manifest = validate_trajectory(session)
    with np.load(session / manifest["artifact"]["path"], allow_pickle=False) as archive:
        frames = archive["frame"]
        actions = archive["action"]
    indices = np.linspace(0, len(frames) - 1, columns * rows).round().astype(int)
    cell_w, cell_h = frames.shape[2], frames.shape[1] + 18
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), (4, 10, 13))
    draw = ImageDraw.Draw(sheet)
    for slot, index in enumerate(indices):
        x, y = slot % columns * cell_w, slot // columns * cell_h
        sheet.paste(Image.fromarray(frames[index], "RGB"), (x, y))
        draw.text((x + 5, y + frames.shape[1] + 3), f"F{index:04}  A{int(actions[index]):02}", fill=(121, 226, 221))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)
    return output
