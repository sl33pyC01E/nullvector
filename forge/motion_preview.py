from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from .config import ARCHETYPES, GAME_GENERATED_DIR, OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an animated review sheet from baked neural sprites."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=GAME_GENERATED_DIR / "sprite_registry.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "neural_sprite_motion.gif",
    )
    return parser.parse_args()


def _frame_sequence(manifest: dict) -> list[dict]:
    animations = manifest["animations"]
    return (
        animations["idle"]["frames"] * 2
        + animations["move"]["frames"] * 2
        + animations["attack"]["frames"]
        + animations["hit"]["frames"]
        + animations["idle"]["frames"]
    )


def _background(width: int, height: int) -> Image.Image:
    image = Image.new("RGBA", (width, height), (3, 5, 12, 255))
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 24):
        color = (31, 66, 84, 58 if x % 96 == 0 else 24)
        draw.line([(x, 42), (x, height)], fill=color, width=1)
    for y in range(42, height, 24):
        color = (31, 66, 84, 58 if y % 96 == 42 else 24)
        draw.line([(0, y), (width, y)], fill=color, width=1)
    draw.line([(20, 34), (width - 20, 34)], fill=(55, 89, 112, 255), width=1)
    draw.text(
        (20, 13),
        "NEURAL BODY DIFFUSION // PROCEDURAL LAYER RIG",
        fill=(215, 244, 255, 255),
    )
    return image


def main() -> None:
    args = parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    manifests = []
    for archetype in ARCHETYPES:
        manifests.append(
            next(sprite for sprite in registry["sprites"] if sprite["archetype"] == archetype)
        )

    root = args.registry.parent
    atlases = [Image.open(root / manifest["atlas"]).convert("RGBA") for manifest in manifests]
    emissions = [
        Image.open(root / manifest["emission_atlas"]).convert("RGBA")
        for manifest in manifests
    ]
    sequences = [_frame_sequence(manifest) for manifest in manifests]
    frame_count = max(len(sequence) for sequence in sequences)
    width, height = 1040, 300
    frames: list[Image.Image] = []
    durations: list[int] = []
    scale = 6
    sprite_size = 32 * scale

    for tick in range(frame_count):
        canvas = _background(width, height)
        draw = ImageDraw.Draw(canvas)
        durations.append(int(sequences[0][tick % len(sequences[0])]["duration_ms"]))
        for index, manifest in enumerate(manifests):
            sequence = sequences[index]
            frame = sequence[tick % len(sequence)]
            x, y, frame_width, frame_height = map(int, frame["rect"])
            region = (x, y, x + frame_width, y + frame_height)
            base = atlases[index].crop(region).resize(
                (sprite_size, sprite_size), Image.Resampling.NEAREST
            )
            emission = emissions[index].crop(region).resize(
                (sprite_size, sprite_size), Image.Resampling.NEAREST
            )
            glow_color = (70, 239, 255, 205) if index == 0 else (255, 54, 157, 190)
            glow = Image.new("RGBA", emission.size, glow_color)
            glow.putalpha(emission.getchannel("A"))
            wide = glow.filter(ImageFilter.GaussianBlur(14))
            tight = glow.filter(ImageFilter.GaussianBlur(5))
            left = 40 + index * 250
            top = 67
            canvas.alpha_composite(wide, (left, top))
            canvas.alpha_composite(tight, (left, top))
            canvas.alpha_composite(base, (left, top))
            draw.text(
                (left + 2, 255),
                manifest["archetype"].upper(),
                fill=(230, 239, 251, 255),
            )
            draw.text(
                (left + 2, 270),
                "SEED 0x%08X" % int(manifest["seed"]),
                fill=(102, 124, 151, 255),
            )
        frames.append(canvas.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(args.output)


if __name__ == "__main__":
    main()
