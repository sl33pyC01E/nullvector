from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from .config import ARCHETYPES, LAYER_NAMES

HULL, ARMOR, LEFT, RIGHT, WEAPON, CORE, CIRCUIT, EMISSION = range(8)


def mix32(value: int) -> int:
    value = (value ^ (value >> 16)) & 0xFFFFFFFF
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value = (value ^ (value >> 15)) & 0xFFFFFFFF
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) & 0xFFFFFFFF


def stream_seed(seed: int, stream: int) -> int:
    return mix32((seed & 0xFFFFFFFF) ^ ((stream * 0x9E3779B1) & 0xFFFFFFFF))


@dataclass(frozen=True, slots=True)
class SpriteGenome:
    seed: int
    archetype: int
    body_width: int
    body_length: int
    core_radius: int
    fin_span: int
    weapon_length: int
    armor_depth: int
    circuit_count: int
    notch_count: int
    x_offset: int
    y_offset: int
    palette_id: int

    @property
    def archetype_name(self) -> str:
        return ARCHETYPES[self.archetype]

    def to_dict(self) -> dict[str, int | str]:
        payload = asdict(self)
        payload["archetype_name"] = self.archetype_name
        return payload


def genome_from_seed(seed: int, archetype: int | str | None = None) -> SpriteGenome:
    seed = int(seed) & 0xFFFFFFFF
    rng = np.random.default_rng(stream_seed(seed, 1))
    if archetype is None:
        archetype_index = int(rng.integers(0, len(ARCHETYPES)))
    elif isinstance(archetype, str):
        archetype_index = ARCHETYPES.index(archetype)
    else:
        archetype_index = int(archetype) % len(ARCHETYPES)

    ranges = {
        0: ((7, 11), (17, 24), (2, 4), (5, 10), (4, 9)),
        1: ((8, 13), (18, 25), (2, 4), (4, 8), (2, 6)),
        2: ((10, 15), (15, 21), (2, 5), (3, 7), (3, 8)),
        3: ((13, 18), (14, 20), (3, 5), (3, 7), (2, 6)),
    }
    width, length, core, fins, weapon = ranges[archetype_index]
    return SpriteGenome(
        seed=seed,
        archetype=archetype_index,
        body_width=int(rng.integers(*width)),
        body_length=int(rng.integers(*length)),
        core_radius=int(rng.integers(*core)),
        fin_span=int(rng.integers(*fins)),
        weapon_length=int(rng.integers(*weapon)),
        armor_depth=int(rng.integers(1, 4)),
        circuit_count=int(rng.integers(2, 6)),
        notch_count=int(rng.integers(0, 4)),
        x_offset=int(rng.integers(-1, 2)),
        y_offset=int(rng.integers(-1, 2)),
        palette_id=int(rng.integers(0, 4)),
    )


def genome_vector(genome: SpriteGenome) -> np.ndarray:
    """Normalize controllable morphology into a stable neural condition vector."""
    values = (
        (genome.body_width - 7.0) / 11.0,
        (genome.body_length - 14.0) / 11.0,
        (genome.core_radius - 2.0) / 3.0,
        (genome.fin_span - 3.0) / 7.0,
        (genome.weapon_length - 2.0) / 7.0,
        (genome.armor_depth - 1.0) / 2.0,
        (genome.circuit_count - 2.0) / 3.0,
        genome.notch_count / 3.0,
    )
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)


def layers_to_tokens(layers: np.ndarray) -> np.ndarray:
    """Flatten overlapping semantic masks into an exclusive palette/part token map."""
    if layers.shape != (len(LAYER_NAMES), 32, 32):
        raise ValueError(f"Unexpected layer shape: {layers.shape}")
    tokens = np.zeros((32, 32), dtype=np.uint8)
    for index in range(len(LAYER_NAMES)):
        tokens[layers[index] > 0] = index + 1
    return tokens


def tokens_to_layers(tokens: np.ndarray) -> np.ndarray:
    if tokens.shape != (32, 32):
        raise ValueError(f"Unexpected token shape: {tokens.shape}")
    layers = np.stack([(tokens == index + 1) for index in range(len(LAYER_NAMES))])
    layers = layers.astype(np.uint8)
    layers[5] |= layers[7]
    layers[0] |= layers[6]
    return layers


def _new_layer(size: int) -> Image.Image:
    return Image.new("L", (size, size), 0)


def _shift_points(
    points: Iterable[tuple[int, int]], x_offset: int, y_offset: int
) -> list[tuple[int, int]]:
    return [(x + x_offset, y + y_offset) for x, y in points]


def _draw_dart(genome: SpriteGenome, layers: list[Image.Image]) -> None:
    cx = 16 + genome.x_offset
    cy = 16 + genome.y_offset
    half = genome.body_width // 2
    nose = max(2, cy - genome.body_length // 2)
    tail = min(29, cy + genome.body_length // 2)
    draw = [ImageDraw.Draw(layer) for layer in layers]

    draw[HULL].polygon(
        [(cx, nose), (cx + half, cy + 2), (cx + 2, tail), (cx - 2, tail), (cx - half, cy + 2)],
        fill=255,
    )
    span = genome.fin_span
    draw[LEFT].polygon(
        [(cx - half + 1, cy - 1), (max(1, cx - half - span), cy + 7), (cx - 2, cy + 5)],
        fill=255,
    )
    draw[RIGHT].polygon(
        [(cx + half - 1, cy - 1), (min(30, cx + half + span), cy + 7), (cx + 2, cy + 5)],
        fill=255,
    )
    draw[ARMOR].line([(cx - half + 2, cy + 1), (cx + half - 2, cy + 1)], fill=255, width=genome.armor_depth)
    draw[WEAPON].line([(cx, nose + 5), (cx, max(0, nose - genome.weapon_length // 2))], fill=255, width=2)
    draw[CORE].ellipse(
        [cx - genome.core_radius, cy - genome.core_radius, cx + genome.core_radius, cy + genome.core_radius],
        fill=255,
    )


def _draw_hound(genome: SpriteGenome, layers: list[Image.Image]) -> None:
    cx = 16 + genome.x_offset
    cy = 16 + genome.y_offset
    half = genome.body_width // 2
    nose = max(2, cy - genome.body_length // 2)
    tail = min(29, cy + genome.body_length // 2)
    draw = [ImageDraw.Draw(layer) for layer in layers]

    draw[HULL].polygon(
        [(cx, nose), (cx + half, cy - 2), (cx + half - 2, tail - 3), (cx, tail), (cx - half + 2, tail - 3), (cx - half, cy - 2)],
        fill=255,
    )
    span = genome.fin_span
    draw[LEFT].polygon(
        [(cx - half + 1, cy - 5), (max(1, cx - half - span), cy + 2), (cx - 3, cy + 8)],
        fill=255,
    )
    draw[RIGHT].polygon(
        [(cx + half - 1, cy - 5), (min(30, cx + half + span), cy + 2), (cx + 3, cy + 8)],
        fill=255,
    )
    draw[ARMOR].polygon(
        [(cx - half + 2, cy - 3), (cx, cy - 7), (cx + half - 2, cy - 3), (cx, cy + 1)],
        fill=255,
    )
    draw[WEAPON].polygon(
        [(cx - 2, nose + 5), (cx, max(1, nose - genome.weapon_length // 3)), (cx + 2, nose + 5)],
        fill=255,
    )
    r = genome.core_radius
    draw[CORE].rectangle([cx - r, cy + 1 - r, cx + r, cy + 1 + r], fill=255)


def _draw_oracle(genome: SpriteGenome, layers: list[Image.Image]) -> None:
    cx = 16 + genome.x_offset
    cy = 16 + genome.y_offset
    half_w = genome.body_width // 2
    half_h = genome.body_length // 2
    draw = [ImageDraw.Draw(layer) for layer in layers]

    draw[HULL].polygon(
        [(cx, cy - half_h), (cx + half_w, cy), (cx, cy + half_h), (cx - half_w, cy)],
        fill=255,
    )
    span = genome.fin_span
    pod_r = max(1, genome.armor_depth - 1)
    draw[LEFT].ellipse(
        [cx - half_w - span - pod_r, cy - pod_r, cx - half_w - span + pod_r, cy + pod_r],
        fill=255,
    )
    draw[RIGHT].ellipse(
        [cx + half_w + span - pod_r, cy - pod_r, cx + half_w + span + pod_r, cy + pod_r],
        fill=255,
    )
    draw[LEFT].line([(cx - half_w, cy), (cx - half_w - span, cy)], fill=255, width=1)
    draw[RIGHT].line([(cx + half_w, cy), (cx + half_w + span, cy)], fill=255, width=1)
    draw[ARMOR].polygon(
        [(cx, cy - half_h + 2), (cx + half_w - 2, cy), (cx, cy + 2), (cx - half_w + 2, cy)],
        fill=255,
    )
    draw[WEAPON].line([(cx, cy - 2), (cx, max(1, cy - half_h - genome.weapon_length))], fill=255, width=2)
    r = genome.core_radius
    draw[CORE].ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)


def _draw_bulwark(genome: SpriteGenome, layers: list[Image.Image]) -> None:
    cx = 16 + genome.x_offset
    cy = 16 + genome.y_offset
    half_w = genome.body_width // 2
    half_h = genome.body_length // 2
    draw = [ImageDraw.Draw(layer) for layer in layers]

    draw[HULL].polygon(
        [
            (cx - half_w + 3, cy - half_h),
            (cx + half_w - 3, cy - half_h),
            (cx + half_w, cy - half_h + 4),
            (cx + half_w, cy + half_h - 4),
            (cx + half_w - 3, cy + half_h),
            (cx - half_w + 3, cy + half_h),
            (cx - half_w, cy + half_h - 4),
            (cx - half_w, cy - half_h + 4),
        ],
        fill=255,
    )
    span = genome.fin_span
    draw[LEFT].rectangle(
        [max(1, cx - half_w - span), cy - half_h + 4, cx - half_w + 1, cy + half_h - 3],
        fill=255,
    )
    draw[RIGHT].rectangle(
        [cx + half_w - 1, cy - half_h + 4, min(30, cx + half_w + span), cy + half_h - 3],
        fill=255,
    )
    depth = genome.armor_depth
    draw[ARMOR].line(
        [(cx - half_w + 2, cy - half_h + 3), (cx + half_w - 2, cy - half_h + 3)],
        fill=255,
        width=depth,
    )
    draw[ARMOR].line(
        [(cx - half_w + 3, cy + half_h - 4), (cx + half_w - 3, cy + half_h - 4)],
        fill=255,
        width=max(1, depth - 1),
    )
    barrel_half = max(1, depth // 2)
    draw[WEAPON].rectangle(
        [cx - barrel_half, max(0, cy - half_h - genome.weapon_length), cx + barrel_half, cy],
        fill=255,
    )
    r = genome.core_radius
    draw[CORE].rectangle([cx - r, cy - r, cx + r, cy + r], fill=255)


def _draw_circuitry(genome: SpriteGenome, layers: list[Image.Image]) -> None:
    cx = 16 + genome.x_offset
    cy = 16 + genome.y_offset
    rng = np.random.default_rng(stream_seed(genome.seed, 22))
    circuit = ImageDraw.Draw(layers[CIRCUIT])
    emission = ImageDraw.Draw(layers[EMISSION])
    hull_union = np.maximum.reduce(
        [np.asarray(layers[index], dtype=np.uint8) for index in (HULL, ARMOR, LEFT, RIGHT, WEAPON)]
    )

    for _ in range(genome.circuit_count):
        angle = float(rng.uniform(0.0, np.pi * 2.0))
        distance = int(rng.integers(genome.core_radius + 2, 11))
        tx = int(round(cx + np.cos(angle) * distance))
        ty = int(round(cy + np.sin(angle) * distance))
        tx = int(np.clip(tx, 1, 30))
        ty = int(np.clip(ty, 1, 30))
        elbow = (tx, cy) if rng.random() < 0.5 else (cx, ty)
        circuit.line([(cx, cy), elbow, (tx, ty)], fill=255, width=1)
        if hull_union[ty, tx] > 0:
            emission.rectangle([tx, ty, tx + 1, ty + 1], fill=255)

    r = max(1, genome.core_radius - 1)
    emission.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    circuit_array = np.asarray(layers[CIRCUIT], dtype=np.uint8)
    circuit_array = np.where(hull_union > 0, circuit_array, 0).astype(np.uint8)
    layers[CIRCUIT] = Image.fromarray(circuit_array)


def _cut_notches(genome: SpriteGenome, layers: list[Image.Image]) -> None:
    if genome.notch_count <= 0:
        return
    rng = np.random.default_rng(stream_seed(genome.seed, 31))
    for _ in range(genome.notch_count):
        x = int(rng.integers(7, 25))
        y = int(rng.integers(8, 27))
        for index in (HULL, ARMOR):
            draw = ImageDraw.Draw(layers[index])
            draw.rectangle([x, y, x + int(rng.integers(0, 2)), y + int(rng.integers(0, 2))], fill=0)


def render_layers(genome: SpriteGenome, size: int = 32) -> np.ndarray:
    if size != 32:
        raise ValueError("The neural grammar is authored on a 32x32 logical grid.")
    layers = [_new_layer(size) for _ in LAYER_NAMES]
    draw_fn = (_draw_dart, _draw_hound, _draw_oracle, _draw_bulwark)[genome.archetype]
    draw_fn(genome, layers)
    _cut_notches(genome, layers)
    _draw_circuitry(genome, layers)
    return np.stack(
        [(np.asarray(layer, dtype=np.uint8) > 0).astype(np.uint8) for layer in layers],
        axis=0,
    )


def _dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant")
        views = [
            padded[y : y + result.shape[0], x : x + result.shape[1]]
            for y in range(3)
            for x in range(3)
        ]
        result = np.logical_or.reduce(views)
    return result


def palette_for_seed(seed: int, faction: str = "hostile") -> dict[str, tuple[int, int, int, int]]:
    palettes = {
        "hostile": (
            ((255, 48, 166, 255), (255, 126, 52, 255)),
            ((255, 61, 92, 255), (255, 183, 60, 255)),
            ((205, 69, 255, 255), (255, 82, 136, 255)),
            ((255, 92, 62, 255), (255, 218, 79, 255)),
        ),
        "player": (
            ((64, 237, 255, 255), (183, 255, 83, 255)),
            ((71, 174, 255, 255), (92, 255, 224, 255)),
            ((139, 108, 255, 255), (71, 241, 255, 255)),
            ((73, 255, 193, 255), (214, 255, 82, 255)),
        ),
    }
    primary, secondary = palettes[faction][mix32(seed) % len(palettes[faction])]
    return {
        "outline": (5, 7, 15, 255),
        "shadow": tuple(max(0, channel // 5) for channel in primary[:3]) + (255,),
        "primary": primary,
        "secondary": secondary,
        "hot": (245, 252, 255, 255),
    }


def compose_rgba(
    layers: np.ndarray,
    seed: int,
    faction: str = "hostile",
    threshold: float = 0.5,
) -> np.ndarray:
    if layers.shape != (len(LAYER_NAMES), 32, 32):
        raise ValueError(f"Expected {(len(LAYER_NAMES), 32, 32)}, got {layers.shape}.")
    masks = layers >= threshold
    major = np.logical_or.reduce(masks[[HULL, ARMOR, LEFT, RIGHT, WEAPON, CORE]])
    outline = _dilate(major, 1) & ~major
    palette = palette_for_seed(seed, faction)
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)

    rgba[outline] = palette["outline"]
    rgba[masks[HULL]] = palette["shadow"]
    rgba[masks[LEFT] | masks[RIGHT]] = palette["secondary"]
    rgba[masks[ARMOR]] = palette["primary"]
    rgba[masks[WEAPON]] = palette["primary"]
    rgba[masks[CORE]] = palette["hot"]
    rgba[masks[CIRCUIT]] = palette["secondary"]
    rgba[masks[EMISSION]] = palette["hot"]
    rgba[..., 3] = np.where(outline | major | masks[CIRCUIT] | masks[EMISSION], 255, 0).astype(np.uint8)
    return rgba
