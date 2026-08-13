from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .constants import CANVAS_SIZE, FAMILIES, LAYER_NAMES, ROLE_NAMES
from .contract import assert_valid_specimen
from .disk_guard import guard_corpus_destination
from .genome import genome_from_seed
from .render import MorphologySpecimen, compose_rgba, render_specimen


PROTOTYPE_SEED = 0x4D4F5250


def prototype_specimens(count_per_family: int = 6) -> list[MorphologySpecimen]:
    if count_per_family < 1:
        raise ValueError("count_per_family must be positive")
    specimens: list[MorphologySpecimen] = []
    for family_index, family in enumerate(FAMILIES):
        for variant in range(count_per_family):
            seed = (
                PROTOTYPE_SEED
                + family_index * 0x01010101
                + variant * 0x00019E37
            ) & 0xFFFFFFFF
            specimen = render_specimen(genome_from_seed(seed, family))
            assert_valid_specimen(specimen)
            specimens.append(specimen)
    return specimens


def role_matrix_specimens(seed: int = PROTOTYPE_SEED ^ 0x524F4C45) -> list[MorphologySpecimen]:
    """Hold identity constant per family while sweeping every combat role."""
    specimens: list[MorphologySpecimen] = []
    for family_index, family in enumerate(FAMILIES):
        base_seed = (int(seed) + family_index * 0x01010101) & 0xFFFFFFFF
        base = genome_from_seed(base_seed, family)
        for role_id in range(len(ROLE_NAMES)):
            specimen = render_specimen(replace(base, role_id=role_id))
            assert_valid_specimen(specimen)
            specimens.append(specimen)
    return specimens


def build_role_contact_sheet(
    specimens: Sequence[MorphologySpecimen], *, scale: int = 4
) -> Image.Image:
    expected = len(FAMILIES) * len(ROLE_NAMES)
    if len(specimens) != expected:
        raise ValueError(f"Role matrix needs {expected} specimens, got {len(specimens)}")
    if scale < 1:
        raise ValueError("scale must be positive")
    cell = CANVAS_SIZE * scale
    top = 28
    left = 74
    sheet = Image.new(
        "RGBA",
        (left + len(ROLE_NAMES) * cell, top + len(FAMILIES) * cell),
        (3, 5, 12, 255),
    )
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, top - 1), fill=(8, 14, 28, 255))
    for role_id, role in enumerate(ROLE_NAMES):
        draw.text((left + role_id * cell + 5, 8), role.upper(), fill=(89, 229, 255, 255))
    for family_id, family in enumerate(FAMILIES):
        y = top + family_id * cell
        draw.text((6, y + 8), family.upper(), fill=(208, 242, 255, 255))
        for role_id in range(len(ROLE_NAMES)):
            specimen = specimens[family_id * len(ROLE_NAMES) + role_id]
            sprite = Image.fromarray(compose_rgba(specimen)).resize(
                (cell, cell), Image.Resampling.NEAREST
            )
            x = left + role_id * cell
            sheet.alpha_composite(sprite, (x, y))
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(25, 88, 122, 255))
    return sheet


def build_contact_sheet(
    specimens: Sequence[MorphologySpecimen],
    *,
    columns: int = 6,
    scale: int = 4,
) -> Image.Image:
    if columns < 1 or scale < 1:
        raise ValueError("columns and scale must be positive")
    rows = (len(specimens) + columns - 1) // columns
    cell = CANVAS_SIZE * scale
    header = 24
    sheet = Image.new(
        "RGBA",
        (columns * cell, header + rows * cell),
        (3, 5, 12, 255),
    )
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, header - 1), fill=(8, 14, 28, 255))
    draw.text(
        (8, 7),
        "BROAD NEURAL MORPHOLOGY // DETERMINISTIC 48x48 SEMANTIC PROTOTYPES",
        fill=(208, 242, 255, 255),
    )
    for index, specimen in enumerate(specimens):
        rgba = compose_rgba(specimen)
        sprite = Image.fromarray(rgba).resize(
            (cell, cell), Image.Resampling.NEAREST
        )
        x = (index % columns) * cell
        y = header + (index // columns) * cell
        sheet.alpha_composite(sprite, (x, y))
        label = f"{specimen.genome.family_name.upper()} {specimen.genome.seed:08X}"
        draw.rectangle((x + 3, y + 3, x + cell - 3, y + 16), fill=(2, 4, 10, 210))
        draw.text((x + 6, y + 5), label, fill=(225, 244, 255, 255))
    return sheet


def write_prototype_outputs(
    destination: Path,
    *,
    count_per_family: int = 6,
) -> dict[str, str | int]:
    destination = Path(destination).resolve()
    specimens = prototype_specimens(count_per_family)
    budget = guard_corpus_destination(destination, len(specimens))
    destination.mkdir(parents=True, exist_ok=True)

    sheet_path = destination / "morphology_prototype_contact_sheet.png"
    manifest_path = destination / "morphology_prototype_manifest.json"
    semantic_path = destination / "morphology_prototype_semantics.npz"
    role_sheet_path = destination / "morphology_role_matrix.png"
    training = [specimen.training_fields() for specimen in specimens]
    build_contact_sheet(specimens, columns=count_per_family).save(sheet_path)
    role_specimens = role_matrix_specimens()
    build_role_contact_sheet(role_specimens).save(role_sheet_path)
    np.savez_compressed(
        semantic_path,
        layers=np.stack([specimen.layers for specimen in specimens]),
        tokens=np.stack([specimen.tokens for specimen in specimens]),
        seeds=np.asarray([specimen.genome.seed for specimen in specimens], dtype=np.uint32),
        families=np.asarray([specimen.genome.family_name for specimen in specimens]),
        layer_names=np.asarray(LAYER_NAMES),
        guide=np.stack([fields.guide for fields in training]),
        part_owner=np.stack([fields.part_owner for fields in training]),
        material=np.stack([fields.material for fields in training]),
        emission_level=np.stack([fields.emission_level for fields in training]),
        morphologies=np.asarray([fields.morphology_index for fields in training], dtype=np.uint8),
        subtypes=np.asarray([fields.subtype_id for fields in training], dtype=np.uint8),
        roles=np.asarray([fields.role_id for fields in training], dtype=np.uint8),
        genes=np.stack([fields.genes for fields in training]),
    )
    manifest_payload = {
        "format": "broad-morphology-prototype-bank-v1",
        "sprite_count": len(specimens),
        "families": list(FAMILIES),
        "count_per_family": count_per_family,
        "contact_sheet": sheet_path.name,
        "role_matrix": role_sheet_path.name,
        "semantic_archive": semantic_path.name,
        "disk_budget": budget.to_dict(),
        "sprites": [specimen.manifest for specimen in specimens],
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2), encoding="utf-8"
    )
    return {
        "sprites": len(specimens),
        "contact_sheet": str(sheet_path),
        "role_matrix": str(role_sheet_path),
        "manifest": str(manifest_path),
        "semantic_archive": str(semantic_path),
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate isolated deterministic 48x48 morphology prototypes."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=project_root / "outputs" / "morphology_prototype",
    )
    parser.add_argument("--count-per-family", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            write_prototype_outputs(
                args.destination, count_per_family=args.count_per_family
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
