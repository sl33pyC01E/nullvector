from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..creature_stage_developmental.contract import FAMILIES, TISSUES
from ..creature_stage_developmental.development import DevelopedOrganism, develop
from ..safety import require_disk_floor
from .genomes import morphology_review_genomes


FORMAT = "nullvector-creature-stage-morphology-review-v2"
SOURCE_FILES = (
    "forge/creature_stage_morphology_v2/__init__.py",
    "forge/creature_stage_morphology_v2/__main__.py",
    "forge/creature_stage_morphology_v2/genomes.py",
    "forge/creature_stage_morphology_v2/motion_review.py",
    "forge/creature_stage_morphology_v2/review.py",
)
COLORS = {
    "skin": (66, 203, 230), "bone": (235, 229, 194), "muscle": (239, 79, 102),
    "tendon": (244, 159, 78), "armor": (135, 159, 185), "neural": (246, 82, 197),
    "vascular": (214, 49, 71), "respiratory": (70, 231, 210), "digestive": (232, 185, 55),
    "sensor": (255, 242, 130), "storage": (175, 117, 235), "root": (105, 224, 82),
    "phase": (179, 79, 246), "machine": (235, 99, 70), "weapon": (255, 75, 82),
}
FAMILY_COLORS = {
    "humanoid": (55, 224, 247), "animalian": (255, 91, 187), "plantlike": (151, 244, 74),
    "anomaly": (189, 106, 255), "machine": (255, 178, 56),
}
REQUIRED_ORGANS = {
    "humanoid": {"heart", "gut", "lung", "brain", "eye"},
    "animalian": {"heart", "gut", "lung", "brain", "eye", "jaw"},
    "plantlike": {"vascular", "bulb", "photoreceptor", "meristem"},
    "anomaly": {"phase_brain", "singularity", "transmuter"},
    "machine": {"processor", "battery", "optic", "coolant_pump"},
}


def _font(size: int, bold: bool = False):
    names = ("consolab.ttf", "consola.ttf") if bold else ("consola.ttf", "arial.ttf")
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256(b"nullvector-morphology-v2-source\0")
    for relative in SOURCE_FILES:
        digest.update(relative.encode("utf-8") + b"\0" + (root / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def _bounds(organism: DevelopedOrganism) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = organism.cell_xy.min(axis=0).astype(np.float32)
    high = organism.cell_xy.max(axis=0).astype(np.float32)
    return low, high, (low + high) * .5


def _symmetry(organism: DevelopedOrganism) -> float:
    cells = {(int(x), int(y)) for x, y in organism.cell_xy}
    if not cells:
        return 0.0
    return sum((-x, y) in cells for x, y in cells) / len(cells)


def _point(point: np.ndarray, center: tuple[float, float], midpoint: np.ndarray, scale: float) -> tuple[float, float]:
    return center[0] + float(point[0] - midpoint[0]) * scale, center[1] + float(point[1] - midpoint[1]) * scale


def _draw_shadow(draw: ImageDraw.ImageDraw, center: tuple[float,float], width: float, y: float) -> None:
    draw.ellipse((center[0]-width, y-6, center[0]+width, y+7), fill=(20, 101, 111, 42), outline=(39, 153, 165, 74), width=1)


def _draw_cells(draw: ImageDraw.ImageDraw, organism: DevelopedOrganism, center: tuple[float,float], scale: float, *, fade: bool = False, points: np.ndarray | None = None) -> None:
    _, _, midpoint = _bounds(organism)
    radius = max(1.05, scale * .39)
    coordinates = organism.cell_xy if points is None else points
    order = np.argsort(coordinates[:, 1])
    for index in order:
        x, y = _point(coordinates[index].astype(np.float32), center, midpoint, scale)
        tissue = TISSUES[int(organism.tissue[index])]
        color = COLORS[tissue]
        alpha = 90 if fade and tissue in {"skin","root","phase","machine","armor"} else 238
        draw.rectangle((x-radius,y-radius,x+radius,y+radius), fill=(*color,alpha))
        if tissue in {"sensor","neural"}:
            draw.point((round(x),round(y)), fill=(255,255,235,255))


def _draw_structure(draw: ImageDraw.ImageDraw, organism: DevelopedOrganism, center: tuple[float,float], scale: float, *, nodes: np.ndarray | None = None) -> None:
    _, _, midpoint = _bounds(organism)
    coordinates = organism.skeleton_nodes if nodes is None else nodes
    for edge in organism.skeleton_edges:
        a = _point(coordinates[int(edge[0]),:2],center,midpoint,scale)
        b = _point(coordinates[int(edge[1]),:2],center,midpoint,scale)
        draw.line((*a,*b),fill=(245,235,193,235),width=max(1,round(scale*.42)))
    for idx, muscle in enumerate(organism.muscles):
        a = _point(coordinates[int(muscle[0]),:2],center,midpoint,scale)
        b = _point(coordinates[int(muscle[1]),:2],center,midpoint,scale)
        color = (255,70,105,180) if idx % 2 == 0 else (57,220,251,180)
        draw.line((*a,*b),fill=color,width=max(1,round(scale*.20)))
    for node in coordinates:
        x,y = _point(node[:2],center,midpoint,scale)
        r=max(1.4,scale*.34)
        draw.ellipse((x-r,y-r,x+r,y+r),fill=(250,240,202,255))


def _validate(organism: DevelopedOrganism, family: str) -> dict[str, object]:
    symmetry = _symmetry(organism)
    if symmetry < .90:
        raise ValueError(f"{organism.genome.genome_id}: raster symmetry {symmetry:.3f}")
    organs = {component.organ for component in organism.genome.components if component.organ != "none"}
    missing = REQUIRED_ORGANS[family] - organs
    if missing:
        raise ValueError(f"{organism.genome.genome_id}: missing organs {sorted(missing)}")
    for appendage in organism.genome.appendages:
        if appendage.side == 0:
            if abs(appendage.root_offset[0]) > 1e-6 or abs(appendage.endpoint[0]) > 1e-6:
                raise ValueError(f"{appendage.appendage_id}: center appendage is asymmetric")
            continue
        partner = next(item for item in organism.genome.appendages if item.appendage_id == appendage.paired_with)
        if not np.allclose(partner.root_offset,(-appendage.root_offset[0],appendage.root_offset[1]),atol=1e-6):
            raise ValueError(f"{appendage.appendage_id}: root mirror drift")
        if not np.allclose(partner.endpoint,(-appendage.endpoint[0],appendage.endpoint[1]),atol=1e-6):
            raise ValueError(f"{appendage.appendage_id}: endpoint mirror drift")
    low,high,_ = _bounds(organism)
    return {
        "genome_id": organism.genome.genome_id,
        "family": family,
        "seed": organism.genome.seed,
        "cells": organism.cell_count,
        "nodes": int(organism.skeleton_nodes.shape[0]),
        "muscles": int(organism.muscles.shape[0]),
        "appendages": len(organism.genome.appendages),
        "organs": sorted(organs),
        "symmetry": round(symmetry,6),
        "width": int(high[0]-low[0]+1),
        "height": int(high[1]-low[1]+1),
        "identity_sha256": organism.identity_sha256,
    }


def _overview(organisms: list[DevelopedOrganism]) -> Image.Image:
    card_w, card_h = 300, 300
    width, height = card_w*6, 72+card_h*5
    image=Image.new("RGBA",(width,height),(2,7,12,255)); draw=ImageDraw.Draw(image,"RGBA")
    draw.rectangle((0,0,width,72),fill=(5,15,23,255))
    draw.text((24,14),"MORPHOLOGY V2 // FAMILY-SPECIFIC BODY PLANS",font=_font(27,True),fill=(225,243,248,255))
    draw.text((width-575,20),"CELLS  /  ORGANS  /  BILATERAL LOAD GRAPHS",font=_font(16),fill=(78,226,246,255))
    for index, organism in enumerate(organisms):
        family=FAMILIES[index//6]; col=index%6; row=index//6; left=col*card_w; top=72+row*card_h
        accent=FAMILY_COLORS[family]
        draw.rectangle((left,top,left+card_w-1,top+card_h-1),outline=(*accent,75),width=1)
        draw.text((left+12,top+10),organism.genome.genome_id.replace("v2_","").upper(),font=_font(12,True),fill=(*accent,255))
        low,high,mid=_bounds(organism); span=float(max(high-low)); scale=min(5.2,178/max(span,1)); center=(left+card_w*.5,top+151)
        floor_y=center[1]+float(high[1]-mid[1])*scale+7
        _draw_shadow(draw,center,min(80,float(high[0]-low[0])*scale*.47),floor_y)
        _draw_cells(draw,organism,center,scale)
        draw.text((left+12,top+card_h-45),f"{organism.cell_count:04d} CELLS  {_symmetry(organism)*100:05.1f}% SYM",font=_font(10),fill=(157,181,190,255))
        draw.text((left+12,top+card_h-27),f"{len(organism.genome.appendages):02d} APPENDAGES  {len(organism.muscles):02d} MUSCLES",font=_font(10),fill=(112,146,158,255))
    return image.convert("RGB")


def _family_sheet(family: str, organisms: list[DevelopedOrganism]) -> Image.Image:
    card_w=600; width=card_w*3; height=960
    image=Image.new("RGBA",(width,height),(2,7,12,255)); draw=ImageDraw.Draw(image,"RGBA"); accent=FAMILY_COLORS[family]
    draw.rectangle((0,0,width,72),fill=(5,15,23,255)); draw.text((24,14),f"{family.upper()} // CELLS + ANATOMICAL CUTAWAY",font=_font(27,True),fill=(*accent,255))
    for index,organism in enumerate(organisms):
        col=index%3; row=index//3; left=col*card_w; top=72+row*444
        draw.rectangle((left,top,left+card_w-1,top+443),outline=(*accent,72),width=1)
        draw.text((left+18,top+15),organism.genome.genome_id.upper(),font=_font(14,True),fill=(222,239,244,255))
        low,high,mid=_bounds(organism); span=float(max(high-low)); scale=min(5.3,180/max(span,1)); cy=top+224
        for cx in (left+160,left+440):
            floor_y=cy+float(high[1]-mid[1])*scale+8; _draw_shadow(draw,(cx,cy),80,floor_y)
        _draw_cells(draw,organism,(left+160,cy),scale)
        _draw_cells(draw,organism,(left+440,cy),scale,fade=True); _draw_structure(draw,organism,(left+440,cy),scale)
        organs=sorted({c.organ for c in organism.genome.components if c.organ!="none"})
        draw.text((left+18,top+366),"ORGANS // "+" · ".join(organs),font=_font(10),fill=(165,193,202,255))
        draw.text((left+18,top+388),f"{organism.cell_count} CELLS  {len(organism.skeleton_nodes)} NODES  {len(organism.muscles)} MUSCLES  {_symmetry(organism)*100:.1f}% MIRRORED",font=_font(10),fill=(*accent,230))
        draw.text((left+18,top+410),"LIVING FIELD",font=_font(10),fill=(101,143,155,255)); draw.text((left+365,top+410),"LOAD + ACTUATOR GRAPH",font=_font(10),fill=(101,143,155,255))
    return image.convert("RGB")


def build_review(destination: Path) -> Path:
    destination=destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite morphology review: {destination}")
    require_disk_floor(destination.parent, floor_gb=100.0)
    genomes=morphology_review_genomes(); organisms=[develop(genome) for genome in genomes]
    records=[]
    for index,organism in enumerate(organisms): records.append(_validate(organism,FAMILIES[index//6]))
    staging=destination.parent/f".{destination.name}.tmp-{uuid.uuid4().hex}"; staging.mkdir(parents=True)
    try:
        overview=staging/"morphology_overview.png"; _overview(organisms).save(overview,compress_level=7)
        artifacts={"morphology_overview.png":{"sha256":_sha(overview),"bytes":overview.stat().st_size}}
        for family_index,family in enumerate(FAMILIES):
            path=staging/f"{family}_anatomy.png"; _family_sheet(family,organisms[family_index*6:(family_index+1)*6]).save(path,compress_level=7)
            artifacts[path.name]={"sha256":_sha(path),"bytes":path.stat().st_size}
        payload={
            "format":FORMAT,"source_sha256":source_sha256(),"count":len(records),"families":list(FAMILIES),
            "variants_per_family":6,"records":records,"artifacts":artifacts,
            "gates":{"all_develop":True,"all_required_organs":True,"all_appendages_mirrored_or_centered":True,"minimum_raster_symmetry_0_90":True},
            "status":"human_review_required",
        }
        payload["semantic_sha256"]=hashlib.sha256(b"nullvector-morphology-v2-review\0"+_canonical(payload)).hexdigest()
        (staging/"review_manifest.json").write_bytes(_canonical(payload))
        staging.replace(destination)
    except BaseException:
        if staging.exists(): shutil.rmtree(staging)
        raise
    return destination/"review_manifest.json"
