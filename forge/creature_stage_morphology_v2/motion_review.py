from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np
from PIL import Image, ImageDraw

from ..creature_stage_developmental.contract import FAMILIES
from ..creature_stage_developmental.development import develop
from ..creature_stage_grounded_locomotion.contract import GroundedLocomotionConfig
from ..creature_stage_grounded_locomotion.physics import GroundedCycle, simulate_grounded_cycle
from ..safety import require_disk_floor
from .genomes import morphology_review_genomes
from .review import FAMILY_COLORS, _bounds, _canonical, _draw_cells, _draw_shadow, _draw_structure, _font, _point, _sha


FORMAT = "nullvector-creature-stage-morphology-grounded-review-v1"


def _frame(organisms, cycles: list[GroundedCycle], frame_index: int) -> Image.Image:
    width,height=1800,620; card_w=360
    image=Image.new("RGBA",(width,height),(2,7,12,255)); draw=ImageDraw.Draw(image,"RGBA")
    draw.rectangle((0,0,width,72),fill=(5,15,23,255))
    draw.text((24,14),"MORPHOLOGY V2 // GROUNDED PHYSICS PROOF",font=_font(27,True),fill=(225,243,248,255))
    draw.text((width-540,20),"ANCHOR → TETHER → MUSCLE → BODY MOTION",font=_font(16),fill=(78,226,246,255))
    for index,(organism,cycle) in enumerate(zip(organisms,cycles,strict=True)):
        family=FAMILIES[index]; accent=FAMILY_COLORS[family]; left=index*card_w; top=72
        draw.rectangle((left,top,left+card_w-1,height-1),outline=(*accent,75),width=1)
        draw.text((left+14,top+14),f"{family.upper()} // {cycle.primary_mode.upper()}",font=_font(15,True),fill=(*accent,255))
        draw.text((left+14,top+38),organism.genome.genome_id.upper(),font=_font(10),fill=(139,171,181,255))
        low,high,mid=_bounds(organism); span=float(max(high-low)); scale=min(6.0,210/max(span,1)); center=(left+card_w*.5,top+260)
        motion=cycle.frames[frame_index]
        floor_y=center[1]+float(cycle.ground_y-mid[1])*scale
        _draw_shadow(draw,center,min(100,float(high[0]-low[0])*scale*.48),floor_y+5)
        for appendage_index in np.flatnonzero(motion.contact_active):
            anchor=motion.contact_anchor_world[appendage_index].copy(); anchor[0]-=motion.body_world_x
            x,y=_point(anchor,center,mid,scale); r=5
            draw.ellipse((x-r,y-r,x+r,y+r),fill=(145,255,105,230),outline=(230,255,205,255),width=1)
            draw.line((x-9,y+7,x+9,y+7),fill=(145,255,105,160),width=2)
        _draw_cells(draw,organism,center,scale,fade=True,points=motion.cells_local)
        _draw_structure(draw,organism,center,scale,nodes=motion.nodes_local)
        active=int(np.count_nonzero(motion.contact_active)); muscles=int(np.count_nonzero(motion.muscle_activation>=.28))
        draw.text((left+14,top+480),f"CONTACTS {active:02d}   ACTIVE MUSCLES {muscles:02d}",font=_font(11),fill=(176,201,209,255))
        draw.text((left+14,top+503),f"TRAVEL {cycle.distance_px:+.2f}px   SLIP {cycle.maximum_contact_slip_px:.3f}px",font=_font(11),fill=(*accent,220))
        draw.text((left+14,top+526),"NO ROTATION · NO SPRITE FLIP",font=_font(10),fill=(102,143,155,255))
    return image.convert("RGB")


def build_motion_review(destination: Path) -> Path:
    destination=destination.resolve()
    if destination.exists(): raise FileExistsError(destination)
    require_disk_floor(destination.parent,floor_gb=100.0)
    all_genomes=morphology_review_genomes(); genomes=[all_genomes[index] for index in (0,6,12,18,24)]
    organisms=[develop(genome) for genome in genomes]
    config=GroundedLocomotionConfig(frame_count=48,settle_cycles=4,substeps=2,edge_iterations=7)
    cycles=[simulate_grounded_cycle(organism,config) for organism in organisms]
    staging=destination.parent/f".{destination.name}.tmp-{uuid.uuid4().hex}"; frames=staging/"frames"; frames.mkdir(parents=True)
    try:
        for index in range(config.frame_count): _frame(organisms,cycles,index).save(frames/f"frame_{index:03d}.png",compress_level=5)
        gif=staging/"grounded_morphology.gif"
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-framerate","12","-i",str(frames/"frame_%03d.png"),"-vf","palettegen=stats_mode=diff","-frames:v","1",str(staging/"palette.png")],check=True)
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-framerate","12","-i",str(frames/"frame_%03d.png"),"-i",str(staging/"palette.png"),"-lavfi","paletteuse=dither=bayer:bayer_scale=3","-loop","0",str(gif)],check=True)
        poster=staging/"grounded_morphology_contact.png"; _frame(organisms,cycles,12).save(poster,compress_level=7)
        shutil.rmtree(frames); (staging/"palette.png").unlink()
        records=[]
        for organism,cycle in zip(organisms,cycles,strict=True):
            records.append({"genome_id":organism.genome.genome_id,"family":FAMILIES[int(np.argmax(organism.genome.family_mix))],"primary_mode":cycle.primary_mode,"distance_px":round(cycle.distance_px,7),"loop_seam_max_abs":round(cycle.loop_seam_max_abs,7),"maximum_edge_strain":round(cycle.maximum_edge_strain,7),"maximum_contact_slip_px":round(cycle.maximum_contact_slip_px,7),"vertical_axis_max_degrees":round(cycle.vertical_axis_max_degrees,7),"cycle_sha256":cycle.identity_sha256})
        payload={"format":FORMAT,"config":config.to_dict(),"records":records,"artifacts":{"grounded_morphology.gif":{"sha256":_sha(gif),"bytes":gif.stat().st_size},"grounded_morphology_contact.png":{"sha256":_sha(poster),"bytes":poster.stat().st_size}},"gates":{"all_families":len(records)==5,"all_finite":all(np.isfinite([r["distance_px"],r["loop_seam_max_abs"],r["maximum_edge_strain"],r["maximum_contact_slip_px"],r["vertical_axis_max_degrees"]]).all() for r in records),"all_vertical_under_12_degrees":all(r["vertical_axis_max_degrees"]<12 for r in records)},"status":"human_review_required"}
        payload["semantic_sha256"]=hashlib.sha256(b"nullvector-morphology-motion-review\0"+_canonical(payload)).hexdigest()
        (staging/"motion_manifest.json").write_bytes(_canonical(payload)); staging.replace(destination)
    except BaseException:
        if staging.exists(): shutil.rmtree(staging)
        raise
    return destination/"motion_manifest.json"
