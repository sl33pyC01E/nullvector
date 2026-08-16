from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from ..creature_stage_developmental import develop
from ..creature_stage_morphology_v2 import morphology_review_genomes
from .contract import FORMAT,Grounded25DConfig
from .physics import Grounded25DRollout,simulate_25d


COLORS=((55,221,255),(255,82,164),(136,255,84),(188,111,255),(255,181,54))


def _render(organisms,rollouts)->Image.Image:
    width,height=1200,5*220
    image=Image.new("RGB",(width,height),(5,13,18));draw=ImageDraw.Draw(image)
    for family in range(5):
        organism=organisms[family*6];rollout=rollouts[family*6];color=COLORS[family];top=family*220
        draw.text((18,top+12),organism.genome.genome_id,fill=color)
        samples=(0,30,60,90,119)
        for column,frame_index in enumerate(samples):
            frame=rollout.frames[frame_index];cx=160+column*205;cy=top+110
            draw.ellipse((cx-35,cy+72,cx+35,cy+82),fill=(0,0,0))
            for edge in organism.skeleton_edges:
                a=frame.nodes_local[int(edge[0])];b=frame.nodes_local[int(edge[1])]
                draw.line((cx+a[0]*3,cy+a[1]*3,cx+b[0]*3,cy+b[1]*3),fill=color,width=2)
            for index in np.flatnonzero(frame.contact_active):
                terminal=np.flatnonzero(organism.skeleton_edge_appendage==index)[-1];node=int(organism.skeleton_edges[terminal,1]);point=frame.nodes_local[node]
                draw.ellipse((cx+point[0]*3-3,cy+point[1]*3-2,cx+point[0]*3+3,cy+point[1]*3+2),fill=(230,255,230))
            draw.text((cx-60,top+193),f"xy {frame.ground_position[0]:+.1f},{frame.ground_position[1]:+.1f}",fill=(130,160,170))
    return image


def build_review(output:Path,*,config:Grounded25DConfig|None=None)->dict[str,object]:
    config=config or Grounded25DConfig();output=Path(output)
    organisms=tuple(develop(g) for g in morphology_review_genomes());rollouts=tuple(simulate_25d(o,config) for o in organisms)
    modes=[r.primary_mode for r in rollouts]
    gates={
        "all_families":len(rollouts)==30,
        "lateral_motion":min(r.lateral_extent for r in rollouts if r.primary_mode!="passive")>.15,
        "depth_motion":min(r.depth_extent for r in rollouts if r.primary_mode!="passive")>.15,
        "appendages_articulate":min(r.appendage_motion_px for r in rollouts if r.primary_mode in {"step","drag","wheel"})>.15,
        "contacts_switch":min(r.contact_switches for r in rollouts if r.primary_mode in {"step","drag","wheel"})>=4,
        "chassis_upright":max(r.maximum_chassis_tilt_degrees for r in rollouts)<8,
        "tethers_hold":max(r.maximum_edge_strain for r in rollouts)<.40,
        "finite":all(np.isfinite(r.path_length) and all(np.isfinite(f.nodes_local).all() for f in r.frames) for r in rollouts),
        "family_modes":all(name in modes for name in ("step","drag","float","wheel")),
    }
    output.mkdir(parents=True,exist_ok=True)
    sheet=_render(organisms,rollouts);sheet_path=output/"grounded_25d_contact_sheet.png";sheet.save(sheet_path)
    report={"format":FORMAT,"scope":{"identities":30,"families":5,"frames":config.frames,"physics":"persistent_ground_plane_contacts","neural_gate":"teacher_target_pending"},"metrics":{"max_contact_slip":max(r.maximum_contact_slip for r in rollouts),"max_edge_strain":max(r.maximum_edge_strain for r in rollouts),"max_chassis_tilt_degrees":max(r.maximum_chassis_tilt_degrees for r in rollouts),"min_depth_extent":min(r.depth_extent for r in rollouts if r.primary_mode!="passive"),"min_lateral_extent":min(r.lateral_extent for r in rollouts if r.primary_mode!="passive"),"min_appendage_motion":min(r.appendage_motion_px for r in rollouts if r.primary_mode in {"step","drag","wheel"})},"modes":modes,"gates":gates,"passed":all(gates.values()),"contact_sheet":{"path":sheet_path.name,"sha256":hashlib.sha256(sheet_path.read_bytes()).hexdigest()},"rollout_sha256":[r.identity_sha256 for r in rollouts]}
    payload=json.dumps(report,sort_keys=True,separators=(",",":"),indent=2).encode()+b"\n";(output/"review.json").write_bytes(payload)
    return report
