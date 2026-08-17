from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from .contract import ACTIONS
from .recorder import validate_trajectory


def render(root: Path):
    root=Path(root);manifest=validate_trajectory(root)
    with np.load(root/manifest["artifact"]["path"],allow_pickle=False) as archive:frames=archive["frame"].copy();actions=archive["action"].copy()
    selected=[]
    for index,name in enumerate(ACTIONS):
        hits=np.flatnonzero(actions==index)
        if hits.size:selected.append((name,int(hits[len(hits)//2])))
    columns=6;cell_w,cell_h=256,278;rows=(len(selected)+columns-1)//columns;sheet=Image.new("RGB",(columns*cell_w,rows*cell_h),(3,10,15));draw=ImageDraw.Draw(sheet)
    for ordinal,(name,index) in enumerate(selected):
        x=(ordinal%columns)*cell_w;y=(ordinal//columns)*cell_h;sheet.paste(Image.fromarray(frames[index]),(x,y+22));draw.text((x+6,y+5),f"{name.upper()} // F{index:04d}",fill=(74,236,244))
    contact=root/"natural_action_contact_sheet.png";sheet.save(contact,optimize=True)
    gif=root/"natural_play_continuous.gif";sampled=np.ascontiguousarray(frames[::2]);command=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","256x256","-r","15","-i","-","-filter_complex","[0:v]scale=512:512:flags=neighbor,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=none","-loop","0",str(gif)];result=subprocess.run(command,input=sampled.tobytes(),capture_output=True,check=False)
    if result.returncode:raise RuntimeError(result.stderr.decode("utf-8",errors="replace"))
    report={"format":"nullvector-natural-play-showcase/1.0.0","frames":len(frames),"gif_frames":len(sampled),"contact_sheet":contact.name,"gif":gif.name};(root/"showcase_report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8");return report
