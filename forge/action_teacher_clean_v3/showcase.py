from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from ..action_teacher_v1.contract import ACTIONS
from .recorder import validate_trajectory


def render(session:Path,output:Path|None=None):
    session=Path(session).resolve();manifest=validate_trajectory(session)
    with np.load(session/manifest["artifact"]["path"],allow_pickle=False) as archive:frames=archive["frame"].copy();actions=archive["action"].copy()
    selected=[]
    for index,name in enumerate(ACTIONS):
        matches=np.flatnonzero(actions==index)
        if len(matches):selected.append((name,int(matches[len(matches)//2])))
    columns=6;cell_w,cell_h=256,282;rows=(len(selected)+columns-1)//columns;sheet=Image.new("RGB",(columns*cell_w,rows*cell_h),(3,9,13));draw=ImageDraw.Draw(sheet)
    for item,(name,index) in enumerate(selected):
        x=(item%columns)*cell_w;y=(item//columns)*cell_h;sheet.paste(Image.fromarray(frames[index]),(x,y+22));draw.text((x+7,y+5),f"{name.upper()} // F{index:03}",fill=(86,230,255))
    destination=session/"clean_action_contact_sheet.png" if output is None else Path(output).resolve();destination.parent.mkdir(parents=True,exist_ok=True);sheet.save(destination,optimize=True);return destination


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("session",type=Path);parser.add_argument("--output",type=Path);args=parser.parse_args();print(render(args.session,args.output))
