from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import DEFAULT_ROOT, generate


parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=DEFAULT_ROOT);parser.add_argument("--session",required=True);parser.add_argument("--frames",type=int,default=1200);parser.add_argument("--seed",type=lambda value:int(value,0),default=0x4E41545552414C34);parser.add_argument("--device",default="cuda");args=parser.parse_args();print(json.dumps(generate(root=args.root,session_id=args.session,frames=args.frames,seed=args.seed,device=args.device),indent=2))
