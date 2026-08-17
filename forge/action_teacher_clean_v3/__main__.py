from __future__ import annotations

import argparse
import json

from .contract import DEFAULT_ROOT
from .curriculum import generate


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",default=DEFAULT_ROOT);parser.add_argument("--session",required=True);parser.add_argument("--repeats",type=int,default=6);parser.add_argument("--seed",type=lambda value:int(value,0),default=0x434C45414E5633);parser.add_argument("--device",default="cuda");args=parser.parse_args();print(json.dumps(generate(root=args.root,session_id=args.session,repeats=args.repeats,seed=args.seed,device=args.device),indent=2))
