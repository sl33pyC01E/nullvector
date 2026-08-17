import argparse, json
from pathlib import Path
from . import DEFAULT_ROOT, generate

parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=DEFAULT_ROOT);parser.add_argument("--session",required=True);parser.add_argument("--frames",type=int,default=240);parser.add_argument("--seed",type=lambda value:int(value,0),default=0x56494557504F5254);parser.add_argument("--device",default="cpu");parser.add_argument("--family",type=int,choices=range(5),default=1);parser.add_argument("--scenario",choices=("journey","migration","feeding","predation","injury","settlement_pan"),default="journey");args=parser.parse_args();print(json.dumps(generate(root=args.root,session_id=args.session,frames=args.frames,seed=args.seed,device=args.device,actor_family=args.family,scenario=args.scenario),indent=2))
