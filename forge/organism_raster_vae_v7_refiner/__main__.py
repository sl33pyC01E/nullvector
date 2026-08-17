from __future__ import annotations

import argparse,json
from pathlib import Path

from .cache import build
from .contract import DEFAULT_OUTPUT,Plan
from .evaluation import evaluate,validate
from .training import train


def main()->None:
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    for name in ("cache","train","evaluate","validate"):
        item=sub.add_parser(name);item.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
        if name in {"cache","evaluate"}:item.add_argument("--device",default="cuda")
    args=parser.parse_args()
    if args.command=="cache":result=build(args.output,device_name=args.device)
    elif args.command=="train":result=train(args.output,Plan())
    elif args.command=="evaluate":result=evaluate(args.output,device_name=args.device)
    else:result=validate(args.output)
    print(json.dumps(result,indent=2,default=str))


if __name__=="__main__":main()
