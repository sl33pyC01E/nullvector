from __future__ import annotations
import argparse,json,os
from pathlib import Path
from .validation import run_single_world

def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,required=True);parser.add_argument("--steps",type=int,required=True);parser.add_argument("--delta",type=float,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();result=run_single_world(seed=args.seed,steps=args.steps,delta=args.delta);args.output.parent.mkdir(parents=True,exist_ok=True);stage=args.output.with_suffix(args.output.suffix+f".tmp-{os.getpid()}");stage.write_text(json.dumps(result,sort_keys=True,separators=(",",":")),"utf-8");os.replace(stage,args.output)
if __name__=="__main__":main()
