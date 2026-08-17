from __future__ import annotations
import argparse,json
from pathlib import Path
from .training import train_segment

def main():
    parser=argparse.ArgumentParser();parser.add_argument("base",type=Path);parser.add_argument("output",type=Path);parser.add_argument("--updates",type=int,default=100);parser.add_argument("--device",choices=("cpu","cuda"),default="cuda");args=parser.parse_args();print(json.dumps(train_segment(args.base,args.output,updates=args.updates,device_name=args.device),indent=2,sort_keys=True))
if __name__=="__main__":main()
