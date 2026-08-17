from __future__ import annotations

import argparse
from pathlib import Path
from .training import train


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("output",type=Path); parser.add_argument("--updates",type=int)
    parser.add_argument("--device",default="cuda"); args=parser.parse_args()
    report=train(args.output,updates=args.updates,device=args.device)
    print(report["status"],report["metrics"],flush=True)


if __name__=="__main__": main()

