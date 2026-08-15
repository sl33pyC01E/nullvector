from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import calibrate


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);parser.add_argument("--steps",type=int,default=600);parser.add_argument("--batch-size",type=int,default=8);args=parser.parse_args();print(calibrate(args.output,args.steps,args.batch_size))


if __name__=="__main__":main()
