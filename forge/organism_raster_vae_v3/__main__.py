from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import calibrate


def main() -> None:
    parser=argparse.ArgumentParser(description="Structured 96px organism VAE v3 calibration")
    parser.add_argument("command",choices=("calibrate",)); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--steps",type=int,default=120); parser.add_argument("--batch-size",type=int,default=4)
    args=parser.parse_args(); print(calibrate(args.output,steps=args.steps,batch_size=args.batch_size))


if __name__=="__main__": main()
