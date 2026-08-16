from __future__ import annotations

import argparse
from pathlib import Path

from .validation import run_long_horizon


def main() -> None:
    parser=argparse.ArgumentParser(description="Validate the multigenerational Nullvector nature simulation")
    parser.add_argument("--seed",type=int,default=0x4E4154555245)
    parser.add_argument("--steps",type=int,default=1200)
    parser.add_argument("--output",type=Path,default=Path("outputs/nature_sim_v2/validation.json"))
    args=parser.parse_args()
    report=run_long_horizon(seed=args.seed,steps=args.steps,output=args.output)
    print(f"NATURE_SIM_V2 {'PASS' if report['passed'] else 'FAIL'} population={report['final']['population']} births={report['final']['births']} colonies={report['final']['colony_count']}")


if __name__=="__main__":main()

