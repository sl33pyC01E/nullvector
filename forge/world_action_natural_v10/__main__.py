from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import DEFAULT_OUTPUT,build,validate

parser=argparse.ArgumentParser();parser.add_argument("command",choices=("build","validate"));parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args();print(json.dumps(build(args.output) if args.command=="build" else validate(args.output),indent=2))
