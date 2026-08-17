from pathlib import Path
import argparse
from .training import train
p=argparse.ArgumentParser();p.add_argument("output",type=Path);p.add_argument("--updates",type=int);p.add_argument("--device",default="cuda");a=p.parse_args();r=train(a.output,updates=a.updates,device=a.device);print(r["status"],r["metrics"])
