from __future__ import annotations
import argparse
from pathlib import Path
from .training import train
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--updates",type=int,default=250);p.add_argument("--device",default="cuda");a=p.parse_args();print(train(a.output,updates=a.updates,device=a.device))
if __name__=="__main__":main()
