from __future__ import annotations
import argparse,json
from .training import evaluate,train
def main():
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("train","evaluate"));args=parser.parse_args();print(json.dumps(train() if args.command=="train" else evaluate(),indent=2))
if __name__=="__main__":main()
