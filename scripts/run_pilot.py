#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsnlife.experiment import run_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "pilot.json"))
    parser.add_argument("--output", default=str(ROOT / "results" / "pilot_raw.csv"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    data = run_config(args.config, args.output, workers=args.workers, resume=not args.no_resume)
    print(f"wrote {len(data)} rows to {args.output}")


if __name__ == "__main__":
    main()
