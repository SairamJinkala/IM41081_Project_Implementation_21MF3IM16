#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from paperindu.config import PipelineConfig
from paperindu.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run hybrid-learning DT pipeline demo.")
    p.add_argument("--output-dir", default=os.path.join(ROOT, "results"), help="Result directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--trace-points", type=int, default=6000, help="NC trace sample points")
    p.add_argument("--quality-points", type=int, default=1800, help="Quality path points")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        random_seed=args.seed,
        num_trace_points=args.trace_points,
        num_quality_points=args.quality_points,
    )
    metrics = run_pipeline(cfg, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
