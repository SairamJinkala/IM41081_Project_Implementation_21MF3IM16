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
from paperindu.visualization import render_animation, render_dashboard


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run and visualize digital twin simulation.")
    p.add_argument("--output-dir", default=os.path.join(ROOT, "results"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trace-points", type=int, default=6000)
    p.add_argument("--quality-points", type=int, default=1800)
    p.add_argument("--frame-stride", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        random_seed=args.seed,
        num_trace_points=args.trace_points,
        num_quality_points=args.quality_points,
    )

    metrics = run_pipeline(cfg, args.output_dir)
    dashboard = render_dashboard(args.output_dir)
    animation = render_animation(args.output_dir, frame_stride=args.frame_stride)

    print(json.dumps({
        "metrics": metrics,
        "dashboard": str(dashboard),
        "animation": str(animation),
    }, indent=2))


if __name__ == "__main__":
    main()
