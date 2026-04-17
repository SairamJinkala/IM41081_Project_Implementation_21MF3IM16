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
from paperindu.physical_twin import render_physical_twin_simulation
from paperindu.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run physical + digital twin machining simulation.")
    p.add_argument("--output-dir", default=os.path.join(ROOT, "results"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trace-points", type=int, default=6000)
    p.add_argument("--quality-points", type=int, default=1800)
    p.add_argument("--frames", type=int, default=80)
    p.add_argument("--grid-nx", type=int, default=75)
    p.add_argument("--grid-ny", type=int, default=45)
    p.add_argument("--tool-radius", type=float, default=1.8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        random_seed=args.seed,
        num_trace_points=args.trace_points,
        num_quality_points=args.quality_points,
        cutter_radius=args.tool_radius,
    )

    metrics = run_pipeline(cfg, args.output_dir)
    gif_path, png_path = render_physical_twin_simulation(
        result_dir=args.output_dir,
        n_frames=args.frames,
        grid_nx=args.grid_nx,
        grid_ny=args.grid_ny,
        tool_radius=args.tool_radius,
        seed=args.seed,
    )

    print(
        json.dumps(
            {
                "metrics": metrics,
                "physical_twin_gif": str(gif_path),
                "physical_twin_final_png": str(png_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
