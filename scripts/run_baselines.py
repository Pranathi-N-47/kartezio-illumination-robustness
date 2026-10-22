"""Run the MATLAB-compatible segmentation baselines from the terminal."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MATLAB_FUNCTION_DIR = REPO_ROOT / "projects" / "illumination_robustness" / "evaluation"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run segmentation-only Otsu/Sobel baselines through MATLAB in batch mode."
    )
    parser.add_argument("--mode", choices=["smoke", "series", "full"], default="full")
    parser.add_argument("--benchmark-root", type=Path, default=REPO_ROOT / "hardware_items")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--matlab", default="matlab", help="MATLAB executable name or full path")
    args = parser.parse_args()

    matlab_dir = str(MATLAB_FUNCTION_DIR.resolve())
    # Put this repository's evaluation directory first and clear any cached or
    # previously resolved baselines function. This prevents MATLAB from using a
    # same-named function elsewhere on the MATLAB path.
    command = [
        args.matlab,
        "-batch",
        "clear baselines; addpath('" + matlab_dir.replace("'", "''") + "','-begin'); run_baselines_terminal",
    ]
    print("Running MATLAB baseline reference:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["KARTEZIO_BASELINE_MODE"] = args.mode
    environment["KARTEZIO_BENCHMARK_ROOT"] = str(args.benchmark_root.resolve())
    environment["KARTEZIO_BASELINE_OUTPUT"] = "" if args.output_dir is None else str(args.output_dir.resolve())
    try:
        completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
    except FileNotFoundError:
        print("MATLAB was not found. Install MATLAB or pass --matlab <path>.")
        return 2
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
