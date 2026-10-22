"""Run the MATLAB-compatible Python segmentation baselines."""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.illumination_robustness.data import benchmark_info as info
from projects.illumination_robustness.data.loader import BenchmarkDataset
from projects.illumination_robustness.evaluation.baselines_python import (
    run_otsu_matlab_compatible,
    run_sobel_matlab_compatible,
)
from projects.illumination_robustness.evaluation.evaluator import evaluate_prediction


def evaluate_one(task):
    """Evaluate both methods for one image in a worker process."""
    benchmark_root, scene, shading, noise, run_id = task
    dataset = BenchmarkDataset(Path(benchmark_root))
    sample = dataset.load_sample(scene, shading, noise)
    rows = []
    for method, function in [("otsu", run_otsu_matlab_compatible), ("sobel", run_sobel_matlab_compatible)]:
        prediction = function(sample.image, ground_truth=sample.mask)
        result = evaluate_prediction(prediction, sample.ground_truth, sample.image_id, shading, noise)
        rows.append(result.to_csv_row(run_id=run_id, method=method, pipeline_id=f"python_matlab_compatible_{method}"))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "series", "full"], default="full")
    parser.add_argument("--benchmark-root", type=Path, default=REPO_ROOT / "hardware_items")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    run_id = f"baselines_python_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or REPO_ROOT / "results" / run_id
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        tuples = [(1, 1, 1), (1, 7, 7), (1, 13, 14)]
    elif args.mode == "series":
        # Paper robustness series: 13 adjacent condition pairs for each scene.
        tuples = [(scene, b, b + 1) for scene in info.SCENES for b in range(1, 14)]
    else:
        tuples = [(r, b, n) for r in info.SCENES for b in info.SHADING_LEVELS for n in info.NOISE_LEVELS]
    dataset = BenchmarkDataset(args.benchmark_root)
    rows = []
    started = time.perf_counter()
    tasks = [(str(args.benchmark_root), scene, shading, noise, run_id) for scene, shading, noise in tuples]
    completed = 0
    ordered_rows = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_one, task): index for index, task in enumerate(tasks)}
        for future in as_completed(futures):
            index = futures[future]
            ordered_rows[index] = future.result()
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(tasks):
                print(f"Processed {completed}/{len(tasks)} images with {args.workers} workers", flush=True)
    for result_rows in ordered_rows:
        rows.extend(result_rows)
    with (metrics_dir / "per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {metrics_dir / 'per_image.csv'}")
    print(f"Elapsed: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
