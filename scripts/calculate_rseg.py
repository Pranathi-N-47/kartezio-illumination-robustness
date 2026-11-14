"""Calculate paper-style segmentation robustness R_seg for evolved elites."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    # Support scene-rotation runs as well as all-scenes and other named runs.
    run_dirs = sorted(
        run_dir for run_dir in args.results_root.iterdir()
        if run_dir.is_dir() and (run_dir / "metrics" / "full_grid.csv").is_file()
    )
    if not run_dirs:
        raise SystemExit(f"No evolved run directories found in {args.results_root}")

    for run_dir in run_dirs:
        source = run_dir / "metrics" / "full_grid.csv"
        if not source.is_file():
            print(f"Skipping {run_dir.name}: missing {source}")
            continue
        rows = read_csv(source)
        series_rows = []
        for scene in sorted({int(row["scene"]) for row in rows}):
            scene_rows = {(int(row["shading"]), int(row["noise"])): row
                          for row in rows if int(row["scene"]) == scene}
            values = []
            for shading in range(1, 14):
                noise = shading + 1
                row = scene_rows.get((shading, noise))
                if row is None:
                    raise ValueError(f"Missing robustness condition scene={scene}, b={shading}, n={noise}")
                q = float(row["q_seg"])
                values.append(q)
                series_rows.append({"run_id": run_dir.name, "scene": scene,
                                    "shading": shading, "noise": noise,
                                    "q_seg": q})
            series_rows.append({"run_id": run_dir.name, "scene": scene,
                                "shading": "mean", "noise": "mean",
                                "q_seg": float(np.mean(values))})
        output = run_dir / "metrics" / "robustness_series.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run_id", "scene", "shading", "noise", "q_seg"])
            writer.writeheader(); writer.writerows(series_rows)
        scene_means = [float(row["q_seg"]) for row in series_rows if row["shading"] == "mean"]
        summary = {"run_id": run_dir.name, "series_images_per_scene": 13,
                   "scenes": len(scene_means), "r_seg_mean": float(np.mean(scene_means)),
                   "r_seg_std": float(np.std(scene_means)),
                   "r_seg_by_scene": {str(i + 1): value for i, value in enumerate(scene_means)}}
        (run_dir / "metrics" / "robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"{run_dir.name}: R_seg={summary['r_seg_mean']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
