"""Analyze evolved-pipeline performance by shading and noise level."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict], key_fields: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        result = dict(zip(key_fields, key))
        for metric in ("dice", "iou", "q_seg", "custom_count_error"):
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            result[f"mean_{metric}"] = float(values.mean())
            result[f"std_{metric}"] = float(values.std(ddof=0))
        result["images"] = len(group)
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot_metric(rows: list[dict], x_field: str, output: Path, title: str) -> None:
    plt.figure(figsize=(10, 6))
    for run_id in sorted({row["run_id"] for row in rows}):
        subset = [row for row in rows if row["run_id"] == run_id]
        subset.sort(key=lambda row: int(row[x_field]))
        plt.plot([int(row[x_field]) for row in subset], [float(row["mean_q_seg"]) for row in subset], marker="o", label=run_id)
    plt.xlabel("Shading level" if x_field == "shading" else "Noise level")
    plt.ylabel("Mean Q_seg")
    plt.title(title); plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True); plt.savefig(output, dpi=160); plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.results_root / "sensitivity_analysis"
    all_rows = []
    for csv_path in sorted(args.results_root.glob("easy_to_medium_hard_train*_seed*/metrics/full_grid.csv")):
        all_rows.extend(read_rows(csv_path))
    if not all_rows:
        raise SystemExit("No full_grid.csv files found")
    by_shading = summarize(all_rows, ["run_id", "shading"])
    by_noise = summarize(all_rows, ["run_id", "noise"])
    by_scene_shading = summarize(all_rows, ["run_id", "scene", "shading"])
    by_scene_noise = summarize(all_rows, ["run_id", "scene", "noise"])
    write_csv(output / "by_shading.csv", by_shading)
    write_csv(output / "by_noise.csv", by_noise)
    write_csv(output / "by_scene_shading.csv", by_scene_shading)
    write_csv(output / "by_scene_noise.csv", by_scene_noise)
    plot_metric(by_shading, "shading", output / "qseg_by_shading.png", "Pipeline sensitivity to shading")
    plot_metric(by_noise, "noise", output / "qseg_by_noise.png", "Pipeline sensitivity to noise")
    summary = {"source_images": len(all_rows), "runs": sorted({row["run_id"] for row in all_rows}),
               "outputs": ["by_shading.csv", "by_noise.csv", "by_scene_shading.csv", "by_scene_noise.csv", "qseg_by_shading.png", "qseg_by_noise.png"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Sensitivity analysis saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
