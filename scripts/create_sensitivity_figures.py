"""Create compact noise/shading sensitivity figures for the main comparison."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVOLVED_ROOTS = [
    ROOT / "results" / "final" / "easy_to_hard_scene_rotations",
    ROOT / "results" / "final" / "easy_to_hard_scene_rotations_all_seeds",
]
BASELINE = ROOT / "results" / "baselines_python_full_corrected" / "metrics" / "per_image.csv"
OUT = ROOT / "results" / "figures" / "sensitivity_main_comparison"

def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def aggregate(records, x):
    values = {}
    for r in records:
        values.setdefault(int(r[x]), []).append(float(r["q_seg"]))
    return [(k, float(np.mean(values[k])), float(np.std(values[k]))) for k in sorted(values)]

def main():
    evolved = []
    for root in EVOLVED_ROOTS:
        for path in sorted(root.glob("easy_to_medium_hard_train*_seed*/metrics/full_grid.csv")):
            run = path.parts[-3]
            seed = run.rsplit("_seed", 1)[1]
            evolved.extend({**r, "group": f"Evolved (seed {seed})"} for r in rows(path))

    baseline = []
    for r in rows(BASELINE):
        baseline.append({"group": r["method"].capitalize(), "shading": r["shading"], "noise": r["noise"], "q_seg": r["q_seg"]})

    OUT.mkdir(parents=True, exist_ok=True)
    for x, label, filename in [("noise", "Noise level", "qseg_by_noise.png"), ("shading", "Shading level", "qseg_by_shading.png")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for group, colour in [("Otsu", "#1f77b4"), ("Sobel", "#d62728"), ("Evolved (seed 7)", "#2ca02c"), ("Evolved (seed 123)", "#ff7f0e"), ("Evolved (seed 31415)", "#9467bd")]:
            source = evolved if group.startswith("Evolved") else baseline
            data = aggregate([r for r in source if r["group"] == group], x)
            if not data:
                continue
            xx = [a for a, _, _ in data]; mean = np.array([b for _, b, _ in data]); std = np.array([c for _, _, c in data])
            ax.plot(xx, mean, marker="o", linewidth=2, markersize=4, label=group, color=colour)
            # Keep the main presentation figures uncluttered; variability is
            # retained in the underlying per-level data and seed summaries.
        ax.set_xlabel(label); ax.set_ylabel(r"Mean $Q_{\mathrm{seg}}$")
        ax.set_title(f"Segmentation sensitivity to {label.lower()}")
        ax.grid(alpha=0.3); ax.set_ylim(0, 1); ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(OUT / filename, dpi=180); plt.close(fig)
    print(OUT)

if __name__ == "__main__":
    main()
