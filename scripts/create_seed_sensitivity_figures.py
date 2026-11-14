"""Create sensitivity plots grouped by seed for the Scene 3 pipeline."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOTS = [ROOT / "results" / "final" / "easy_to_hard_scene_rotations", ROOT / "results" / "final" / "easy_to_hard_scene_rotations_all_seeds"]
BASELINE = ROOT / "results" / "baselines_python_full_corrected" / "metrics" / "per_image.csv"
OUT = ROOT / "results" / "figures" / "sensitivity_by_seed_scene3"

def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def mean_by(records, field):
    grouped = {}
    for r in records:
        grouped.setdefault(int(r[field]), []).append(float(r["q_seg"]))
    return [(x, float(np.mean(grouped[x]))) for x in sorted(grouped)]

def main():
    evolved = []
    for root in RUN_ROOTS:
        for path in sorted(root.glob("easy_to_medium_hard_train3_seed*/metrics/full_grid.csv")):
            run = path.parts[-3]
            seed = run.rsplit("_seed", 1)[1]
            evolved.extend({**r, "group": f"Seed {seed}"} for r in read(path))
    baseline = []
    for r in read(BASELINE):
        baseline.append({"group": r["method"].capitalize(), "shading": r["shading"], "noise": r["noise"], "q_seg": r["q_seg"]})

    OUT.mkdir(parents=True, exist_ok=True)
    colours = {"Otsu": "#1f77b4", "Sobel": "#d62728", "Seed 7": "#2ca02c", "Seed 123": "#ff7f0e", "Seed 31415": "#9467bd"}
    for field, label, filename in [("noise", "Noise level", "qseg_by_noise_seed_scene3.png"), ("shading", "Shading level", "qseg_by_shading_seed_scene3.png")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for group in ["Otsu", "Sobel", "Seed 7", "Seed 123", "Seed 31415"]:
            source = baseline if group in ("Otsu", "Sobel") else evolved
            data = mean_by([r for r in source if r["group"] == group], field)
            xx = [a for a, _ in data]; yy = [b for _, b in data]
            ax.plot(xx, yy, marker="o", linewidth=2, markersize=4, label=group, color=colours[group])
        ax.set_xlabel(label); ax.set_ylabel(r"Mean $Q_{\mathrm{seg}}$")
        ax.set_title(f"Scene 3 sensitivity to {label.lower()} by seed")
        ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(OUT / filename, dpi=180); plt.close(fig)
    print(OUT)

if __name__ == "__main__":
    main()
