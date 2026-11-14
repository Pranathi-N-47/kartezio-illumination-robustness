"""Create sensitivity plots grouped by training scene (seed 7)."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "final" / "easy_to_hard_scene_rotations"
BASELINE = ROOT / "results" / "baselines_python_full_corrected" / "metrics" / "per_image.csv"
OUT = ROOT / "results" / "figures" / "sensitivity_by_training_scene"

def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def grouped(records, field):
    groups = {}
    for r in records:
        groups.setdefault(int(r[field]), []).append(float(r["q_seg"]))
    return [(k, float(np.mean(groups[k]))) for k in sorted(groups)]

def main():
    evolved = []
    for path in sorted(RUN_ROOT.glob("easy_to_medium_hard_train*_seed7/metrics/full_grid.csv")):
        run = path.parts[-3]
        scene = run.split("_train", 1)[1].split("_", 1)[0]
        evolved.extend({**r, "group": f"Trained on scene {scene}"} for r in read(path))

    baseline = []
    for r in read(BASELINE):
        baseline.append({"group": r["method"].capitalize(), "shading": r["shading"], "noise": r["noise"], "q_seg": r["q_seg"]})

    OUT.mkdir(parents=True, exist_ok=True)
    colours = {"Otsu": "#1f77b4", "Sobel": "#d62728", "Trained on scene 1": "#2ca02c", "Trained on scene 2": "#ff7f0e", "Trained on scene 3": "#9467bd", "Trained on scene 4": "#8c564b"}
    for field, label, filename in [("noise", "Noise level", "qseg_by_noise_training_scene.png"), ("shading", "Shading level", "qseg_by_shading_training_scene.png")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for group in ["Otsu", "Sobel", "Trained on scene 1", "Trained on scene 2", "Trained on scene 3", "Trained on scene 4"]:
            source = baseline if group in ("Otsu", "Sobel") else evolved
            data = grouped([r for r in source if r["group"] == group], field)
            xx = [a for a, _ in data]; yy = [b for _, b in data]
            ax.plot(xx, yy, marker="o", linewidth=2, markersize=4, label=group, color=colours[group])
        ax.set_xlabel(label); ax.set_ylabel(r"Mean $Q_{\mathrm{seg}}$")
        ax.set_title(f"Segmentation sensitivity to {label.lower()} by training scene")
        ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(OUT / filename, dpi=180); plt.close(fig)
    print(OUT)

if __name__ == "__main__":
    main()
