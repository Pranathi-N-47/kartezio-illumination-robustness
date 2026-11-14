from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "hardware_items" / "data"
OUT = ROOT / "results" / "dataset_overview.png"

scenes = [1, 2, 3, 4]
conditions = [(1, 1), (1, 8), (7, 1), (7, 8), (13, 1), (13, 14)]
fig, axes = plt.subplots(len(scenes), len(conditions), figsize=(18, 11))

for row, scene in enumerate(scenes):
    for col, (shading, noise) in enumerate(conditions):
        path = DATA / f"benchdata_{scene:02d}_{shading:02d}_{noise:02d}.tif"
        image = np.asarray(Image.open(path))
        ax = axes[row, col]
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        if row == 0:
            ax.set_title(f"b={shading}, n={noise}", fontsize=11)
    axes[row, 0].set_ylabel(f"Scene {scene}", fontsize=12)

fig.suptitle(
    "Hardware-items benchmark: representative shading and Gaussian-noise conditions",
    fontsize=16,
    y=0.98,
)
fig.text(
    0.5, 0.015,
    "Columns: bright/low noise → bright/high noise → medium shading → "
    "medium shading/high noise → dark/low noise → dark/high noise",
    ha="center", fontsize=10,
)
plt.tight_layout(rect=(0, 0.04, 1, 0.95))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(OUT)
