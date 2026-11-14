"""Create a cross-scene qualitative comparison at one difficult condition."""
from pathlib import Path
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(SRC))
from kartezio.core.components import Genotype
from kartezio.core.endpoints import EndpointThreshold
from kartezio.evolution import KartezioTrainer
from projects.illumination_robustness.data.loader import BenchmarkDataset
from projects.illumination_robustness.evolution.fitness import PilotTestQSegFitness
from projects.illumination_robustness.primitives import build_illumination_library

OUT = ROOT / "results" / "figures" / "qualitative_cross_scene_b09_n14.png"
RUN_ROOT = ROOT / "results" / "final" / "easy_to_hard_scene_rotations"

def main():
    dataset = BenchmarkDataset(ROOT / "hardware_items")
    trainers = []
    for train_scene, label in [(1, "Train scene 1"), (3, "Train scene 3"), (4, "Train scene 4")]:
        run = RUN_ROOT / f"easy_to_medium_hard_train{train_scene}_seed7"
        info = json.loads((run / "pipeline" / "elite.json").read_text(encoding="utf-8"))
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        trainer = KartezioTrainer(n_inputs=1, n_nodes=int(config["nodes"]), libraries=build_illumination_library(), endpoint=EndpointThreshold(128), fitness=PilotTestQSegFitness())
        trainers.append((label, trainer, Genotype.__from_dict__(info["genotype"])))

    fig, axes = plt.subplots(4, 5, figsize=(15, 11))
    headers = ["Input", "Ground truth", "Train scene 1", "Train scene 3", "Train scene 4"]
    for row, scene in enumerate((1, 2, 3, 4)):
        sample = dataset.load_sample(scene, 9, 14)
        panels = [sample.image, sample.mask]
        for _, trainer, elite in trainers:
            prediction, _ = trainer.decoder.decode(elite, [[sample.image]])
            panels.append(np.asarray(prediction[0][0]) > 0)
        for col, image in enumerate(panels):
            axes[row, col].imshow(image, cmap="gray")
            title = f"Scene {scene}: Input" if col == 0 else headers[col]
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"Scene {scene}\nb=9, n=14", fontsize=10)
    fig.suptitle("Cross-scene qualitative comparison at b=9, n=14", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(OUT)

if __name__ == "__main__":
    main()
