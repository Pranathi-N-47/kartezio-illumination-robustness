"""Create a compact qualitative comparison for selected conditions and elites."""
from pathlib import Path
import json
import sys
import argparse
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

OUT = ROOT / "results" / "figures" / "qualitative_comparison_scene3.png"
RUN_ROOT = ROOT / "results" / "final" / "easy_to_hard_scene_rotations"
RUNS = [(1, "Train scene 1"), (3, "Train scene 3"), (4, "Train scene 4")]
IMAGES = [(1, 14, "Bright: b=1, n=14"), (7, 14, "Medium: b=7, n=14"), (13, 14, "Severe: b=13, n=14")]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    scene_id = args.scene
    output_path = args.output or ROOT / "results" / "figures" / f"qualitative_comparison_scene{scene_id}.png"
    dataset = BenchmarkDataset(ROOT / "hardware_items")
    trainers = {}
    for scene, label in RUNS:
        run = RUN_ROOT / f"easy_to_medium_hard_train{scene}_seed7"
        info = json.loads((run / "pipeline" / "elite.json").read_text(encoding="utf-8"))
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        trainer = KartezioTrainer(n_inputs=1, n_nodes=int(config["nodes"]), libraries=build_illumination_library(), endpoint=EndpointThreshold(128), fitness=PilotTestQSegFitness())
        trainers[label] = (trainer, Genotype.__from_dict__(info["genotype"]))

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    headers = ["Input", "Ground truth", "Train scene 1", "Train scene 3", "Train scene 4"]
    for row, (shading, noise, condition) in enumerate(IMAGES):
        sample = dataset.load_sample(scene_id, shading, noise)
        panels = [sample.image, sample.mask]
        for label, (trainer, elite) in trainers.items():
            prediction, _ = trainer.decoder.decode(elite, [[sample.image]])
            panels.append(np.asarray(prediction[0][0]) > 0)
        for col, image in enumerate(panels):
            axes[row, col].imshow(image, cmap="gray")
            axes[row, col].set_title(f"Scene {scene_id}: {headers[col]}", fontsize=9)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(condition, fontsize=10)
    fig.suptitle(f"Qualitative comparison on Scene {scene_id} at increasing shading", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(output_path)

if __name__ == "__main__":
    main()
