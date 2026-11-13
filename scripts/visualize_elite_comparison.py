"""Visualize the four scene-rotation elites on one benchmark image."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))

from kartezio.core.components import Genotype
from kartezio.core.endpoints import EndpointThreshold
from kartezio.evolution import KartezioTrainer
from projects.illumination_robustness.data.loader import BenchmarkDataset
from projects.illumination_robustness.evolution.experiment_runner import parse_id
from projects.illumination_robustness.evolution.fitness import PilotTestQSegFitness
from projects.illumination_robustness.primitives import build_illumination_library


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image-id", required=True, help="Image ID such as r01_b09_n14")
    p.add_argument("--results-root", type=Path, default=Path("results/final/easy_to_hard_scene_rotations"))
    p.add_argument("--benchmark-root", type=Path, default=REPO_ROOT / "hardware_items")
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    scene, shading, noise = parse_id(args.image_id)
    dataset = BenchmarkDataset(args.benchmark_root)
    sample = dataset.load_sample(scene, shading, noise)
    run_dirs = sorted(args.results_root.glob("easy_to_medium_hard_train*_seed7"))
    if not run_dirs:
        raise SystemExit(f"No scene-rotation elites found in {args.results_root}")

    masks = []
    labels = []
    for run_dir in run_dirs:
        elite_info = json.loads((run_dir / "pipeline" / "elite.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        trainer = KartezioTrainer(n_inputs=1, n_nodes=int(config["nodes"]),
            libraries=build_illumination_library(), endpoint=EndpointThreshold(128),
            fitness=PilotTestQSegFitness())
        elite = Genotype.__from_dict__(elite_info["genotype"])
        prediction, _ = trainer.decoder.decode(elite, [[sample.image]])
        masks.append(np.asarray(prediction[0][0]) > 0)
        labels.append(run_dir.name)

    panels = [(sample.image, "Input"), (sample.mask, "Ground truth")]
    panels.extend(zip(masks, labels))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (image, title) in zip(axes.ravel(), panels):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Elite comparison: {args.image_id}")
    fig.tight_layout()
    output = args.output or args.results_root / "figures" / f"elite_comparison_{args.image_id}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
