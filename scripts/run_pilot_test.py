"""Run a tiny deterministic Kartezio pilot test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage.measure import label

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kartezio.core.endpoints import EndpointThreshold
from kartezio.evolution import KartezioTrainer
from projects.illumination_robustness.evolution.fitness import PilotTestQSegFitness
from projects.illumination_robustness.primitives import build_illumination_library


def fixtures(fixture_dir: Path):
    samples, labels, names = [], [], []
    for image_path in sorted(fixture_dir.glob("*_input.tif")):
        name = image_path.name.removesuffix("_input.tif")
        mask_path = fixture_dir / f"{name}_ground_truth.tif"
        image = tifffile.imread(image_path).astype(np.uint8)
        mask = tifffile.imread(mask_path).astype(np.uint8)
        samples.append([image])
        labels.append([(mask > 0).astype(np.uint8) * 255])
        names.append(name)
    if not samples:
        raise FileNotFoundError(f"No pilot fixtures found in {fixture_dir}")
    return samples, labels, names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixture-dir", type=Path, default=Path("results/pilot_fixtures"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilot_test"))
    args = parser.parse_args()
    np.random.seed(args.seed)
    x_all, y_all, fixture_names = fixtures(args.fixture_dir)
    split_index = max(1, int(round(len(x_all) * 2 / 3)))
    x_train, y_train = x_all[:split_index], y_all[:split_index]
    x_test, y_test = x_all[split_index:], y_all[split_index:]
    train_names, test_names = fixture_names[:split_index], fixture_names[split_index:]
    trainer = KartezioTrainer(
        n_inputs=1,
        n_nodes=args.nodes,
        libraries=build_illumination_library(),
        endpoint=EndpointThreshold(128),
        fitness=PilotTestQSegFitness(),
    )
    trainer.set_mutation_rates(node_rate=0.05, out_rate=0.1)
    elite, history = trainer.fit(args.generations, x_train, y_train)
    train_scores = trainer.evaluate(x_train, y_train)
    test_scores = trainer.evaluate(x_test, y_test)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    history_fitness = [float(state.get_best_fitness()[1]) for state in history]
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(history_fitness)), history_fitness, marker="o")
    plt.xlabel("Evolution step")
    plt.ylabel("Best training loss (1 - mean Q_seg)")
    plt.title("Pilot-test fitness history")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "fitness_history.png", dpi=150)
    plt.close()

    predicted, _ = trainer.model.predict(x_test)
    active_graph = trainer.decoder.parse_to_graphs(elite)[0][0]
    active_nodes = sorted({int(node) for node, _ in active_graph if node >= 1})
    pipeline_nodes = []
    for node in active_nodes:
        function_index = trainer.decoder.adapter.get_function(
            elite, "chromosome_0", "matrix", node - 1
        )
        arity = trainer.decoder.arity_of("matrix", function_index)
        edges = trainer.decoder.adapter.get_active_edges(
            elite, "chromosome_0", "matrix", node - 1, arity
        )
        parameters = trainer.decoder.adapter.get_parameters(
            elite, "chromosome_0", "matrix", node - 1
        )
        pipeline_nodes.append({
            "node": node,
            "primitive": trainer.decoder.name_of("matrix", function_index),
            "edges": [int(edge) for edge in edges],
            "parameters": [int(parameter) for parameter in parameters],
        })
    pipeline = {
        "available_nodes": args.nodes,
        "active_nodes": len(pipeline_nodes),
        "nodes": pipeline_nodes,
        "outputs": [int(value) for value in trainer.decoder.adapter.get_outputs(elite, "chromosome_0")],
        "genotype": elite.__to_dict__(),
    }
    (args.output_dir / "pipeline.json").write_text(json.dumps(pipeline, indent=2), encoding="utf-8")
    (args.output_dir / "pipeline.txt").write_text(
        "Pilot-test elite pipeline\n"
        f"Active nodes: {len(pipeline_nodes)} / {args.nodes}\n" +
        "\n".join(f"node {item['node']}: {item['primitive']} edges={item['edges']} params={item['parameters']}"
                  for item in pipeline_nodes) + "\n",
        encoding="utf-8",
    )
    complexity = [int(label(y_item[0] > 0, connectivity=2).max()) for y_item in y_test]
    most_complex_index = int(np.argmax(complexity))
    n_test = len(x_test)
    fig, axes = plt.subplots(n_test, 3, figsize=(12, 3.5 * n_test))
    if n_test == 1:
        axes = np.expand_dims(axes, 0)
    for i in range(n_test):
        axes[i, 0].imshow(x_test[i][0], cmap="gray")
        axes[i, 0].set_title(f"Test input: {test_names[i]}")
        axes[i, 1].imshow(y_test[i][0] > 0, cmap="gray")
        axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(np.asarray(predicted[i][0]) > 0, cmap="gray")
        title = f"Best elite ({len(pipeline_nodes)} active nodes)"
        if i == most_complex_index:
            title += " — most complex test image"
        axes[i, 2].set_title(title)
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "elite_predictions.png", dpi=150)
    plt.close(fig)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"generations": args.generations, "nodes": args.nodes, "seed": args.seed,
                    "history_length": len(history),
                    "train_fitness": np.asarray(train_scores).tolist(),
                    "test_fitness": np.asarray(test_scores).tolist(),
                    "train_images": train_names, "test_images": test_names,
                    "active_nodes": len(pipeline_nodes),
                    "most_complex_fixture": test_names[most_complex_index],
                    "most_complex_object_count": complexity[most_complex_index],
                    "figures": ["figures/fitness_history.png", "figures/elite_predictions.png"],
                    "pipeline": ["pipeline.json", "pipeline.txt"]}, indent=2),
        encoding="utf-8",
    )
    print(f"Pilot test evolution complete: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
