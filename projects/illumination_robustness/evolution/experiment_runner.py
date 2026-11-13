"""Shared execution and artifact handling for final segmentation experiments."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from kartezio.core.endpoints import EndpointThreshold
from kartezio.callback import Callback
from kartezio.evolution import KartezioTrainer
from projects.illumination_robustness.data import benchmark_info as info
from projects.illumination_robustness.data.loader import BenchmarkDataset
from projects.illumination_robustness.evaluation.evaluator import evaluate_prediction
from projects.illumination_robustness.evolution.fitness import PilotTestQSegFitness, HybridDiceCountFitness
from projects.illumination_robustness.primitives import build_illumination_library


class GenerationTimingCallback(Callback):
    """Print Kartezio progress plus wall-clock time for each full generation."""

    def __init__(self, fitness):
        super().__init__()
        self.fitness = fitness
        self._generation_started = None
        self._fitness_seconds_at_start = 0.0

    def on_generation_start(self, iteration, state):
        self._generation_started = time.perf_counter()
        self._fitness_seconds_at_start = self.fitness.total_evaluation_seconds

    def on_generation_end(self, iteration, state):
        elapsed = time.perf_counter() - self._generation_started
        fitness_seconds = self.fitness.total_evaluation_seconds - self._fitness_seconds_at_start
        graph_seconds = max(0.0, elapsed - fitness_seconds)
        _, fitness, internal_time = state.get_best_fitness()
        fps = float("inf") if internal_time == 0 else 1.0 / internal_time
        print(
            f"[G {iteration:06}] loss={fitness:.6f} "
            f"internal={internal_time:.6f}s ({fps:.0f} eval/s) "
            f"generation_wall={elapsed:.2f}s "
            f"fitness={fitness_seconds:.2f}s graph/overhead={graph_seconds:.2f}s",
            flush=True,
        )


def image_ids(scenes: Iterable[int], conditions: Iterable[tuple[int, int]]) -> list[str]:
    return [info.image_id(s, b, n) for s in scenes for b, n in conditions]


def parse_id(image_id: str) -> tuple[int, int, int]:
    parts = image_id.split("_")
    if len(parts) != 3 or not parts[0].startswith("r") or not parts[1].startswith("b") or not parts[2].startswith("n"):
        raise ValueError(f"Invalid image ID: {image_id!r}; expected rRR_bBB_nNN")
    return int(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:])


def conditions_from_group(spec: dict) -> list[tuple[int, int]]:
    if spec.get("all_conditions"):
        return [(b, n) for b in info.SHADING_LEVELS for n in info.NOISE_LEVELS]
    if "difficulty_max" in spec or "difficulty_min_exclusive" in spec:
        result = []
        for b in info.SHADING_LEVELS:
            for n in info.NOISE_LEVELS:
                difficulty = ((b - 1) / 12 + (n - 1) / 13) / 2
                if spec.get("difficulty_min_exclusive") is not None and difficulty <= spec["difficulty_min_exclusive"]:
                    continue
                if spec.get("difficulty_max") is not None and difficulty > spec["difficulty_max"]:
                    continue
                result.append((b, n))
        return result
    return [(b, n) for b in spec["shading_indices"] for n in spec["noise_indices"]]


def load_items(dataset: BenchmarkDataset, ids: list[str]):
    x, y, samples = [], [], []
    for item in ids:
        scene, shading, noise = parse_id(item)
        sample = dataset.load_sample(scene, shading, noise)
        x.append([sample.image])
        y.append([(sample.mask.astype(np.uint8) * 255)])
        samples.append(sample)
    return x, y, samples


def pipeline_record(trainer, elite, nodes: int) -> dict:
    graph = trainer.decoder.parse_to_graphs(elite)[0][0]
    active = sorted({int(node) for node, _ in graph if int(node) >= 1})
    records = []
    for node in active:
        fn = trainer.decoder.adapter.get_function(elite, "chromosome_0", "matrix", node - 1)
        arity = trainer.decoder.arity_of("matrix", fn)
        edges = trainer.decoder.adapter.get_active_edges(elite, "chromosome_0", "matrix", node - 1, arity)
        params = trainer.decoder.adapter.get_parameters(elite, "chromosome_0", "matrix", node - 1)
        records.append({"node": node, "primitive": trainer.decoder.name_of("matrix", fn),
                        "edges": [int(v) for v in edges], "parameters": [int(v) for v in params]})
    return {"available_nodes": nodes, "active_nodes": len(records), "nodes": records,
            "outputs": [int(v) for v in trainer.decoder.adapter.get_outputs(elite, "chromosome_0")],
            "genotype": elite.__to_dict__()}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run_experiment(*, run_id: str, experiment: str, config: dict, benchmark_root: Path,
                   train_ids: list[str], test_ids: list[str], output: Path, seed: int,
                   generations: int | None = None, smoke: bool = False,
                   allow_train_test_overlap: bool = False) -> Path:
    if set(train_ids) & set(test_ids) and not allow_train_test_overlap:
        raise ValueError(f"Train/test overlap in {run_id}")
    print(f"[{run_id}] starting seed={seed} generations={generations or config['generations']}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("pipeline", "metrics", "figures", "figures/failure_cases"):
        (output / name).mkdir(parents=True, exist_ok=True)
    resolved = dict(config); resolved["seed"] = seed
    resolved["generations"] = generations or config["generations"]
    resolved["smoke"] = smoke
    (output / "config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    split = {"run_id": run_id, "experiment": experiment, "seed": seed,
             "train_image_ids": train_ids, "test_image_ids": test_ids,
             "train_count": len(train_ids), "test_count": len(test_ids)}
    (output / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (output / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    print(f"[{run_id}] loading benchmark samples: train={len(train_ids)}, test={len(test_ids)}", flush=True)
    dataset = BenchmarkDataset(benchmark_root)
    x_train, y_train, _ = load_items(dataset, train_ids)
    x_test, y_test, test_samples = load_items(dataset, test_ids)
    print(f"[{run_id}] samples loaded; initializing Kartezio population={config['population_size']}, nodes={config['nodes']}", flush=True)
    np.random.seed(seed)
    objective = config.get("fitness_objective", "qseg")
    if objective == "hybrid_dice_count":
        fitness = HybridDiceCountFitness()
    else:
        fitness = PilotTestQSegFitness()
    print(f"[{run_id}] fitness objective={objective}", flush=True)
    trainer = KartezioTrainer(n_inputs=1, n_nodes=int(config["nodes"]),
        libraries=build_illumination_library(), endpoint=EndpointThreshold(128),
        fitness=fitness)
    trainer.set_mutation_rates(node_rate=float(config["mutation_node_rate"]),
                               out_rate=float(config["mutation_output_rate"]))
    trainer.set_n_children(int(config["population_size"]))
    print(f"[{run_id}] evolution started; progress is reported for every generation", flush=True)
    elite, history = trainer.fit(
        int(resolved["generations"]),
        x_train,
        y_train,
        callbacks=[GenerationTimingCallback(fitness)],
    )
    print(f"[{run_id}] evolution finished; exporting elite pipeline", flush=True)
    record = pipeline_record(trainer, elite, int(config["nodes"]))
    (output / "pipeline" / "elite.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    text = [f"Elite pipeline: {run_id}", f"Active nodes: {record['active_nodes']} / {record['available_nodes']}"]
    text.extend(f"node {n['node']}: {n['primitive']} edges={n['edges']} params={n['parameters']}" for n in record["nodes"])
    (output / "pipeline" / "elite.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"[{run_id}] evaluating train and test predictions", flush=True)
    evaluation_started = time.perf_counter()
    train_pred, _ = trainer.model.predict(x_train)
    test_pred, _ = trainer.model.predict(x_test)
    rows = []
    for split_name, preds, samples in (("train", train_pred, load_items(dataset, train_ids)[2]), ("test", test_pred, test_samples)):
        for pred, sample in zip(preds, samples):
            result = evaluate_prediction(pred[0], sample.ground_truth, sample.image_id, sample.shading, sample.noise, connectivity=8)
            row = result.to_csv_row(run_id=run_id, method="evolved", seed=seed, split_id=run_id,
                                    pipeline_id=run_id, metric_protocol=config["metric_protocol"])
            row["split"] = split_name; rows.append(row)
    write_csv(output / "metrics" / "all_per_image.csv", rows)
    write_csv(output / "metrics" / "train_per_image.csv", [r for r in rows if r["split"] == "train"])
    write_csv(output / "metrics" / "test_per_image.csv", [r for r in rows if r["split"] == "test"])
    test_q = [float(r["q_seg"]) for r in rows if r["split"] == "test"]
    summary = {"run_id": run_id, "experiment": experiment, "seed": seed,
               "train_q_seg": float(np.mean([r["q_seg"] for r in rows if r["split"] == "train"])),
               "test_q_seg": float(np.mean(test_q)), "test_dice": float(np.mean([r["dice"] for r in rows if r["split"] == "test"])),
               "test_iou": float(np.mean([r["iou"] for r in rows if r["split"] == "test"])),
               "active_nodes": record["active_nodes"], "history_length": len(history)}
    (output / "metrics" / "aggregate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fitness = [float(state.get_best_fitness()[1]) for state in history]
    plt.figure(figsize=(7, 4)); plt.plot(fitness); plt.xlabel("Generation"); plt.ylabel("Best training loss"); plt.tight_layout(); plt.savefig(output / "figures" / "fitness_history.png", dpi=150); plt.close()
    (output / "metadata.json").write_text(json.dumps({"status": "complete", "run_id": run_id, "seed": seed, "commit": _git_commit()}, indent=2), encoding="utf-8")
    print(f"[{run_id}] final evaluation wall={time.perf_counter() - evaluation_started:.2f}s", flush=True)
    print(f"[{run_id}] complete; results saved to {output}", flush=True)
    return output


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"
