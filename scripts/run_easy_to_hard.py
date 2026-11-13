"""Run deterministic condition-transfer experiments."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))
from projects.illumination_robustness.data import benchmark_info as info
from projects.illumination_robustness.evolution.experiment_runner import conditions_from_group, image_ids, run_experiment

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--config", type=Path, required=True); p.add_argument("--common-config", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--run"); p.add_argument("--seed", type=int); p.add_argument("--generations", type=int); p.add_argument("--population-size", type=int); p.add_argument("--fitness-objective", choices=["qseg", "hybrid_dice_count"], default=None); p.add_argument("--mutation-node-rate", type=float); p.add_argument("--mutation-output-rate", type=float); p.add_argument("--train-scenes", help="Comma-separated scene numbers for training, e.g. 1"); p.add_argument("--test-scenes", help="Comma-separated scene numbers for testing, default: all"); p.add_argument("--smoke", action="store_true")
    a = p.parse_args(); cfg = json.loads(a.config.read_text()); common = json.loads(a.common_config.read_text()); groups = {k: conditions_from_group(v) for k, v in cfg["condition_groups"].items()}; runs = cfg["runs"]
    if a.fitness_objective: common["fitness_objective"] = a.fitness_objective
    if a.mutation_node_rate is not None: common["mutation_node_rate"] = a.mutation_node_rate
    if a.mutation_output_rate is not None: common["mutation_output_rate"] = a.mutation_output_rate
    if a.population_size is not None: common["population_size"] = a.population_size
    if a.run: runs = [r for r in runs if r["name"] == a.run]
    if a.run and not runs: p.error(f"Unknown run: {a.run}")
    seeds = [a.seed] if a.seed is not None else common["seeds"]; generations = a.generations or (5 if a.smoke else common["generations"])
    train_scenes = [int(v) for v in a.train_scenes.split(",")] if a.train_scenes else list(info.SCENES)
    test_scenes = [int(v) for v in a.test_scenes.split(",")] if a.test_scenes else list(info.SCENES)
    if not set(train_scenes).issubset(info.SCENES) or not set(test_scenes).issubset(info.SCENES):
        p.error("Scene numbers must be in 1,2,3,4")
    for spec in runs:
        train_conditions = [c for group in spec["train_groups"] for c in groups[group]]; test_conditions = [c for group in spec["test_groups"] for c in groups[group]]
        train = image_ids(train_scenes, train_conditions); test = image_ids(test_scenes, test_conditions)
        for seed in seeds:
            scene_suffix = "allscenes" if train_scenes == list(info.SCENES) else "train" + "".join(str(v) for v in train_scenes)
            run_id = f"{spec['name']}_{scene_suffix}_seed{seed}"
            run_experiment(run_id=run_id, experiment=cfg["experiment"], config=common, benchmark_root=REPO_ROOT / common["benchmark_root"], train_ids=train, test_ids=test, output=a.output_root / run_id, seed=seed, generations=generations, smoke=a.smoke, allow_train_test_overlap=bool(cfg.get("allow_train_test_overlap", False)))
            print(f"Completed {run_id}", flush=True)
    return 0
if __name__ == "__main__": raise SystemExit(main())
