"""Run one- and two-scene full-condition holdout experiments."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))
from projects.illumination_robustness.data import benchmark_info as info
from projects.illumination_robustness.evolution.experiment_runner import image_ids, run_experiment

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--config", type=Path, required=True); p.add_argument("--common-config", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True); p.add_argument("--rotation"); p.add_argument("--seed", type=int); p.add_argument("--generations", type=int); p.add_argument("--population-size", type=int); p.add_argument("--fitness-objective", choices=["qseg", "hybrid_dice_count"], default=None); p.add_argument("--mutation-node-rate", type=float); p.add_argument("--mutation-output-rate", type=float); p.add_argument("--smoke", action="store_true")
    a = p.parse_args(); cfg = json.loads(a.config.read_text()); common = json.loads(a.common_config.read_text()); config = {**common, **cfg}
    conditions = [(b, n) for b in info.SHADING_LEVELS for n in info.NOISE_LEVELS]
    rotations = cfg["rotations"]
    if a.rotation: rotations = [r for r in rotations if r["name"] == a.rotation]
    if a.rotation and not rotations: p.error(f"Unknown rotation: {a.rotation}")
    if a.fitness_objective: common["fitness_objective"] = a.fitness_objective
    if a.mutation_node_rate is not None: common["mutation_node_rate"] = a.mutation_node_rate
    if a.mutation_output_rate is not None: common["mutation_output_rate"] = a.mutation_output_rate
    if a.population_size is not None: common["population_size"] = a.population_size
    seeds = [a.seed] if a.seed is not None else common["seeds"]
    generations = a.generations or (5 if a.smoke else common["generations"])
    for rotation in rotations:
        train = image_ids(rotation["train_scenes"], conditions); test = image_ids(rotation["test_scenes"], conditions)
        for seed in seeds:
            run_id = f"{cfg['experiment']}_{rotation['name']}_seed{seed}"
            run_experiment(run_id=run_id, experiment=cfg["experiment"], config=common, benchmark_root=REPO_ROOT / common["benchmark_root"], train_ids=train, test_ids=test, output=a.output_root / run_id, seed=seed, generations=generations, smoke=a.smoke)
            print(f"Completed {run_id}", flush=True)
    return 0
if __name__ == "__main__": raise SystemExit(main())
