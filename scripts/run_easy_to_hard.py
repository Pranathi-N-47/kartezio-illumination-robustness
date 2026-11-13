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
    p = argparse.ArgumentParser(); p.add_argument("--config", type=Path, required=True); p.add_argument("--common-config", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--run"); p.add_argument("--seed", type=int); p.add_argument("--generations", type=int); p.add_argument("--smoke", action="store_true")
    a = p.parse_args(); cfg = json.loads(a.config.read_text()); common = json.loads(a.common_config.read_text()); groups = {k: conditions_from_group(v) for k, v in cfg["condition_groups"].items()}; runs = cfg["runs"]
    if a.run: runs = [r for r in runs if r["name"] == a.run]
    if a.run and not runs: p.error(f"Unknown run: {a.run}")
    seeds = [a.seed] if a.seed is not None else common["seeds"]; generations = a.generations or (5 if a.smoke else common["generations"])
    scenes = list(info.SCENES)
    for spec in runs:
        train_conditions = [c for group in spec["train_groups"] for c in groups[group]]; test_conditions = [c for group in spec["test_groups"] for c in groups[group]]
        train = image_ids(scenes, train_conditions); test = image_ids(scenes, test_conditions)
        for seed in seeds:
            run_id = f"{spec['name']}_seed{seed}"
            run_experiment(run_id=run_id, experiment=cfg["experiment"], config=common, benchmark_root=REPO_ROOT / common["benchmark_root"], train_ids=train, test_ids=test, output=a.output_root / run_id, seed=seed, generations=generations, smoke=a.smoke)
            print(f"Completed {run_id}", flush=True)
    return 0
if __name__ == "__main__": raise SystemExit(main())
