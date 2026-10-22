"""Run multiple pilot generation budgets and preserve the most complex elite."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", nargs="+", type=int, default=[50, 100, 250, 500, 1000])
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixture-dir", type=Path, default=Path("results/pilot_fixtures"))
    parser.add_argument("--output-root", type=Path, default=Path("results/pilot_sweep"))
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for generations in args.generations:
        run_dir = args.output_root / f"generations_{generations}"
        command = [
            sys.executable, str(REPO_ROOT / "scripts" / "run_pilot_test.py"),
            "--fixture-dir", str(args.fixture_dir), "--generations", str(generations),
            "--nodes", str(args.nodes), "--seed", str(args.seed),
            "--output-dir", str(run_dir),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        records.append({"generations": generations, "run_dir": str(run_dir),
                        "active_nodes": summary["active_nodes"],
                        "fitness": float(summary["train_fitness"][0]),
                        "test_fitness": float(summary["test_fitness"][0])})

    best = min(records, key=lambda record: (record["fitness"], -record["generations"]))
    source = Path(best["run_dir"])
    destination = args.output_root / "best_pipeline"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in ["pipeline.json", "pipeline.txt", "summary.json"]:
        shutil.copy2(source / name, destination / name)
    shutil.copytree(source / "figures", destination / "figures")
    (args.output_root / "sweep_summary.json").write_text(
        json.dumps({"runs": records, "best_pipeline": best,
                    "best_pipeline_output": str(destination)}, indent=2),
        encoding="utf-8",
    )
    print(f"Best-performing pipeline saved to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
