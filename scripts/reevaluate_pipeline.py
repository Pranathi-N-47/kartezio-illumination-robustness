"""Re-evaluate saved elites on the complete 728-image benchmark grid."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))

from kartezio.core.components import Genotype
from kartezio.core.endpoints import EndpointThreshold
from kartezio.evolution import KartezioTrainer
from projects.illumination_robustness.data import benchmark_info as info
from projects.illumination_robustness.data.loader import BenchmarkDataset
from projects.illumination_robustness.evaluation.evaluator import evaluate_prediction
from projects.illumination_robustness.evolution.fitness import PilotTestQSegFitness
from projects.illumination_robustness.primitives import build_illumination_library


def evaluate_run(run_root: Path, benchmark_root: Path) -> dict:
    elite_data = json.loads((run_root / "pipeline" / "elite.json").read_text(encoding="utf-8"))
    config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    elite = Genotype.__from_dict__(elite_data["genotype"])
    trainer = KartezioTrainer(
        n_inputs=1,
        n_nodes=int(config["nodes"]),
        libraries=build_illumination_library(),
        endpoint=EndpointThreshold(128),
        fitness=PilotTestQSegFitness(),
    )
    dataset = BenchmarkDataset(benchmark_root)
    rows = []
    for scene in info.SCENES:
        for shading in info.SHADING_LEVELS:
            for noise in info.NOISE_LEVELS:
                sample = dataset.load_sample(scene, shading, noise)
                prediction, _ = trainer.decoder.decode(elite, [[sample.image]])
                result = evaluate_prediction(
                    np.asarray(prediction[0][0]), sample.ground_truth,
                    sample.image_id, shading, noise, connectivity=8,
                )
                rows.append(result.to_csv_row(
                    run_id=run_root.name, method="evolved_full_grid", seed=int(config["seed"]),
                    split_id="full_grid_728", pipeline_id=run_root.name,
                    metric_protocol=config["metric_protocol"],
                ))
    metrics_dir = run_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "full_grid.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"run_id": run_root.name, "images": len(rows),
               "dice": float(np.mean([r["dice"] for r in rows])),
               "iou": float(np.mean([r["iou"] for r in rows])),
               "q_seg": float(np.mean([r["q_seg"] for r in rows])),
               "count_error": float(np.mean([r["custom_count_error"] for r in rows]))}
    (metrics_dir / "full_grid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--input-root", type=Path, required=True); p.add_argument("--benchmark-root", type=Path, default=REPO_ROOT / "hardware_items"); p.add_argument("--run"); p.add_argument("--check-original-metrics", action="store_true"); a = p.parse_args()
    if a.run:
        root = a.input_root / a.run
        print(json.dumps(evaluate_run(root, a.benchmark_root), indent=2)); return 0
    failures = []
    for run in sorted(p for p in a.input_root.rglob("config.json") if p.parent.name != "aggregate"):
        root = run.parent
        required = [root / "split.json", root / "metadata.json", root / "pipeline" / "elite.json", root / "pipeline" / "elite.txt", root / "metrics" / "train_per_image.csv", root / "metrics" / "test_per_image.csv"]
        missing = [str(x) for x in required if not x.is_file()]
        if missing: failures.append({"run": str(root), "missing": missing}); continue
        try: json.loads((root / "pipeline" / "elite.json").read_text(encoding="utf-8")); json.loads((root / "split.json").read_text(encoding="utf-8"))
        except Exception as exc: failures.append({"run": str(root), "error": str(exc)})
    report = {"runs_checked": len(list(a.input_root.rglob("pipeline/elite.json"))), "failures": failures, "passed": not failures}
    (a.input_root / "reevaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return 1 if failures else 0
if __name__ == "__main__": raise SystemExit(main())
