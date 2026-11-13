"""Check that saved elite pipelines and saved per-image metrics exist."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--input-root", type=Path, required=True); p.add_argument("--check-original-metrics", action="store_true"); a = p.parse_args()
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
