"""Aggregate completed final experiment summaries."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--input-root", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); a = p.parse_args()
    rows = []
    for path in sorted(a.input_root.rglob("aggregate_summary.json")):
        if path.parent.parent.name == "aggregate": continue
        try: rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError: continue
    if not rows: raise SystemExit("No completed run summaries found")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with (a.output_dir / "final_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (a.output_dir / "final_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} run summaries to {a.output_dir}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
