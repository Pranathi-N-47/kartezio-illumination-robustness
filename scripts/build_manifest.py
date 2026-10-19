"""CLI script to scan benchmark dataset and generate manifest files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repository root to pythonpath
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from projects.illumination_robustness.data.manifest import (
    read_manifest,
    scan_benchmark,
    write_manifest,
    write_validation_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate benchmark manifest.")
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=repo_root / "hardware_items",
        help="Root directory of the benchmark dataset (default: hardware_items)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "data" / "manifests" / "benchmark_manifest.csv",
        help="Path where the CSV manifest will be saved.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=repo_root / "data" / "manifests" / "manifest_summary.json",
        help="Path where the validation summary JSON will be saved.",
    )

    args = parser.parse_args()

    print(f"Scanning benchmark at: {args.benchmark_root} ...")
    result = scan_benchmark(benchmark_root=args.benchmark_root, repo_root=repo_root)

    print(f"Writing manifest to: {args.output_csv} ...")
    write_manifest(result.records, args.output_csv)

    print(f"Writing summary to: {args.summary_json} ...")
    write_validation_summary(result, args.summary_json)

    summary = result.summary()
    print("\n--- Manifest Summary ---")
    print(f"Files scanned: {summary['n_files_scanned']}")
    print(f"Valid records: {summary['n_ok_records']} / {summary['n_expected_tuples']}")
    print(f"Record statuses: {summary['record_status_counts']}")
    print(f"Issue counts: {summary['issue_counts']}")
    print(f"Passed: {summary['passed']}")

    # Verify read back
    loaded = read_manifest(args.output_csv)
    print(f"Verified round-trip: read {len(loaded)} records from CSV.")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
