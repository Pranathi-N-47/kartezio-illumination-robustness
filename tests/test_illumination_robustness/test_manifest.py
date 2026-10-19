"""Tests for benchmark manifest generation and validation."""

import tempfile
from pathlib import Path

import pytest

from projects.illumination_robustness.data import benchmark_info as info
from projects.illumination_robustness.data.manifest import (
    ManifestIssue,
    ManifestRecord,
    ManifestResult,
    match_benchmark_filename,
    parse_benchmark_filename,
    read_manifest,
    scan_benchmark,
    write_manifest,
    write_validation_summary,
)


def test_parse_benchmark_filename_valid():
    scene, shading, noise = parse_benchmark_filename("benchdata_01_02_03.tif")
    assert (scene, shading, noise) == (1, 2, 3)

    scene, shading, noise = parse_benchmark_filename("benchdata_04_13_14.tiff")
    assert (scene, shading, noise) == (4, 13, 14)


def test_parse_benchmark_filename_invalid():
    with pytest.raises(ValueError, match="does not match"):
        parse_benchmark_filename("benchdata_1_2_3.tif")

    with pytest.raises(ValueError, match="does not match"):
        parse_benchmark_filename("other_01_02_03.tif")

    with pytest.raises(ValueError, match="scene 5 not in"):
        parse_benchmark_filename("benchdata_05_01_01.tif")

    with pytest.raises(ValueError, match="shading 15 not in"):
        parse_benchmark_filename("benchdata_01_15_01.tif")

    with pytest.raises(ValueError, match="noise 0 not in"):
        parse_benchmark_filename("benchdata_01_01_00.tif")


def test_manifest_csv_round_trip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "manifest.csv"
        records = [
            ManifestRecord(
                image_id="r01_b01_n01",
                scene=1,
                shading=1,
                noise=1,
                image_path="hardware_items/data/benchdata_01_01_01.tif",
                height=1000,
                width=1500,
                dtype="uint8",
                channels=1,
                sha256="abc123def456",
                gt_mask_path="hardware_items/truth/benchdata_1_gd.mat",
                gt_class_path="hardware_items/truth/benchdata_1_class.mat",
                status="ok",
                error="",
            ),
            ManifestRecord(
                image_id="r02_b05_n10",
                scene=2,
                shading=5,
                noise=10,
                image_path="hardware_items/data/benchdata_02_05_10.tif",
                height=1000,
                width=1500,
                dtype="uint8",
                channels=1,
                sha256="def789",
                gt_mask_path="",
                gt_class_path="",
                status="missing_annotation",
                error="Annotation missing",
            ),
        ]

        write_manifest(records, csv_path)
        loaded = read_manifest(csv_path)

        assert len(loaded) == 2
        assert loaded[0].image_id == "r01_b01_n01"
        assert loaded[0].height == 1000
        assert loaded[0].channels == 1
        assert loaded[0].status == "ok"
        assert loaded[1].image_id == "r02_b05_n10"
        assert loaded[1].status == "missing_annotation"


def test_manifest_validation_summary():
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "summary.json"
        res = ManifestResult(
            benchmark_root="hardware_items",
            records=[
                ManifestRecord("r01_b01_n01", 1, 1, 1, "path1", status="ok"),
            ],
            issues=[
                ManifestIssue("missing_tuple", "", "benchdata_01_01_02.tif missing"),
            ],
            n_files_scanned=1,
        )

        write_validation_summary(res, json_path)
        summary = res.summary()
        assert summary["n_files_scanned"] == 1
        assert summary["n_records"] == 1
        assert summary["passed"] is False  # due to missing_tuple issue
