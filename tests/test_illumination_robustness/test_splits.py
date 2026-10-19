"""Tests for train/test splits and data leakage prevention."""

import tempfile
from pathlib import Path

import pytest

from projects.illumination_robustness.data import benchmark_info as info
from projects.illumination_robustness.data.splits import (
    EASY_CONDITIONS,
    STRATIFIED_CONDITIONS_9,
    create_loso_split,
)


def test_create_loso_splits_quality_only():
    for test_s in info.SCENES:
        split = create_loso_split(test_s, "quality_only")
        split.validate()

        assert split.test_scene == test_s
        assert test_s not in split.train_scenes
        assert len(split.train_scenes) == 3
        # 3 train scenes * 9 easy conditions = 27 train images
        assert len(split.train_image_ids) == 27
        # 1 test scene * 182 conditions = 182 test images
        assert len(split.test_image_ids) == 182
        # Robustness series has 13 images
        assert len(split.test_robustness_series_ids) == 13

        # Zero leakage
        assert set(split.train_image_ids).isdisjoint(set(split.test_image_ids))


def test_create_loso_splits_robustness_aware():
    for test_s in info.SCENES:
        split = create_loso_split(test_s, "robustness_aware")
        split.validate()

        assert split.test_scene == test_s
        assert len(split.train_conditions) == 9
        assert len(split.train_image_ids) == 27
        assert len(split.test_image_ids) == 182
        assert set(split.train_image_ids).isdisjoint(set(split.test_image_ids))


def test_split_serialization_round_trip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "split.json"
        split = create_loso_split(1, "robustness_aware")
        split.save(json_path)

        loaded = split.load(json_path)
        assert loaded.split_id == split.split_id
        assert loaded.test_scene == split.test_scene
        assert loaded.train_scenes == split.train_scenes
        assert loaded.train_conditions == split.train_conditions
        assert loaded.train_image_ids == split.train_image_ids
        assert loaded.test_image_ids == split.test_image_ids
