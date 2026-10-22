"""Unit tests for Otsu and Sobel baseline pipelines."""

import numpy as np
import pytest

from projects.illumination_robustness.data.loader import BenchmarkDataset
from extras.baselines_python import (
    OtsuConfig,
    SobelConfig,
    run_otsu_baseline,
    run_sobel_baseline,
)
from projects.illumination_robustness.evaluation.metrics import dice_score, paper_q_seg


def test_baselines_constant_image():
    blank = np.zeros((100, 100), dtype=np.uint8)
    mask_ot = run_otsu_baseline(blank)
    mask_so = run_sobel_baseline(blank)

    assert mask_ot.shape == (100, 100)
    assert mask_ot.dtype == bool
    assert np.all(mask_ot == False)

    assert mask_so.shape == (100, 100)
    assert mask_so.dtype == bool
    assert np.all(mask_so == False)


def test_baselines_synthetic_disk():
    # Synthetic image: bright background (220), dark circular object (30) in center
    img = np.full((120, 120), 220, dtype=np.uint8)
    y, x = np.ogrid[:120, :120]
    center_y, center_x = 60, 60
    dist_sq = (y - center_y) ** 2 + (x - center_x) ** 2
    img[dist_sq <= 25**2] = 30  # radius 25 disk

    # Run Otsu
    mask_ot = run_otsu_baseline(img)
    assert mask_ot.shape == (120, 120)
    assert np.any(mask_ot)
    # The center of the disk should be detected as foreground
    assert mask_ot[60, 60] is np.True_
    # The outer boundary should be background
    assert mask_ot[5, 5] is np.False_

    # Run Sobel
    mask_so = run_sobel_baseline(img)
    assert mask_so.shape == (120, 120)
    assert np.any(mask_so)
    assert mask_so[60, 60] is np.True_
    assert mask_so[5, 5] is np.False_


def test_baselines_determinism():
    img = np.arange(10000, dtype=np.uint8).reshape(100, 100)
    m1 = run_otsu_baseline(img)
    m2 = run_otsu_baseline(img)
    np.testing.assert_array_equal(m1, m2)

    s1 = run_sobel_baseline(img)
    s2 = run_sobel_baseline(img)
    np.testing.assert_array_equal(s1, s2)


def test_baselines_on_real_sample():
    dataset = BenchmarkDataset("hardware_items")
    sample = dataset.load_sample(1, 1, 1)

    # Test Otsu
    mask_ot = run_otsu_baseline(sample.image)
    assert mask_ot.shape == sample.mask.shape
    dice_ot = dice_score(mask_ot, sample.mask)
    q_ot = paper_q_seg(mask_ot, sample.mask).q_seg
    assert dice_ot > 0.80
    assert q_ot > 0.70

    # Test Sobel
    mask_so = run_sobel_baseline(sample.image)
    assert mask_so.shape == sample.mask.shape
    dice_so = dice_score(mask_so, sample.mask)
    q_so = paper_q_seg(mask_so, sample.mask).q_seg
    assert dice_so > 0.85
    assert q_so > 0.85
