"""Tests for ground-truth loading, binarization, and area-resize logic."""

import numpy as np
import pytest

from projects.illumination_robustness.data.ground_truth import (
    GroundTruthConfig,
    check_alignment,
    resize_binary_mask,
    to_binary_foreground,
)
from projects.illumination_robustness.evaluation.object_metrics import (
    count_blobs,
    label_blobs,
)


def test_to_binary_foreground():
    arr = np.array([[0, 1, 2], [5, 0, 0]], dtype=np.uint8)
    bin_mask = to_binary_foreground(arr)
    expected = np.array([[False, True, True], [True, False, False]], dtype=bool)
    np.testing.assert_array_equal(bin_mask, expected)

    # Rejection of NaN / Inf
    float_arr = np.array([[0.0, np.nan], [1.0, 0.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        to_binary_foreground(float_arr)


def test_resize_binary_mask_area():
    # 4x4 mask resized to 2x2 with 0.5 threshold
    mask = np.zeros((4, 4), dtype=bool)
    # top-left 2x2 block: 3 foreground pixels (fraction 3/4 = 0.75 >= 0.5 -> True)
    mask[0, 0] = True
    mask[0, 1] = True
    mask[1, 0] = True
    # top-right 2x2 block: 1 foreground pixel (fraction 1/4 = 0.25 < 0.5 -> False)
    mask[0, 2] = True

    resized = resize_binary_mask(mask, (2, 2), foreground_threshold=0.5)
    assert resized.shape == (2, 2)
    assert resized[0, 0] is np.True_
    assert resized[0, 1] is np.False_
    assert resized[1, 0] is np.False_
    assert resized[1, 1] is np.False_


def test_label_blobs_order():
    # Test column-major (MATLAB) BLOB numbering
    # Blob 1 at (row 2, col 0); Blob 2 at (row 0, col 2)
    mask = np.zeros((4, 4), dtype=bool)
    mask[2, 0] = True
    mask[0, 2] = True

    labels, n_blobs = label_blobs(mask, connectivity=8)
    assert n_blobs == 2
    # In column-major order, col 0 is scanned before col 2
    assert labels[2, 0] == 1
    assert labels[0, 2] == 2
    assert count_blobs(mask, connectivity=8) == 2


def test_check_alignment():
    img = np.zeros((100, 200), dtype=np.uint8)

    class DummyGT:
        scene = 1
        foreground = np.zeros((100, 200), dtype=bool)

    check_alignment(img, DummyGT())  # Should succeed

    class MismatchedGT:
        scene = 1
        foreground = np.zeros((100, 150), dtype=bool)

    with pytest.raises(ValueError, match="does not match ground truth"):
        check_alignment(img, MismatchedGT())
