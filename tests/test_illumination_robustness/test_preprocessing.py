"""Unit tests for image normalisation and mask postprocessing."""

import numpy as np
import pytest

from projects.illumination_robustness.preprocessing.normalization import (
    NormalizationConfig,
    normalize_image,
)
from projects.illumination_robustness.preprocessing.postprocessing import (
    dilate_mask_rect,
    fill_mask_holes,
    filter_small_objects,
    mask_clear_border,
    open_mask_disk,
)


def test_normalize_image_regular():
    # 10x10 gradient from 0 to 99
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    # Default: 2%-98% percentile with polarity inversion (dark -> bright)
    norm = normalize_image(img)
    assert norm.shape == (10, 10)
    assert norm.dtype == np.float64
    assert 0.0 <= norm.min() <= norm.max() <= 1.0
    # Inverted: smallest raw value should become 1.0
    assert norm[0, 0] == 1.0
    # Largest raw value should become 0.0
    assert norm[9, 9] == 0.0


def test_normalize_image_no_inversion():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    cfg = NormalizationConfig(invert_polarity=False)
    norm = normalize_image(img, cfg)
    assert norm[0, 0] == 0.0
    assert norm[9, 9] == 1.0


def test_normalize_image_constant():
    # Constant image -> should return all zeros
    img = np.full((20, 20), 128, dtype=np.uint8)
    norm = normalize_image(img)
    assert np.all(norm == 0.0)


def test_normalize_image_nan_inf():
    with pytest.raises(ValueError, match="NaN or Inf"):
        normalize_image(np.array([[1.0, np.nan], [0.0, 1.0]]))

    with pytest.raises(ValueError, match="NaN or Inf"):
        normalize_image(np.array([[1.0, np.inf], [0.0, 1.0]]))


def test_normalize_image_non_2d():
    with pytest.raises(ValueError, match="must be 2-D"):
        normalize_image(np.zeros((10, 10, 3)))


def test_fill_mask_holes():
    # Square with a hole in the center
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    mask[3, 3] = False  # hole

    filled = fill_mask_holes(mask)
    assert filled[3, 3] is np.True_
    assert filled[0, 0] is np.False_


def test_mask_clear_border():
    mask = np.zeros((8, 8), dtype=bool)
    # Blob 1 touching border
    mask[0, 0:3] = True
    # Blob 2 in interior
    mask[3:5, 3:5] = True

    cleared = mask_clear_border(mask)
    assert np.all(~cleared[0, 0:3])
    assert np.all(cleared[3:5, 3:5])


def test_filter_small_objects():
    mask = np.zeros((10, 10), dtype=bool)
    # Small 2-pixel component
    mask[0, 0:2] = True
    # Larger 6-pixel component
    mask[4:6, 4:7] = True

    filtered = filter_small_objects(mask, min_size=5, connectivity=8)
    assert np.all(~filtered[0, 0:2])
    assert np.all(filtered[4:6, 4:7])


def test_morphology_open_dilate():
    mask = np.zeros((15, 15), dtype=bool)
    mask[5:10, 5:10] = True
    # Isolated noise pixel outside the main square
    mask[12, 12] = True

    opened = open_mask_disk(mask, radius=1)
    assert opened[12, 12] is np.False_
    assert opened[7, 7] is np.True_

    dilated = dilate_mask_rect(mask, size=3)
    assert dilated[4, 4] is np.True_
