"""Python port of the released MATLAB segmentation-only baselines.

This module intentionally follows ``hardware_items/code/benchAutoOtsu.m`` and
``benchAutoEdge.m`` rather than the newer project-defined baseline. It excludes
classification and reports the segmentation component only. MATLAB remains the
reference for numerical validation because library-level image-operation
differences can remain at boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import disk, opening, remove_small_objects
from skimage.segmentation import clear_border

from projects.illumination_robustness.evaluation.metrics import paper_q_seg


@dataclass(frozen=True)
class OtsuConfig:
    """Parameters matching the released MATLAB macro."""

    thresholds: tuple[float, ...] = tuple(np.arange(0.01, 0.991, 0.04))
    min_area: int = 100


@dataclass(frozen=True)
class SobelConfig:
    """Parameters matching the released MATLAB macro."""

    threshold_factors: tuple[float, ...] = tuple(np.arange(0.0, 1.001, 0.05))
    min_area: int = 100


def matlab_minmax_invert(image: np.ndarray) -> np.ndarray:
    """Approximate MATLAB ``im_norm(..., 'minmax', pixrev=true)``."""
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("Expected a 2-D grayscale image")
    if not np.isfinite(array).all():
        raise ValueError("Image contains NaN or Inf")
    values = array.astype(np.float64)
    if values.min() == values.max():
        return np.zeros(values.shape, dtype=np.float64)
    low, high = values.min(), values.max()
    return 1.0 - (values - low) / (high - low)


def _matlab_uint8_complement(image: np.ndarray) -> np.ndarray:
    """Match ``imcomplement(uint8(image)); im2double(...)`` for uint8 data."""
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("Expected a 2-D grayscale image")
    if not np.isfinite(array).all():
        raise ValueError("Image contains NaN or Inf")
    # The benchmark images are uint8. MATLAB's uint8 conversion saturates
    # non-integer inputs; casting here is exact for the benchmark input type.
    values = np.clip(array, 0, 255).astype(np.uint8)
    return (255.0 - values.astype(np.float64)) / 255.0


def _matlab_sobel_magnitude(normalized: np.ndarray) -> np.ndarray:
    """Compute the MATLAB-style 3x3 Sobel gradient magnitude."""
    kernel_x = np.asarray(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)), dtype=np.float64) / 8.0
    kernel_y = kernel_x.T
    # MATLAB image operators replicate the nearest border values for this
    # local gradient calculation; scipy's ``nearest`` gives the same policy.
    gx = ndimage.convolve(normalized, kernel_x, mode="nearest")
    gy = ndimage.convolve(normalized, kernel_y, mode="nearest")
    return np.hypot(gx, gy)


def _fill(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_fill_holes(mask)


def _postprocess_otsu(mask: np.ndarray, min_area: int) -> np.ndarray:
    result = _fill(mask)
    result = cv2.dilate(result.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    result = _fill(result)
    result = opening(result, disk(9))
    result = clear_border(_fill(result))
    return remove_small_objects(result, min_size=min_area, connectivity=2)


def _postprocess_sobel(mask: np.ndarray, min_area: int) -> np.ndarray:
    result = _fill(mask)
    result = cv2.dilate(result.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    result = _fill(result)
    result = opening(result, disk(5))
    result = clear_border(_fill(result))
    return remove_small_objects(result, min_size=min_area, connectivity=2)


def run_otsu_matlab_compatible(
    image: np.ndarray, ground_truth: Optional[np.ndarray] = None, config: Optional[OtsuConfig] = None
) -> np.ndarray:
    """Run the MATLAB threshold sweep and return the selected segmentation.

    When ``ground_truth`` is provided, candidates are selected by segmentation
    quality exactly as the released MATLAB baseline selects its best candidate.
    Without ground truth, the first threshold is used and no test-time oracle
    selection occurs.
    """
    cfg = config or OtsuConfig()
    normalized = matlab_minmax_invert(image)
    candidates = [_postprocess_otsu(normalized > threshold, cfg.min_area) for threshold in cfg.thresholds]
    if ground_truth is None:
        return candidates[0].astype(bool)
    scores = [paper_q_seg(candidate, np.asarray(ground_truth) > 0).q_seg for candidate in candidates]
    return candidates[int(np.argmax(scores))].astype(bool)


def run_sobel_matlab_compatible(
    image: np.ndarray, ground_truth: Optional[np.ndarray] = None, config: Optional[SobelConfig] = None
) -> np.ndarray:
    """Run the closest available Python equivalent of MATLAB Sobel ``edge``.

    The Sobel kernels and uint8 complement now match the MATLAB reference.
    ``threshold_otsu`` is used as the automatic base-threshold approximation;
    exact MATLAB ``edge`` threshold selection still requires cross-validation
    against MATLAB/Image Processing Toolbox.
    """
    cfg = config or SobelConfig()
    normalized = _matlab_uint8_complement(image)
    base_edges = _matlab_sobel_magnitude(normalized)
    base_threshold = 0.0 if not np.any(base_edges) else float(threshold_otsu(base_edges))
    candidates = [
        _postprocess_sobel(base_edges > base_threshold * factor, cfg.min_area)
        for factor in cfg.threshold_factors
    ]
    if ground_truth is None:
        return candidates[0].astype(bool)
    scores = [paper_q_seg(candidate, np.asarray(ground_truth) > 0).q_seg for candidate in candidates]
    return candidates[int(np.argmax(scores))].astype(bool)


def run_otsu_baseline(image: np.ndarray, config: Optional[OtsuConfig] = None, ground_truth=None) -> np.ndarray:
    """Compatibility wrapper for the MATLAB-compatible Otsu implementation."""
    return run_otsu_matlab_compatible(image, ground_truth=ground_truth, config=config)


def run_sobel_baseline(image: np.ndarray, config: Optional[SobelConfig] = None, ground_truth=None) -> np.ndarray:
    """Compatibility wrapper for the MATLAB-compatible Sobel implementation."""
    return run_sobel_matlab_compatible(image, ground_truth=ground_truth, config=config)
