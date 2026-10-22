"""Otsu and Sobel classical segmentation baseline pipelines.

Directly reproduces the processing steps from Khan et al. (2015) and the released
MATLAB implementation (benchAutoOtsu.m, benchAutoEdge.m) with explicit parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from skimage.filters import sobel, threshold_otsu

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


@dataclass(frozen=True)
class OtsuConfig:
    """Explicit configuration parameters for the Otsu baseline pipeline.

    Attributes:
        norm_config: Normalisation configuration (default: 2-98% Hazen, polarity inverted).
        initial_min_area: Minimum area for initial noise filter (default: 50).
        dilation_size: Size of rectangular dilation structuring element (default: 7).
        opening_radius: Radius of disk opening structuring element (default: 5).
        final_min_area: Minimum area for final component filtering (default: 100).
    """

    norm_config: NormalizationConfig = field(default_factory=NormalizationConfig)
    initial_min_area: int = 50
    dilation_size: int = 7
    opening_radius: int = 5
    final_min_area: int = 100


@dataclass(frozen=True)
class SobelConfig:
    """Explicit configuration parameters for the Sobel baseline pipeline.

    Attributes:
        norm_config: Normalisation configuration.
        threshold_multiplier: Multiplier for RMS gradient edge thresholding (default: 2.0,
            matching MATLAB's edge(I, 'sobel') default).
        dilation_size: Size of rectangular dilation structuring element (default: 3).
        opening_radius: Radius of disk opening structuring element (default: 5).
        final_min_area: Minimum area for final component filtering (default: 100).
    """

    norm_config: NormalizationConfig = field(default_factory=NormalizationConfig)
    threshold_multiplier: float = 2.0
    dilation_size: int = 3
    opening_radius: int = 5
    final_min_area: int = 100


def run_otsu_baseline(
    image: np.ndarray,
    config: Optional[OtsuConfig] = None,
) -> np.ndarray:
    """Execute the Otsu segmentation baseline pipeline on a raw image.

    Pipeline stages:
      1. Normalise intensity (2-98% percentiles, inverted polarity).
      2. Global Otsu thresholding with inversion fallback if empty.
      3. Clear borders.
      4. Filter small components (< initial_min_area).
      5. Fill holes -> Dilate (size 7) -> Fill holes.
      6. Morphological opening with disk (radius 5).
      7. Clear border -> Fill holes -> Clear border.
      8. Final small-object filter (< final_min_area).

    Args:
        image: 2-D raw grayscale image array.
        config: Pipeline configuration; defaults to :class:`OtsuConfig`.

    Returns:
        2-D boolean mask of shape identical to ``image``.
    """
    cfg = config or OtsuConfig()
    norm = normalize_image(image, cfg.norm_config)

    # Constant / zero image handling
    if np.all(norm == 0.0):
        return np.zeros(norm.shape, dtype=bool)

    # 1. Otsu thresholding
    try:
        t = float(threshold_otsu(norm))
        mask = norm > t
    except ValueError:
        # Occurs if image has a single unique value
        return np.zeros(norm.shape, dtype=bool)

    # Inversion fallback (matching MATLAB benchAutoOtsu.m line 75-79)
    if not np.any(mask):
        inv = 1.0 - norm
        try:
            t_inv = float(threshold_otsu(inv))
            mask = inv > t_inv
        except ValueError:
            return np.zeros(norm.shape, dtype=bool)

    # 2. Morphological postprocessing (matching benchAutoOtsu.m lines 81-103)
    mask = mask_clear_border(mask)
    if cfg.initial_min_area > 0:
        mask = filter_small_objects(mask, min_size=cfg.initial_min_area, connectivity=8)

    mask = fill_mask_holes(mask)
    if cfg.dilation_size > 0:
        mask = dilate_mask_rect(mask, size=cfg.dilation_size)
    mask = fill_mask_holes(mask)

    if cfg.opening_radius > 0:
        mask = open_mask_disk(mask, radius=cfg.opening_radius)

    mask = mask_clear_border(mask)
    mask = mask_clear_border(fill_mask_holes(mask))

    if cfg.final_min_area > 0:
        mask = filter_small_objects(mask, min_size=cfg.final_min_area, connectivity=8)

    return mask.astype(bool)


def run_sobel_baseline(
    image: np.ndarray,
    config: Optional[SobelConfig] = None,
) -> np.ndarray:
    """Execute the Sobel edge detection baseline pipeline on a raw image.

    Pipeline stages:
      1. Normalise intensity (2-98% percentiles, inverted polarity).
      2. Compute Sobel gradient magnitude.
      3. Threshold gradient using RMS threshold: thresh = multiplier * sqrt(mean(G^2)).
      4. Fill holes inside closed edges.
      5. Dilate (size 3) -> Fill holes.
      6. Morphological opening with disk (radius 5).
      7. Fill holes -> Clear border.
      8. Final small-object filter (< final_min_area).

    Args:
        image: 2-D raw grayscale image array.
        config: Pipeline configuration; defaults to :class:`SobelConfig`.

    Returns:
        2-D boolean mask of shape identical to ``image``.
    """
    cfg = config or SobelConfig()
    norm = normalize_image(image, cfg.norm_config)

    # Constant / zero image handling
    if np.all(norm == 0.0):
        return np.zeros(norm.shape, dtype=bool)

    # 1. Sobel gradient magnitude
    grad = sobel(norm)

    # 2. RMS gradient threshold (matching MATLAB edge(I, 'sobel') default)
    mean_sq = float(np.mean(grad**2))
    thresh = cfg.threshold_multiplier * np.sqrt(mean_sq)
    edges = grad > thresh

    # 3. Morphological postprocessing (matching benchAutoEdge.m lines 227-238)
    mask = fill_mask_holes(edges)
    if cfg.dilation_size > 0:
        mask = dilate_mask_rect(mask, size=cfg.dilation_size)
    mask = fill_mask_holes(mask)

    if cfg.opening_radius > 0:
        mask = open_mask_disk(mask, radius=cfg.opening_radius)

    mask = mask_clear_border(fill_mask_holes(mask))

    if cfg.final_min_area > 0:
        mask = filter_small_objects(mask, min_size=cfg.final_min_area, connectivity=8)

    return mask.astype(bool)
