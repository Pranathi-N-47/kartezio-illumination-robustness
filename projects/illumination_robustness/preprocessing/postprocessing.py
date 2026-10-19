"""Shared, documented binary mask postprocessing operations.

Implements clear border, hole filling, small object removal, and morphological
operations used by baseline and evaluation routines.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import clear_border


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a binary foreground mask.

    Args:
        mask: 2-D boolean mask.

    Returns:
        2-D boolean mask with interior holes filled.
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-D, got {mask.shape}")
    return binary_fill_holes(mask != 0)


def mask_clear_border(mask: np.ndarray, buffer_size: int = 1) -> np.ndarray:
    """Clear objects connected to the image border.

    Args:
        mask: 2-D boolean mask.
        buffer_size: Margin at the border.

    Returns:
        2-D boolean mask with border objects removed.
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-D, got {mask.shape}")
    return clear_border(mask != 0, buffer_size=buffer_size)


def filter_small_objects(mask: np.ndarray, min_size: int = 100, connectivity: int = 8) -> np.ndarray:
    """Remove connected foreground components smaller than min_size pixels.

    Args:
        mask: 2-D boolean mask.
        min_size: Minimum number of pixels required to keep an object.
        connectivity: 4 or 8 connectivity.

    Returns:
        2-D boolean mask with small components removed.
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-D, got {mask.shape}")
    conn = 1 if connectivity == 4 else 2
    return remove_small_objects(mask != 0, min_size=min_size, connectivity=conn)


def open_mask_disk(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Perform binary morphological opening with a disk structuring element.

    Matches MATLAB ``imopen(mask, strel('disk', radius))``.

    Args:
        mask: 2-D boolean mask.
        radius: Radius of the disk element.

    Returns:
        Opened 2-D boolean mask.
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-D, got {mask.shape}")
    from skimage.morphology import binary_opening
    return binary_opening(mask != 0, disk(radius))


def dilate_mask_rect(mask: np.ndarray, size: int = 7) -> np.ndarray:
    """Perform binary morphological dilation with a square kernel of size x size.

    Args:
        mask: 2-D boolean mask.
        size: Size of the square structuring element (e.g. 7 for ones(7) in MATLAB).

    Returns:
        Dilated 2-D boolean mask.
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-D, got {mask.shape}")
    kernel = np.ones((size, size), dtype=np.uint8)
    arr_u8 = (mask != 0).astype(np.uint8)
    dilated = cv2.dilate(arr_u8, kernel)
    return dilated != 0
