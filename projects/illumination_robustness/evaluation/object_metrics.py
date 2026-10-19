"""Connected-component (BLOB) labelling and object-count rules.

[S4] defines BLOBs as connected sets of foreground pixels. Connectivity is 8 by
default, matching MATLAB ``bwlabel``/``regionprops`` used by the reference code [M].
Touching objects form one BLOB under this rule; that limitation is inherent to the
paper's definition and is documented in ``docs/implementation_decisions.md`` (D7).
"""

from __future__ import annotations

import numpy as np
from skimage.measure import label

_SKIMAGE_CONNECTIVITY = {4: 1, 8: 2}


def label_blobs(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Label connected foreground components in MATLAB (column-major) order.

    [S4] processes predicted BLOBs sequentially, and the reference implementation
    iterates in ``regionprops`` order, i.e. by first pixel in column-major scan
    order. Labelling the transposed mask with a row-major labeller reproduces that
    order exactly, so sequential matching is deterministic and source-faithful.

    Args:
        mask: 2-D array; nonzero pixels are foreground.
        connectivity: 4 or 8 (2-D pixel connectivity).

    Returns:
        ``(labels, n_blobs)`` where ``labels`` is an int32 array of the same shape
        with 0 = background and 1..n_blobs in column-major first-pixel order.

    Raises:
        ValueError: If ``mask`` is not 2-D or ``connectivity`` is not 4 or 8.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if connectivity not in _SKIMAGE_CONNECTIVITY:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    foreground = np.asarray(mask) != 0
    labels_t, n_blobs = label(
        foreground.T, connectivity=_SKIMAGE_CONNECTIVITY[connectivity], return_num=True
    )
    return np.ascontiguousarray(labels_t.T, dtype=np.int32), int(n_blobs)


def count_blobs(mask: np.ndarray, connectivity: int = 8) -> int:
    """Return the number of connected foreground components (see :func:`label_blobs`)."""
    return label_blobs(mask, connectivity)[1]
