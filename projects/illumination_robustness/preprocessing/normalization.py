"""Explicit normalisation policies for the benchmark images.

Follows the benchmark paper specification [P, p.5] and decision D9 in
``docs/implementation_decisions.md``.

Quantile normalisation clips pixel intensities to the 2nd and 98th Hazen percentiles,
rescales linearly to [0, 1], and inverts polarity so dark objects on a bright
background become bright foreground against a dark background.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

NormMode = Literal["percentile", "minmax", "none"]


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration for image intensity normalisation.

    Attributes:
        mode: Normalisation strategy ('percentile', 'minmax', or 'none').
        quantiles: Lower and upper percentiles (default: (2.0, 98.0)).
        invert_polarity: If True, invert so dark objects become bright (default: True).
        clip: If True, clip intensities to the quantile range before scaling.
        output_dtype: Floating point type of output ('float32' or 'float64').
    """

    mode: NormMode = "percentile"
    quantiles: Tuple[float, float] = (2.0, 98.0)
    invert_polarity: bool = True
    clip: bool = True
    output_dtype: Literal["float32", "float64"] = "float64"

    def __post_init__(self) -> None:
        q_low, q_high = self.quantiles
        if not (0.0 <= q_low < q_high <= 100.0):
            raise ValueError(f"Invalid quantiles {self.quantiles}; expected 0 <= q_low < q_high <= 100")


def normalize_image(
    image: np.ndarray,
    config: NormalizationConfig | None = None,
) -> np.ndarray:
    """Normalise a 2-D grayscale image according to the explicit policy.

    Args:
        image: 2-D numeric array.
        config: Normalisation configuration; defaults to 2%-98% Hazen percentile
            clipping with polarity inversion.

    Returns:
        2-D array of type ``config.output_dtype`` with values in [0.0, 1.0].
        If the image is constant (or ``p_high == p_low``), an array of zeros is returned.

    Raises:
        ValueError: If ``image`` is not 2-D or contains NaN / Inf values.
    """
    config = config or NormalizationConfig()
    arr = np.asarray(image)

    if arr.ndim != 2:
        raise ValueError(f"Image must be 2-D, got shape {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
        raise ValueError("Image array contains NaN or Inf values")

    target_dt = np.float32 if config.output_dtype == "float32" else np.float64

    if config.mode == "none":
        return arr.astype(target_dt)

    if config.mode == "minmax":
        val_min = float(np.min(arr))
        val_max = float(np.max(arr))
        if val_max <= val_min:
            return np.zeros(arr.shape, dtype=target_dt)
        scaled = (arr.astype(target_dt) - val_min) / (val_max - val_min)
        if config.invert_polarity:
            return 1.0 - scaled
        return scaled

    if config.mode == "percentile":
        q_low, q_high = config.quantiles
        # Hazen percentile definition matches MATLAB prctile
        p_low, p_high = np.percentile(arr, [q_low, q_high], method="hazen")
        p_low = float(p_low)
        p_high = float(p_high)

        if p_high <= p_low:
            # Constant or near-constant image
            return np.zeros(arr.shape, dtype=target_dt)

        work = arr.astype(target_dt)
        if config.clip:
            work = np.clip(work, p_low, p_high)

        scaled = (work - p_low) / (p_high - p_low)
        if config.invert_polarity:
            return 1.0 - scaled
        return scaled

    raise ValueError(f"Unsupported normalisation mode: {config.mode!r}")
