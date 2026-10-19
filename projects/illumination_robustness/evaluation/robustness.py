"""Illumination robustness calculations across prescribed condition series.

Implements the 13-image robustness series X(r, b, b+1) [P, Eq. 3] and computes
robustness index R_seg as well as supplementary R_dice and R_iou.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from projects.illumination_robustness.data import benchmark_info as info


@dataclass(frozen=True)
class RobustnessCurvePoint:
    """A single condition point on the robustness curve.

    Attributes:
        shading: Shading index b.
        noise: Noise index n.
        artifact_level: A(r, b, n) in [0, 1].
        q_seg: Paper segmentation quality Q_seg.
        dice: Dice score.
        iou: IoU score.
    """

    shading: int
    noise: int
    artifact_level: float
    q_seg: float
    dice: float
    iou: float


@dataclass(frozen=True)
class RobustnessSeriesResult:
    """Robustness evaluation result for one scene across the 13-image series.

    Attributes:
        scene: Scene index r.
        points: Ordered list of 13 curve points (b=1..13, n=b+1).
        r_seg: Mean Q_seg across the 13 images.
        r_dice: Mean Dice score across the 13 images.
        r_iou: Mean IoU score across the 13 images.
    """

    scene: int
    points: List[RobustnessCurvePoint]
    r_seg: float
    r_dice: float
    r_iou: float


def compute_robustness_indices(
    scene: int,
    points: Sequence[RobustnessCurvePoint],
) -> RobustnessSeriesResult:
    """Compute R_seg, R_dice, and R_iou from the 13-image robustness series.

    Follows [P, Eq. 3]:
        R_seg = (1 / 13) * sum_{b=1..13} Q_seg(r, b, b+1)

    Args:
        scene: Scene index r.
        points: Sequence of exactly 13 RobustnessCurvePoints.

    Returns:
        :class:`RobustnessSeriesResult`.

    Raises:
        ValueError: If points does not have exactly 13 items.
    """
    if len(points) != info.N_SHADING:
        raise ValueError(
            f"Expected {info.N_SHADING} points for the robustness series, got {len(points)}"
        )

    pts = list(points)
    r_seg = float(np.mean([p.q_seg for p in pts]))
    r_dice = float(np.mean([p.dice for p in pts]))
    r_iou = float(np.mean([p.iou for p in pts]))

    return RobustnessSeriesResult(
        scene=scene,
        points=pts,
        r_seg=r_seg,
        r_dice=r_dice,
        r_iou=r_iou,
    )
