"""Comprehensive metric definitions for binary segmentation and illumination robustness.

Implements:
1. Standard overlap metrics: Dice, IoU (with frozen empty-mask semantics).
2. Custom object-count error and count quality.
3. Paper-defined fuzzy segmentation score:
     q1 (object count match with S4 sequential matching rule),
     q2 (non-overlapping pixels / total pixels = 1 - IoU),
     mu1, mu2 (spline-based fuzzy membership [S3, S4]),
     Q_seg = mu1 * mu2.
4. Artifact level A(r, b, n) aggregating shading and noise [S3].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from projects.illumination_robustness.evaluation.object_metrics import label_blobs


def fuzzy_zmf(
    theta: float,
    s: float = 2.0,
    alpha: float = 0.0,
    beta: float = 1.0,
) -> float:
    """Compute the monotonically decreasing spline-based fuzzy membership function mu [S3, Eq. 2].

    For s=2, this is mathematically identical to MATLAB's ``evalmf(theta, [alpha beta], 'zmf')``.

    Args:
        theta: Evaluated scalar value.
        s: Curvature parameter (default 2.0).
        alpha: Lower bound where membership equals 1.0.
        beta: Upper bound where membership equals 0.0.

    Returns:
        Fuzzy membership value in [0.0, 1.0].
    """
    if beta <= alpha:
        raise ValueError(f"beta ({beta}) must be strictly greater than alpha ({alpha})")

    if theta <= alpha:
        return 1.0
    if theta >= beta:
        return 0.0

    midpoint = (alpha + beta) / 2.0
    factor = 2.0 ** (s - 1.0)
    range_span = beta - alpha

    if theta <= midpoint:
        term = (theta - alpha) / range_span
        return float(1.0 - factor * (term**s))
    else:
        term = (beta - theta) / range_span
        return float(factor * (term**s))


def dice_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute binary Dice similarity coefficient.

    Args:
        prediction: 2-D array; nonzero pixels are treated as foreground.
        ground_truth: 2-D array of identical shape; nonzero pixels are foreground.

    Returns:
        Dice coefficient in [0.0, 1.0].
        Empty-mask policy:
          - Both masks empty -> 1.0
          - Exactly one mask empty -> 0.0

    Raises:
        ValueError: If arrays are not 2-D or shapes differ.
    """
    if prediction.ndim != 2 or ground_truth.ndim != 2:
        raise ValueError("Inputs must be 2-D arrays")
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {ground_truth.shape}")

    pred_fg = prediction != 0
    gt_fg = ground_truth != 0

    pred_count = int(np.count_nonzero(pred_fg))
    gt_count = int(np.count_nonzero(gt_fg))

    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0

    intersection = int(np.count_nonzero(pred_fg & gt_fg))
    return float(2.0 * intersection / (pred_count + gt_count))


def iou_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute binary Intersection-over-Union (Jaccard Index).

    Args:
        prediction: 2-D array; nonzero pixels are treated as foreground.
        ground_truth: 2-D array of identical shape; nonzero pixels are foreground.

    Returns:
        IoU in [0.0, 1.0].
        Empty-mask policy:
          - Both masks empty -> 1.0
          - Exactly one mask empty -> 0.0

    Raises:
        ValueError: If arrays are not 2-D or shapes differ.
    """
    if prediction.ndim != 2 or ground_truth.ndim != 2:
        raise ValueError("Inputs must be 2-D arrays")
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {ground_truth.shape}")

    pred_fg = prediction != 0
    gt_fg = ground_truth != 0

    pred_count = int(np.count_nonzero(pred_fg))
    gt_count = int(np.count_nonzero(gt_fg))

    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0

    intersection = int(np.count_nonzero(pred_fg & gt_fg))
    union = pred_count + gt_count - intersection
    return float(intersection / union)


def custom_count_metrics(
    n_pred: int,
    n_gt: int,
) -> Tuple[float, float]:
    """Compute custom relative object count error and count quality.

    Args:
        n_pred: Number of predicted connected components.
        n_gt: Number of ground-truth connected components.

    Returns:
        (count_error, count_quality) where:
          count_error = |n_pred - n_gt| / max(n_gt, 1)
          count_quality = max(0.0, 1.0 - count_error)
    """
    denom = max(n_gt, 1)
    error = float(abs(n_pred - n_gt) / denom)
    quality = float(max(0.0, 1.0 - error))
    return error, quality


@dataclass(frozen=True)
class PaperSegmentationScore:
    """Detailed breakdown of paper-defined segmentation quality Q_seg [S4].

    Attributes:
        q1: Object count error criterion [S4, Eq. 2].
        mu1: Fuzzy object count quality [S4, Eq. 3, D3].
        q2: Non-overlapping pixels criterion [S4, Eq. 4] (= 1 - IoU).
        mu2: Fuzzy overlap quality [S4, Eq. 5, D3].
        q_seg: Segmentation quality Q_seg = mu1 * mu2 [S4, Eq. 9].
        n_gt_blobs: Number of ground-truth BLOBs (card(T)).
        n_pred_blobs: Number of predicted BLOBs (card(S)).
        n_matched_blobs: Number of correctly segmented BLOBs (card(S_c)).
    """

    q1: float
    mu1: float
    q2: float
    mu2: float
    q_seg: float
    n_gt_blobs: int
    n_pred_blobs: int
    n_matched_blobs: int


def paper_q_seg(
    prediction: np.ndarray,
    gt_foreground: np.ndarray,
    gt_blob_labels: Optional[np.ndarray] = None,
    n_gt_blobs: Optional[int] = None,
    connectivity: int = 8,
    overlap_threshold: float = 1.0 / 3.0,
) -> PaperSegmentationScore:
    """Calculate paper-defined segmentation score Q_seg = mu1 * mu2 [S4].

    Implements the sequential matching rule [S4, Eq. 1]:
      Each predicted BLOB S_u in S is evaluated against remaining ground-truth BLOBs.
      If max_z card(S_u ∩ T_z) > (1/3) * card(S_u), S_u is added to S_c and all
      overlapping T_v are removed from remaining ground truth.

    Args:
        prediction: 2-D binary/nonzero array of predicted segmentation.
        gt_foreground: 2-D boolean ground-truth foreground mask.
        gt_blob_labels: Pre-computed int32 GT BLOB label map (optional for performance).
        n_gt_blobs: Total number of GT BLOBs V (optional for performance).
        connectivity: 4 or 8 connectivity for BLOB labelling.
        overlap_threshold: Required fraction of S_u overlapping a GT BLOB (default: 1/3).

    Returns:
        :class:`PaperSegmentationScore` with all intermediate and final terms.
    """
    if prediction.shape != gt_foreground.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {gt_foreground.shape}")

    pred_fg = prediction != 0
    gt_fg = gt_foreground != 0

    # 1. Compute q2 and mu2
    # q2 = |G XOR P| / |G ∪ P| = 1 - IoU
    iou_val = iou_score(pred_fg, gt_fg)
    q2 = 1.0 - iou_val
    mu2 = fuzzy_zmf(q2, s=2.0, alpha=0.0, beta=1.0)

    # 2. GT BLOB labelling
    if gt_blob_labels is None or n_gt_blobs is None:
        gt_labels, n_gt = label_blobs(gt_fg, connectivity=connectivity)
    else:
        gt_labels = gt_blob_labels
        n_gt = n_gt_blobs

    # 3. Predicted BLOB labelling in column-major order
    pred_labels, n_pred = label_blobs(pred_fg, connectivity=connectivity)

    if n_gt == 0:
        if n_pred == 0:
            return PaperSegmentationScore(
                q1=0.0, mu1=1.0, q2=q2, mu2=mu2, q_seg=mu2,
                n_gt_blobs=0, n_pred_blobs=0, n_matched_blobs=0
            )
        else:
            return PaperSegmentationScore(
                q1=1.0, mu1=0.0, q2=q2, mu2=mu2, q_seg=0.0,
                n_gt_blobs=0, n_pred_blobs=n_pred, n_matched_blobs=0
            )

    if n_pred == 0:
        # No predicted objects
        q1 = 1.0
        mu1 = fuzzy_zmf(q1, s=2.0, alpha=0.0, beta=1.0)
        return PaperSegmentationScore(
            q1=q1, mu1=mu1, q2=q2, mu2=mu2, q_seg=float(mu1 * mu2),
            n_gt_blobs=n_gt, n_pred_blobs=0, n_matched_blobs=0
        )

    # 4. Sequential BLOB matching [S4, Eq. 1]
    # Build the predicted/GT overlap table once. The previous implementation
    # repeatedly created a full-image boolean mask and called np.unique for
    # every predicted BLOB. This contingency table contains the same counts
    # while scanning the labelled images only once.
    n_gt_plus_background = n_gt + 1
    pair_codes = pred_labels.astype(np.int64, copy=False).ravel() * n_gt_plus_background + gt_labels.ravel()
    overlap_table = np.bincount(
        pair_codes,
        minlength=(n_pred + 1) * n_gt_plus_background,
    ).reshape((n_pred + 1, n_gt_plus_background))

    # Remaining active GT blob labels. A boolean array is faster than a set and
    # preserves the same sequential removal rule.
    active_gt = np.ones(n_gt_plus_background, dtype=bool)
    active_gt[0] = False
    matched_count = 0

    # For each predicted BLOB u = 1..n_pred
    for u in range(1, n_pred + 1):
        overlap_row = overlap_table[u]
        c_u = int(np.sum(overlap_row))
        if c_u == 0:
            continue

        # Only consider currently active, non-background GT blobs.
        active_overlaps = overlap_row.copy()
        active_overlaps[~active_gt] = 0
        labels_arr = np.flatnonzero(active_overlaps)
        if labels_arr.size == 0:
            continue

        max_overlap = int(np.max(active_overlaps[labels_arr]))

        # Check threshold condition: max(card(S_u ∩ T_z)) > (1/3) * C_u
        if max_overlap > overlap_threshold * c_u:
            matched_count += 1
            # Remove all overlapping GT BLOBs from active set [S4, Eq. 1 & text]
            active_gt[labels_arr] = False

    q1 = float(abs(n_gt - matched_count) / n_gt)
    mu1 = fuzzy_zmf(q1, s=2.0, alpha=0.0, beta=1.0)
    q_seg = float(mu1 * mu2)

    return PaperSegmentationScore(
        q1=q1,
        mu1=mu1,
        q2=q2,
        mu2=mu2,
        q_seg=q_seg,
        n_gt_blobs=n_gt,
        n_pred_blobs=n_pred,
        n_matched_blobs=matched_count,
    )


def compute_artifact_level(raw_image: np.ndarray, noise_index: int) -> float:
    """Compute the combined artifact level A(r, b, n) in [0.0, 1.0] [S3, Eqs. 1-4].

    Args:
        raw_image: Raw grayscale benchmark image (pixel values in [0, 255]).
        noise_index: Noise level index n in 1..14.

    Returns:
        Artifact level A(r, b, n) where 0 = no distortion, 1 = maximum distortion.
    """
    x_m = float(np.mean(raw_image))
    # mu_shading: mean pixel value in [0, 255] [S3, Eq. 3]
    mu_shading = fuzzy_zmf(x_m, s=2.0, alpha=0.0, beta=255.0)

    # mu_noise: noise std dev sigma_b = sqrt(125 * (n - 1)) [S3, Eq. 4]
    sigma_b = math.sqrt(125.0 * (noise_index - 1))
    mu_noise = 1.0 - fuzzy_zmf(sigma_b, s=2.0, alpha=10.0, beta=45.0)

    # Aggregated artifact level [S3, Eq. 1]
    artifact = 1.0 - (1.0 - mu_shading) * (1.0 - mu_noise)
    return float(artifact)
