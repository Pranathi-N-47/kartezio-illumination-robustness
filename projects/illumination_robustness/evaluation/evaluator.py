"""Evaluator module: scores sample predictions and produces structured results."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from projects.illumination_robustness.data.ground_truth import SceneGroundTruth
from projects.illumination_robustness.data.loader import BenchmarkSample
from projects.illumination_robustness.evaluation.metrics import (
    PaperSegmentationScore,
    custom_count_metrics,
    dice_score,
    iou_score,
    paper_q_seg,
)
from projects.illumination_robustness.evaluation.object_metrics import count_blobs


@dataclass(frozen=True)
class SampleEvaluationResult:
    """Complete evaluation metrics for one sample prediction.

    Attributes:
        image_id: Unique image identifier.
        scene: Scene index r.
        shading: Shading index b.
        noise: Noise index n.
        dice: Binary Dice score in [0, 1].
        iou: Binary IoU score in [0, 1].
        count_error: Custom count error.
        count_quality: Custom count quality in [0, 1].
        gt_object_count: Ground truth BLOB count (n_gt).
        predicted_object_count: Predicted BLOB count (n_pred).
        q1: Paper count error criterion.
        mu1: Paper fuzzy count quality.
        q2: Paper non-overlap criterion (1 - IoU).
        mu2: Paper fuzzy overlap quality.
        q_seg: Paper segmentation quality (mu1 * mu2).
        runtime_ms: Execution runtime in milliseconds.
        status: 'ok' or 'error'.
        error_type: Error class name if failed, else empty.
    """

    image_id: str
    scene: int
    shading: int
    noise: int
    dice: float
    iou: float
    count_error: float
    count_quality: float
    gt_object_count: int
    predicted_object_count: int
    q1: float
    mu1: float
    q2: float
    mu2: float
    q_seg: float
    runtime_ms: float = 0.0
    status: str = "ok"
    error_type: str = ""

    def to_csv_row(
        self,
        run_id: str = "",
        method: str = "",
        seed: int = 0,
        split_id: str = "",
        pipeline_id: str = "",
        metric_protocol: str = "paper_q_seg_v1",
    ) -> Dict[str, Any]:
        """Convert to dictionary matching the per-image CSV schema in Section 12."""
        return {
            "run_id": run_id,
            "method": method,
            "seed": seed,
            "split_id": split_id,
            "scene": self.scene,
            "shading": self.shading,
            "noise": self.noise,
            "image_id": self.image_id,
            "pipeline_id": pipeline_id,
            "metric_protocol": metric_protocol,
            "dice": self.dice,
            "iou": self.iou,
            "custom_count_error": self.count_error,
            "q_seg": self.q_seg,
            "mu1": self.mu1,
            "mu2": self.mu2,
            "q1": self.q1,
            "q2": self.q2,
            "gt_object_count": self.gt_object_count,
            "predicted_object_count": self.predicted_object_count,
            "status": self.status,
            "error_type": self.error_type,
            "runtime_ms": self.runtime_ms,
        }


def evaluate_prediction(
    prediction: np.ndarray,
    ground_truth: SceneGroundTruth,
    image_id: str,
    shading: int,
    noise: int,
    runtime_ms: float = 0.0,
    connectivity: int = 8,
) -> SampleEvaluationResult:
    """Evaluate a predicted binary mask against scene ground truth.

    Args:
        prediction: 2-D binary/nonzero array of predicted segmentation.
        ground_truth: Aligned ground-truth object for the scene.
        image_id: Unique sample identifier string.
        shading: Shading index b.
        noise: Noise index n.
        runtime_ms: Elapsed pipeline runtime in ms.
        connectivity: Connectivity for object counting and matching (4 or 8).

    Returns:
        :class:`SampleEvaluationResult` with all computed metrics.
    """
    pred_u8 = (prediction != 0).astype(np.uint8)
    gt_bool = ground_truth.foreground

    dice = dice_score(pred_u8, gt_bool)
    iou = iou_score(pred_u8, gt_bool)

    # BLOB counting
    n_pred = count_blobs(pred_u8, connectivity=connectivity)
    n_gt = ground_truth.n_blobs

    count_err, count_qual = custom_count_metrics(n_pred, n_gt)

    # Paper Q_seg
    paper_res = paper_q_seg(
        prediction=pred_u8,
        gt_foreground=gt_bool,
        gt_blob_labels=ground_truth.blob_labels,
        n_gt_blobs=ground_truth.n_blobs,
        connectivity=connectivity,
    )

    return SampleEvaluationResult(
        image_id=image_id,
        scene=ground_truth.scene,
        shading=shading,
        noise=noise,
        dice=dice,
        iou=iou,
        count_error=count_err,
        count_quality=count_qual,
        gt_object_count=n_gt,
        predicted_object_count=n_pred,
        q1=paper_res.q1,
        mu1=paper_res.mu1,
        q2=paper_res.q2,
        mu2=paper_res.mu2,
        q_seg=paper_res.q_seg,
        runtime_ms=runtime_ms,
        status="ok",
        error_type="",
    )
