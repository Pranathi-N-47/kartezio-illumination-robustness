 """Unit tests for metrics, fuzzy membership, Q_seg, and robustness calculations."""

import numpy as np
import pytest

from projects.illumination_robustness.data.ground_truth import (
    GroundTruthConfig,
    build_scene_ground_truth,
    RawSceneAnnotation,
)
from projects.illumination_robustness.evaluation.evaluator import evaluate_prediction
from projects.illumination_robustness.evaluation.metrics import (
    compute_artifact_level,
    custom_count_metrics,
    dice_score,
    fuzzy_zmf,
    iou_score,
    paper_q_seg,
)
from projects.illumination_robustness.evaluation.robustness import (
    RobustnessCurvePoint,
    compute_robustness_indices,
)


def test_fuzzy_zmf():
    # alpha=0, beta=1, s=2
    assert fuzzy_zmf(-0.5, s=2, alpha=0, beta=1) == 1.0
    assert fuzzy_zmf(0.0, s=2, alpha=0, beta=1) == 1.0
    assert fuzzy_zmf(0.5, s=2, alpha=0, beta=1) == 0.5
    assert fuzzy_zmf(1.0, s=2, alpha=0, beta=1) == 0.0
    assert fuzzy_zmf(1.5, s=2, alpha=0, beta=1) == 0.0

    # Monotonicity test
    thetas = np.linspace(0.0, 1.0, 20)
    vals = [fuzzy_zmf(t, s=2, alpha=0, beta=1) for t in thetas]
    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1]


def test_dice_and_iou_scores():
    # 1. Perfect match
    m1 = np.array([[1, 1], [0, 0]], dtype=bool)
    assert dice_score(m1, m1) == 1.0
    assert iou_score(m1, m1) == 1.0

    # 2. Disjoint
    m2 = np.array([[0, 0], [1, 1]], dtype=bool)
    assert dice_score(m1, m2) == 0.0
    assert iou_score(m1, m2) == 0.0

    # 3. Partial overlap:
    # m1 has 2 pixels; m3 has 2 pixels; intersection is 1 pixel
    m3 = np.array([[0, 1], [1, 0]], dtype=bool)
    # Dice = 2*1 / (2 + 2) = 0.5
    assert dice_score(m1, m3) == 0.5
    # IoU = 1 / (2 + 2 - 1) = 1/3
    assert pytest.approx(iou_score(m1, m3)) == 1.0 / 3.0

    # 4. Empty-mask semantics
    empty = np.zeros((2, 2), dtype=bool)
    # Both empty -> 1.0
    assert dice_score(empty, empty) == 1.0
    assert iou_score(empty, empty) == 1.0
    # Exactly one empty -> 0.0
    assert dice_score(m1, empty) == 0.0
    assert iou_score(m1, empty) == 0.0
    assert dice_score(empty, m1) == 0.0
    assert iou_score(empty, m1) == 0.0


def test_custom_count_metrics():
    # Perfect count
    err, qual = custom_count_metrics(10, 10)
    assert err == 0.0
    assert qual == 1.0

    # Miss by 5 out of 10
    err, qual = custom_count_metrics(5, 10)
    assert err == 0.5
    assert qual == 0.5

    # Huge over-segmentation: 30 vs 10 -> err = 2.0 -> qual = 0.0 (clamped)
    err, qual = custom_count_metrics(30, 10)
    assert err == 2.0
    assert qual == 0.0


def test_paper_q_seg_perfect_match():
    # 2 distinct GT blobs
    gt = np.zeros((10, 10), dtype=bool)
    gt[1:3, 1:3] = True  # blob 1
    gt[7:9, 7:9] = True  # blob 2

    res = paper_q_seg(gt, gt)
    assert res.n_gt_blobs == 2
    assert res.n_pred_blobs == 2
    assert res.n_matched_blobs == 2
    assert res.q1 == 0.0
    assert res.mu1 == 1.0
    assert res.q2 == 0.0
    assert res.mu2 == 1.0
    assert res.q_seg == 1.0


def test_paper_q_seg_sequential_matching_overlap_two():
    # GT has 2 separate blobs
    gt = np.zeros((10, 10), dtype=bool)
    gt[2, 2:4] = True  # blob 1 (2 pixels)
    gt[2, 6:8] = True  # blob 2 (2 pixels)

    # Predicted has 1 large blob bridging across both GT blobs
    pred = np.zeros((10, 10), dtype=bool)
    pred[2, 2:8] = True  # 6 pixels, overlaps both blob 1 and blob 2

    # Under S4 rule: max overlap with either blob is 2 pixels.
    # 2 pixels == (1/3)*6 -> threshold is > 1/3*Cu -> 2 is not > 2, so fails
    # Let's check with smaller bridge (pred size 5 pixels -> 2 > 5/3=1.67 -> matches!)
    pred5 = np.zeros((10, 10), dtype=bool)
    pred5[2, 2:7] = True  # 5 pixels: covers blob 1 (2 px) + blob 2 pixel (1 px)
    # max overlap is 2 > 5/3 -> matches!
    # On match, BOTH overlapping GT blobs are removed from active set
    res = paper_q_seg(pred5, gt)
    assert res.n_gt_blobs == 2
    assert res.n_pred_blobs == 1
    assert res.n_matched_blobs == 1
    # q1 = |2 - 1| / 2 = 0.5
    assert res.q1 == 0.5


def test_compute_artifact_level():
    # Pure bright white, noise level 1 (sigma=0)
    img_clean = np.full((10, 10), 255, dtype=np.uint8)
    art_clean = compute_artifact_level(img_clean, noise_index=1)
    assert art_clean < 0.01  # practically 0

    # Very dark, noise level 14 (sigma=sqrt(13*125)=40.3)
    img_distorted = np.full((10, 10), 0, dtype=np.uint8)
    art_distorted = compute_artifact_level(img_distorted, noise_index=14)
    assert art_distorted > 0.95  # practically 1


def test_robustness_indices():
    # 13 points with known values
    pts = [
        RobustnessCurvePoint(
            shading=b,
            noise=b + 1,
            artifact_level=b / 13.0,
            q_seg=0.5,
            dice=0.6,
            iou=0.4,
        )
        for b in range(1, 14)
    ]

    res = compute_robustness_indices(scene=1, points=pts)
    assert res.scene == 1
    assert pytest.approx(res.r_seg) == 0.5
    assert pytest.approx(res.r_dice) == 0.6
    assert pytest.approx(res.r_iou) == 0.4

    # Check validation of 13 points
    with pytest.raises(ValueError, match="Expected 13 points"):
        compute_robustness_indices(scene=1, points=pts[:10])


def test_evaluator_prediction():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 20:40] = True
    raw_annot = RawSceneAnnotation(
        scene=1,
        mask=mask,
        class_map=None,
        mask_path="dummy_mask",
        class_map_path=None,
    )
    gt = build_scene_ground_truth(raw_annot, GroundTruthConfig(target_shape=(100, 100)))

    # Evaluate exact prediction
    res = evaluate_prediction(
        prediction=mask,
        ground_truth=gt,
        image_id="r01_b01_n01",
        shading=1,
        noise=1,
        runtime_ms=12.5,
    )
    assert res.dice == 1.0
    assert res.iou == 1.0
    assert res.q_seg == 1.0
    assert res.runtime_ms == 12.5

    # Check to_csv_row schema
    row = res.to_csv_row(run_id="run_123", method="otsu")
    assert row["run_id"] == "run_123"
    assert row["method"] == "otsu"
    assert row["dice"] == 1.0
    assert row["status"] == "ok"
