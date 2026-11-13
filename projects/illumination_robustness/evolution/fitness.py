"""Kartezio fitness adapter for the pilot-test objective."""

import time

import numpy as np

from kartezio.core.components import Fitness, register
from projects.illumination_robustness.evaluation.metrics import paper_q_seg
from projects.illumination_robustness.evaluation.metrics import dice_score, custom_count_metrics
from projects.illumination_robustness.evaluation.object_metrics import label_blobs


@register(Fitness)
class PilotTestQSegFitness(Fitness):
    """Return ``1 - mean(Q_seg)`` for each individual in the batch."""

    def __init__(self, reduction="mean"):
        super().__init__(reduction)
        self.total_evaluation_seconds = 0.0
        self.evaluation_calls = 0
        self._ground_truth_cache = {}

    def evaluate(self, y_true, y_pred):
        started = time.perf_counter()
        losses = []
        for truth_item, prediction_item in zip(y_true, y_pred):
            truth = np.asarray(truth_item[0]) > 0
            prediction = np.asarray(prediction_item[0]) > 0
            if truth.shape != prediction.shape:
                raise ValueError("Prediction and ground truth shapes differ")
            # Ground-truth components are invariant across candidates and
            # generations. Cache them once per training-mask object instead of
            # relabelling the same full-resolution mask for every prediction.
            cache_key = (id(truth_item[0]), truth.shape)
            cached = self._ground_truth_cache.get(cache_key)
            if cached is None:
                gt_labels, n_gt = label_blobs(truth, connectivity=8)
                cached = (gt_labels, n_gt)
                self._ground_truth_cache[cache_key] = cached
            gt_labels, n_gt = cached
            score = paper_q_seg(
                prediction,
                truth,
                gt_blob_labels=gt_labels,
                n_gt_blobs=n_gt,
                connectivity=8,
            ).q_seg
            losses.append(1.0 - float(score))
        self.total_evaluation_seconds += time.perf_counter() - started
        self.evaluation_calls += 1
        return np.asarray(losses, dtype=np.float32)


@register(Fitness)
class HybridDiceCountFitness(Fitness):
    """Fast training loss: 0.8 Dice loss plus 0.2 object-count error."""

    def __init__(self, dice_weight=0.8, count_weight=0.2, reduction="mean"):
        super().__init__(reduction)
        self.dice_weight = float(dice_weight)
        self.count_weight = float(count_weight)
        self.total_evaluation_seconds = 0.0
        self.evaluation_calls = 0
        self._ground_truth_cache = {}

    def evaluate(self, y_true, y_pred):
        started = time.perf_counter()
        losses = []
        for truth_item, prediction_item in zip(y_true, y_pred):
            truth = np.asarray(truth_item[0]) > 0
            prediction = np.asarray(prediction_item[0]) > 0
            if truth.shape != prediction.shape:
                raise ValueError("Prediction and ground truth shapes differ")
            dice_loss = 1.0 - float(dice_score(prediction, truth))
            cache_key = (id(truth_item[0]), truth.shape)
            cached = self._ground_truth_cache.get(cache_key)
            if cached is None:
                gt_labels, gt_count = label_blobs(truth, connectivity=8)
                cached = (gt_labels, gt_count)
                self._ground_truth_cache[cache_key] = cached
            _, gt_count = cached
            _, pred_count = label_blobs(prediction, connectivity=8)
            count_error, _ = custom_count_metrics(pred_count, gt_count)
            losses.append(self.dice_weight * dice_loss + self.count_weight * count_error)
        self.total_evaluation_seconds += time.perf_counter() - started
        self.evaluation_calls += 1
        return np.asarray(losses, dtype=np.float32)
