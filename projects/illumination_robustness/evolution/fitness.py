"""Kartezio fitness adapter for the pilot-test objective."""

import numpy as np

from kartezio.core.components import Fitness, register
from projects.illumination_robustness.evaluation.metrics import paper_q_seg


@register(Fitness)
class PilotTestQSegFitness(Fitness):
    """Return ``1 - mean(Q_seg)`` for each individual in the batch."""

    def __init__(self, reduction="mean"):
        super().__init__(reduction)

    def evaluate(self, y_true, y_pred):
        losses = []
        for truth_item, prediction_item in zip(y_true, y_pred):
            truth = np.asarray(truth_item[0]) > 0
            prediction = np.asarray(prediction_item[0]) > 0
            if truth.shape != prediction.shape:
                raise ValueError("Prediction and ground truth shapes differ")
            score = paper_q_seg(prediction, truth).q_seg
            losses.append(1.0 - float(score))
        return np.asarray(losses, dtype=np.float32)
