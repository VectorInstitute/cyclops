"""Metrics used for model training/validation."""

from dataclasses import dataclass

import numpy as np
import torch
from mlflow import log_metrics
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


@dataclass(init=False, eq=False)
class AverageBinaryClassificationMetric:
    """Metrics for binary classification tasks.

    Attributes
    ----------
    loss: list
        List to track loss.
    preds: list
        List of predictions.
    targets: list
        List of targets.
    """

    def reset(self):
        """Reset the tracked loss, prediction and target containers."""
        self.loss_list = []
        self.preds = []
        self.targets = []

    def __init__(self):
        """Instantiate."""
        self.reset()

    def add_step(self, loss: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor):
        """Add loss, predictions and targets during a step of training/validation.

        Parameters
        ----------
        loss: torch.Tensor
            Loss for current step.
        logits: torch.Tensor
            Prediction at current step.
        targets: torch.Tensor
            Classification targets at current step.
        """
        preds = logits.detach().sigmoid().cpu().numpy()
        targets = targets.cpu().numpy()

        self.preds.append(preds >= 0.5)
        self.targets.append(targets)
        if loss:
            self.loss_list.append(loss.item())

    def compute_metrics(self):
        """Compute metrics for current step, logs them to MLFlow.

        Returns
        -------
        dict
            Computed metrics (average loss, accuracy, f1 score,\
                    precision, recall and AUC).
        """
        avg_loss = (
            sum(self.loss_list) / len(self.loss_list) if len(self.loss_list) > 0 else 0
        )

        preds = np.concatenate(self.preds).ravel()
        target = np.concatenate(self.targets).astype(preds.dtype).ravel()

        acc = (preds == target).mean()

        # Metrics.
        metrics_dict = {
            "epoch_loss": avg_loss,
            "accuracy": acc,
            "f1_score": f1_score(target, preds),
            "precision": precision_score(target, preds),
            "recall": recall_score(target, preds),
            "AUC": roc_auc_score(target, preds),
        }
        log_metrics(metrics_dict)

        return metrics_dict
