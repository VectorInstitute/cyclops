from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from mlflow import log_metrics

# TODO: perhaps we can add the wandb/mlflow blah blah blah
# metric here

# we can also register these and pull the right metric


@dataclass(init=False, eq=False)
class AverageBinaryClassificationMetric:

    def reset(self):
        self.loss_list = []
        self.preds = []
        self.targets = []

    def __init__(self):
        self.reset()

    def add_step(self, loss, logits, targets):
        """Note logits is pre-sigmoid
            All inputs should be tensors
        """
        preds = logits.detach().sigmoid().cpu().numpy()
        targets = targets.cpu().numpy()

        self.preds.append(preds >= 0.5)
        self.targets.append(targets)
        self.loss_list.append(loss.item())

    def compute_metrics(self):

        avg_loss = sum(self.loss_list) / len(self.loss_list) if len(
            self.loss_list) > 0 else 0

        preds = np.concatenate(self.preds).ravel()
        target = np.concatenate(self.targets).astype(preds.dtype).ravel()

        acc = (preds == target).mean()

        f1 = f1_score(target, preds)
        precision = precision_score(target, preds)
        recall = recall_score(target, preds)

        # MLflow metrics
        mlflow_metric_dict = {"epoch_loss": avg_loss, "accuracy": acc, "f1_score": f1, "precision": precision, "recall": recall}
        log_metrics(mlflow_metric_dict)

        return {
            "epoch_loss": avg_loss,
            "accuracy": acc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
        }
