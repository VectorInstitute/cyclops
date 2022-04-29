import pandas as pd
import torch
import torch.nn.functional as F
import logging
from utils_prediction.nn.metrics import (
    StandardEvaluator,
    relative_calibration_error,
)


class MetricLogger:
    """
    Handles metric logging during training
    """

    def __init__(
        self,
        metrics=None,
        threshold_metrics=None,
        thresholds=None,
        losses=None,
        phases=None,
        output_dict_keys=None,
        weighted_evaluation=False,
        evaluate_by_group=False,
        compute_group_min_max=False,
        disable_metric_logging=False,
        compute_relative_calibration_error=False,
    ):
        if metrics is None:
            metrics = ["auc", "auprc", "brier", "loss_bce"]

        if losses is None:
            losses = ["loss"]
        if phases is None:
            phases = ["train", "val"]

        if output_dict_keys is None:
            output_dict_keys = ["outputs", "pred_probs", "labels", "row_id"]

        self.metrics = metrics
        self.threshold_metrics = threshold_metrics
        self.thresholds = thresholds
        self.losses = losses
        self.phases = phases
        self.output_dict_keys = output_dict_keys
        self.weighted_evaluation = weighted_evaluation
        self.evaluate_by_group = evaluate_by_group
        self.compute_group_min_max = compute_group_min_max
        self.disable_metric_logging = disable_metric_logging
        self.compute_relative_calibration_error = compute_relative_calibration_error

        self.evaluator = StandardEvaluator(
            metrics=metrics, threshold_metrics=threshold_metrics, thresholds=thresholds
        )
        self.evaluation_dict_overall = self.init_evaluation_dict_overall()
        self.evaluation_df = pd.DataFrame({})

        self.init_metric_dicts()

    def init_evaluation_dict_overall(self):
        return {phase: [] for phase in self.phases}

    def init_metric_dicts(self):
        self.loss_dict = LossDict(metrics=self.losses)
        self.output_dict = OutputDict(keys=self.output_dict_keys)

    def compute_metrics_epoch(self, phase=None):
        if phase is None:
            raise ValueError("Must provide phase to compute_metrics_epoch")

        self.loss_dict.compute_losses()

        self.evaluation_df = self.get_loss_df(self.loss_dict.metric_dict)

        if not self.disable_metric_logging:
            self.output_dict.finalize_output_dict()
            output_df = self.get_output_df()
            metric_df = self.evaluator.evaluate(
                output_df,
                weight_var="weights" if self.weighted_evaluation else None,
                strata_vars=["group"] if self.evaluate_by_group else None,
            )
            self.evaluation_df = pd.concat([metric_df, self.evaluation_df])
            if self.compute_group_min_max and self.evaluate_by_group:
                self.evaluation_df = pd.concat(
                    [
                        self.evaluation_df,
                        self.compute_group_min_max_fn(self.evaluation_df),
                    ]
                )
            if self.compute_relative_calibration_error:
                self.evaluation_df = pd.concat(
                    [
                        self.evaluation_df,
                        self.get_relative_calibration_error(output_df),
                    ]
                )

        self.evaluation_dict_overall[phase].append(self.evaluation_df)

        if self.evaluate_by_group:
            result = self.evaluation_df.query("group.isnull()", engine='python')
        else:
            result = self.evaluation_df
        return dict(zip(result["metric"], result["performance"]))

    def compute_group_min_max_fn(self, df):
        """
        Computes the min and max of metrics across groups
        (TODO) Move this logic into an Evaluator class
        """
        result = (
            df.query("~group.isnull()", engine='python')
            .groupby("metric")[["performance"]]
            .agg(["min", "max"])
            .reset_index()
            .melt(id_vars="metric")
            .assign(metric=lambda x: x["metric"].str.cat(x["variable_1"], sep="_"))
            .rename(columns={"value": "performance"})
            .drop(columns=["variable_0", "variable_1"])
        )
        return result

    def get_relative_calibration_error(self, df):
        result = relative_calibration_error(
            labels=df.labels,
            pred_probs=df.pred_probs,
            group=df.group,
            metric_variant="abs",
            model_type="logistic",
            transform="log",
            compute_ace=False,
            return_models=False,
            return_calibration_density=False,
        )["result"]
        result = result.rename(columns={"relative_calibration_error": "performance"})
        result = result.assign(metric="relative_calibration_error")
        return result

    def get_output_dict(self):
        return self.output_dict.output_dict

    def get_output_df(self):
        return pd.DataFrame(
            {
                key: value[:, -1] if key == "outputs" else value
                for key, value in self.output_dict.output_dict.items()
            }
        )

    def get_loss_df(self, loss_dict):
        return pd.DataFrame(
            {"metric": list(loss_dict.keys()), "performance": list(loss_dict.values())}
        )

    def get_evaluation_overall(self):
        return (
            pd.concat(
                {
                    key: pd.concat(value, keys=range(len(value)))
                    for key, value in self.evaluation_dict_overall.items()
                }
            )
            .reset_index(level=-1, drop=True)
            .rename_axis(["phase", "epoch"])
            .reset_index()
        )

    def update_loss_dict(self, *args, **kwargs):
        self.loss_dict.update_loss_dict(*args, **kwargs)

    def update_output_dict(self, *args, **kwargs):
        if not self.disable_metric_logging:
            self.output_dict.update_output_dict(*args, **kwargs)

    def print_metrics(self):
        logging.info(self.evaluation_df)


class OutputDict:
    """
        Accumulates outputs over an epoch
    """

    def __init__(self, keys=None):
        self.init_output_dict(keys=keys)

    def init_output_dict(self, keys=None):
        if keys is None:
            keys = ["outputs", "pred_probs", "labels", "row_id"]

        if "pred_probs" not in keys:
            keys.append("pred_probs")

        self.output_dict = {key: [] for key in keys}

    def update_output_dict(self, **kwargs):
        kwargs["pred_probs"] = F.softmax(kwargs["outputs"], dim=1)[:, 1]
        for key, value in kwargs.items():
            if key in self.output_dict.keys():
                self.output_dict[key].append(value.detach().cpu())

    def finalize_output_dict(self):
        """
        Convert an output_dict to numpy
        """
        self.output_dict = {
            key: torch.cat(value, axis=0).numpy()
            for key, value in self.output_dict.items()
        }


class LossDict:
    """
        Accumulates loss over an epoch and aggregates results
    """

    def __init__(self, metrics=["loss"], init_value=0.0, mode="mean"):

        self.init_loss_dict(metrics=metrics, init_value=0.0)
        self.running_batch_size = 0
        self.mode = mode

    def init_loss_dict(self, metrics=None, init_value=None):
        """
        Initialize a dict of metrics
        """
        if metrics is None:
            metrics = [""]

        if init_value is None:
            init_value = 0.0

        self.metric_dict = {metric: init_value for metric in metrics}

    def update_loss_dict(self, update_dict, batch_size=None):
        if self.mode == "mean":
            self.running_batch_size += batch_size
        for key in self.metric_dict.keys():
            if self.mode == "mean":
                self.metric_dict[key] += update_dict[key].item() * batch_size
            else:
                self.metric_dict[key] += update_dict[key].item()

    def compute_losses(self):
        if self.mode == "mean":
            for key in self.metric_dict.keys():
                self.metric_dict[key] = self.metric_dict[key] / float(
                    self.running_batch_size
                )

    def print_metrics(self):
        """
        Print method
        """
        print(
            "".join([" {}: {:4f},".format(k, v) for k, v in self.metric_dict.items()])
        )