"""Evaluator class."""
from typing import Dict, NamedTuple, Sequence, Union, get_args

from models.wrappers import WrappedModel

from cyclops.evaluate.metrics.metric import Metric, MetricCollection


class Evaluator:
    """Evaluate one or more models on a dataset.

    warning:: This class is experimental and will change in the future.

    Parameters
    ----------
    models : Union[WrappedModel, Sequence[WrappedModel], Dict[str, WrappedModel]]
        Model(s) to evaluate.
    data : NamedTuple
        Dataset to evaluate the model(s) on.
    metrics : Union[Metric, Sequence[Metric], Dict[str, Metric], MetricCollection]
        Metric(s) to use for evaluation.

    """

    def __init__(
        self,
        model: Union[WrappedModel, Sequence[WrappedModel], Dict[str, WrappedModel]],
        data: NamedTuple,
        metrics: Union[Metric, Sequence[Metric], Dict[str, Metric], MetricCollection],
    ):
        self.model = self.prepare_model(model)
        self.data = data
        self.metrics = self.prepare_metric(metrics)

    def prepare_model(
        self,
        model: Union[WrappedModel, Sequence[WrappedModel], Dict[str, WrappedModel]],
    ) -> Dict[str, WrappedModel]:
        """Prepare the model(s) for evaluation.

        Parameters
        ----------
        model : Union[WrappedModel, Sequence[WrappedModel], Dict[str, WrappedModel]]
            Model(s) to evaluate.

        Returns
        -------
        model : Dict[str, WrappedModel]
            Model(s) to evaluate.

        """
        if isinstance(model, get_args(WrappedModel)):
            model = {model.model_.__class__.__name__: model}
        elif isinstance(model, (list, tuple)):
            assert all(isinstance(m, get_args(WrappedModel)) for m in model)
            model = {m.getattr("model_").__class__.__name__: m for m in model}
        elif isinstance(model, dict):
            assert all(isinstance(m, get_args(WrappedModel)) for m in model.values())
        else:
            raise ValueError(f"Invalid model type: {type(model)}")

        return model

    def prepare_metric(
        self,
        metrics: Union[Metric, Sequence[Metric], Dict[str, Metric], MetricCollection],
    ) -> MetricCollection:
        """Prepare the metric(s) for evaluation.

        A MetricCollection object is returned, which is a dict-like object that
        can be used to compute the metric(s) on a set of predictions.

        Parameters
        ----------
        metrics : Union[Metric, Sequence[Metric], Dict[str, Metric], MetricCollection]
            Metric(s) to use for evaluation.

        Returns
        -------
        MetricCollection
            Metric(s) to use for evaluation.

        """
        if isinstance(metrics, (Metric, Sequence, Dict)):
            metrics = MetricCollection(metrics)
        elif isinstance(metrics, MetricCollection):
            pass
        else:
            raise ValueError(f"Invalid metric type: {type(metrics)}")

        return metrics

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """Evaluate the model(s) on the dataset.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary containing the evaluation results.

        """
        predictions = {}
        for model_name, model in self.model.items():
            # TODO: Add support for predict_proba kwargs
            predictions[model_name] = model.predict_proba(self.data.features)

        # XXX: post-process predictions?

        # TODO: create slices of the data

        # compute metrics for each dataset slice and model
        scores = {}
        for model_name, model_predictions in predictions.items():
            scores[model_name] = self.metrics(self.data.target, model_predictions)

        return scores

    # TODO: function to compute confidence intervals for the metric(s)
